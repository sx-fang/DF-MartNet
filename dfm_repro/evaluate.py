"""Compute-worker evaluation; every run uses its own learned control paths.

The caller must pass the torch module returned by gpu_worker.gpu_check().
Importing this module alone never imports torch or executes numerical work.
"""
from pathlib import Path
import math
from .common import ROOT,json_read,json_write,read_csv,write_csv,sha256
from .protocol import HJB_CASES,evaluation_counts,compare_own_policy,require_complete_evaluation

def evaluate(spec_path,torch,config,use_source):
    import numpy as np
    plan=json_read(spec_path);arms=plan['arms'];case=plan['case'];kind=plan['kind']
    directory=Path(plan['case_dir']);out=directory/'evaluation';source_root=Path(plan['source_root'])
    if len({a['source'] for a in arms})!=1:raise ValueError('One source version is required per evaluation worker')
    source=source_root/arms[0]['source'];solver=use_source(source)
    import utils
    counts=evaluation_counts(case,{'kind':kind,'arms':arms})
    torch.set_default_device('cuda:0');torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    cfg=config(directory/arms[0]['id']/'actual.ini');problem=solver.parse_problem(cfg,use_dist=False)
    d=problem.dim_x;N=round((problem.te-problem.t0)/problem.dt)
    if N!=32:raise ValueError('The submitted deployment/profile protocol requires 32 time steps')

    def load(arm):
        acfg=config(directory/arm['id']/'actual.ini')
        for section in ('Network',):
            if dict(acfg.items(section))!=dict(cfg.items(section)):
                raise ValueError('Network architecture differs within a prescribed cohort')
        if acfg.get('Example','name')!=cfg.get('Example','name') or acfg.getint('Example','dim_x')!=d:
            raise ValueError('Different physical model within one evaluation cohort')
        nets,_,_,_=solver.parse_nets(acfg,problem,use_dist=False)
        weights=directory/arm['id']/'training/selected_r0_weights.pt'
        state=torch.load(weights,map_location='cpu',weights_only=True)
        for name in ('u_alpha','v_theta'):
            nets[name].load_state_dict(state[name]);nets[name].eval()
            for module in nets[name].modules():
                if hasattr(module,'enable_autocast'):module.enable_autocast=False
        return nets['v_theta'],nets['u_alpha'],sha256(weights)

    def batched(fun,x,block=16):
        values=[]
        with torch.no_grad():
            for lo in range(0,len(x),block):values.append(fun(x[lo:lo+block]).double().cpu())
        result=torch.cat(values).numpy()
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite prediction/reference')
        return result

    def origin(m):
        x=torch.zeros((m,d));x[:,0]=float(problem.t0);x[:,1:]=float(problem.z0)
        return x

    def walk(control,start,seed,save_states=False):
        gen=torch.Generator(device='cuda:0');gen.manual_seed(seed)
        x=start.clone();acc=torch.zeros((len(x),1));layers=[]
        with torch.no_grad():
            for _ in range(N):
                if save_states:layers.append(x.cpu().numpy())
                u=control(x)
                if not torch.isfinite(u).all():raise RuntimeError('Nonfinite feedback')
                acc+=problem.running_cost(torch.zeros_like(u[...,:1]),x,u)*problem.dt
                dw=torch.normal(0.,math.sqrt(problem.dt),x[:,1:].shape,generator=gen,device=x.device)
                x=problem.system_step(x,u,dw=dw)
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite deployment state')
            acc+=problem.terminal_cost(problem.project_onto_boundary(x))
        costs=acc.double().cpu().numpy().reshape(-1)
        if not np.isfinite(costs).all():raise RuntimeError('Nonfinite cost')
        return costs,layers

    def stats(values):
        mean=math.fsum(map(float,values))/len(values)
        sd=math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/(len(values)-1))
        return mean,sd,sd/math.sqrt(len(values))

    def relative(pred,truth):
        denominator=float(np.abs(truth).sum())
        if denominator<=0:raise ValueError('Zero reference norm in relative error')
        return float(np.abs(pred-truth).sum()/denominator)

    s0,x0=utils.t_diag_curve(d,100,problem.s_range,False)
    s1,x1=utils.t_manifold_curve(d,100,problem.s_range,False)
    grid=torch.cat([x0,x1]);torch.manual_seed(20260918)
    ref_grid=batched(problem.v,grid)
    np.savez(out/'grid_reference.npz',x=grid.cpu().numpy(),v_true=ref_grid,
             M_mc=int(problem.M_mc),seed=20260918)
    profile={};curve_x={}
    for name,fn in problem.x0_curves.items():
        s,x=fn(d,101,problem.s_range,False);curve_x[name]=x
        profile['s_'+name]=s.cpu().numpy();profile['v_true_'+name]=batched(problem.v,x).reshape(-1)
    # A frozen plotting reference is a public input, not a fresh measurement.
    # Accuracy/path references above and below are evaluated for this run.
    contour=ROOT/'data/archived/hjb_table'/(case+'_t0.csv')
    if case in ['hjb1_d10000','hjb2_d10000','hjb3_d10000']:
        rs=read_csv(contour)
        profile.update(s_true_wide=np.array([float(r['s']) for r in rs]),
            v_true_wide=np.array([float(r['value']) for r in rs]))
        terminal=read_csv(ROOT/'data/archived/hjb_table'/(case+'_terminal.csv'))
        profile.update(s_terminal_wide=np.array([float(r['s']) for r in terminal]),
            v_terminal_wide=np.array([float(r['value']) for r in terminal]))
    epsilon_meta=[]
    if kind=='eps':
        epsilon_meta=[r for r in read_csv(ROOT/'data/archived/epsilon_sweep/per_seed_profiles.csv') if r['case']==case]
    result=[];path_pins=[]
    for ai,arm in enumerate(arms):
        value,control,weight_sha=load(arm);tag=arm['id']
        pred_grid=batched(value,grid)
        for name,x in curve_x.items():profile['v_pred_'+name+'_'+tag]=batched(value,x).reshape(-1)
        torch.manual_seed(counts['reference_seed'])
        value0=float(batched(problem.v,origin(1)).reshape(-1)[0])
        costs,_=walk(control,origin(counts['cost_paths']),20260821+ai)
        mean,sd,se=stats(costs)
        row={'arm':tag,'seed':arm['seed'],'repeat':0,
             're_t0_s0s1':relative(pred_grid,ref_grid),'jhat_origin_fp32':mean,
             'jhat_mc_sd':sd,'jhat_mc_se':se,'v_reference_origin':value0,
             'rc_signed_fp32':(mean-value0)/abs(value0),'M_cost':counts['cost_paths'],
             'weights_sha256':weight_sha,'reference_semantics':'eps0 comparator' if kind=='eps' else 'native reference'}
        np.savez(out/(tag+'_grid.npz'),x=grid.cpu().numpy(),v_pred=pred_grid,v_true=ref_grid)
        np.savez(out/(tag+'_cost.npz'),per_path_cost=costs,M=counts['cost_paths'],seed=20260821+ai)
        if case in HJB_CASES:
            def draw(policy):
                _,layers=walk(policy,origin(counts['deployment_paths']),counts['path_seed'],True)
                return np.concatenate(layers)
            def reference(states):
                torch.manual_seed(counts['reference_seed'])
                return batched(problem.v,torch.from_numpy(states).cuda())
            states,truth,pred=compare_own_policy(control,draw,reference,
                lambda x:batched(value,torch.from_numpy(x).cuda()))
            row['re_path']=relative(pred,truth)
            np.savez(out/(tag+'_path.npz'),x=states,v_pred=pred,v_true=truth,
                     n_paths=counts['deployment_paths'],n_levels=N,weights_sha256=weight_sha)
            m=counts['deployment_paths'];layers=[]
            for k in range(N):
                sl=slice(k*m,(k+1)*m)
                layers.append(dict(layer=k,t=k*problem.dt,n=m,re_v=relative(pred[sl],truth[sl])))
            write_csv(out/(tag+'_layers.csv'),layers)
            path_pins.append(dict(arm=tag,weights_sha256=weight_sha,
                paired_path_sha256=sha256(out/(tag+'_path.npz')),control='own learned feedback'))
        if kind=='k1':
            _,layers=walk(control,origin(counts['deployment_paths']),20260920,True)
            states=torch.from_numpy(np.concatenate(layers)).cuda()
            vt=batched(problem.v,states);vp=batched(value,states)
            ut=batched(problem.u_star,states);up=batched(control,states)
            row.update(re_path=relative(vp,vt),re_u_path=relative(up,ut))
            # Path RE includes t0. The historical QQ pool retains its 31
            # interior layers and therefore gets separate diagnostic fields.
            m=counts['deployment_paths']
            np.savez(out/(tag+'_path.npz'),v_true=vt,v_pred=vp,u_true=ut,u_pred=up,
                     n_paths=m,n_levels=N,weights_sha256=weight_sha)
            np.savez(out/(tag+'_qq.npz'),t=states[m:,0].cpu().numpy(),
                     v_true=vt[m:],v_pred=vp[m:],u_true=ut[m:],u_pred=up[m:])
        if kind=='eps':
            saved={int(r['point']):r for r in epsilon_meta if r['arm']==tag}
            if set(saved)!=set(range(101)):raise ValueError('Missing prescribed epsilon evaluation seeds')
            sp,xp=utils.t_diag_curve(d,101,problem.s_range,False);scan=[];vectors=[]
            for point in range(101):
                meta=saved[point];seed=int(meta['seed']);M=int(meta['M'])
                if M!=counts['cost_paths'] or int(meta['n_levels'])!=N:
                    raise ValueError('Saved profile budgets differ from the submitted protocol')
                c,_=walk(control,xp[point:point+1].repeat(M,1),seed)
                mean,sd,se=stats(c);vectors.append(c)
                pred=float(batched(value,xp[point:point+1]).reshape(-1)[0])
                scan.append(dict(s=float(sp[point]),v_pred=pred,jhat=mean,jhat_std=sd,jhat_se=se,
                    gap=mean-pred,seed=seed,M=M,v_ref=float(profile['v_true_diag'][point])))
            write_csv(out/(tag+'_eps_cost.csv'),scan)
            np.savez(out/(tag+'_eps_costs.npz'),costs=np.stack(vectors),s=sp.cpu().numpy(),M=counts['cost_paths'])
        result.append(row);del value,control;torch.cuda.empty_cache()
    require_complete_evaluation(arms,result)
    write_csv(out/'per_seed_eval.csv',result);np.savez(out/'profiles.npz',**profile)
    json_write(out/'protocol.json',dict(kind=kind,source=str(source),counts=counts,
        path_N=N,path_control='each run own learned feedback',path_error_layers='all preterminal layers including t0',
        dtype='float32 networks/dynamics; float64 statistics',path_inputs=path_pins,
        wide_contour_reference_sha256=sha256(contour) if contour.is_file() else None,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
