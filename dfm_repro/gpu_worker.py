"""Compute-node adapters. No numerical imports occur on import or --help.

The solver computations are unchanged; publication comments are curated.
The training adapter selects the
paper's r0 and captures final weights for old solvers without a save hook.
Evaluation uses the selected source's native equations and reference routines.
"""
from __future__ import annotations
import argparse
import configparser
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
if __package__ in ('', None):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dfm_repro.common import json_read, json_write, sha256, write_csv, expression

def gpu_check(count=1):
    if sys.platform!='linux':raise RuntimeError('Numerical work is disabled on Windows and macOS')
    if not os.environ.get('SLURM_JOB_ID') and os.environ.get('DFM_COMPUTE_NODE')!='1':
        raise RuntimeError('Use a Slurm compute allocation, or set DFM_COMPUTE_NODE=1 on a dedicated compute node. Login-node execution is forbidden.')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<count:
        raise RuntimeError(f'{count} CUDA GPUs required; no CPU or reduced-GPU fallback')
    return torch

def config(path):
    cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    if not cfg.read(path):raise FileNotFoundError(path)
    return cfg

def use_source(path):
    # One historical source per worker; switching sources in a live interpreter
    # would retain imported modules and is deliberately unsupported.
    sys.path.insert(0,str(Path(path).resolve()))
    import solver
    if Path(solver.__file__).resolve().parent!=Path(path).resolve():raise RuntimeError('Wrong solver source imported')
    return solver

def train(args):
    arm=json.loads(args.arm_json);torch=gpu_check(arm['gpus']);source=Path(args.source)
    solver=use_source(source);import runtask
    cfg=config(args.config);rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);dev=torch.device('cuda',local)
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    if distributed:runtask.init_distributed(dev)
    # The native entry initialises DDP before set_seed, which adds global rank.
    torch.set_default_device(dev);runtask.set_torchdtype(cfg);runtask.set_seed(cfg)
    work=Path(args.work);(work/'training').mkdir(exist_ok=True)
    captured={};native_parse=solver.parse_nets;native_curve=solver.res_on_curve
    native_save=getattr(solver,'_save_weights',None);save_seconds=[]
    def capture(*a,**kw):
        result=native_parse(*a,**kw);captured.update(result[0]);return result
    solver.parse_nets=capture
    # Output-only adapter: immediately after final training, before native curve
    # output. No loss, optimiser, random stream or training timer is changed.
    def curve(*a,**kw):
        if rank==0 and not (work/'training/selected_r0_weights.pt').is_file():
            started=time.perf_counter()
            dest=work/'training/selected_r0_weights.pt'
            state={name:{k:v.detach().cpu() for k,v in getattr(captured[name],'module',captured[name]).state_dict().items()}
                   for name in ('v_theta','u_alpha')}
            torch.save(state,dest)
            save_seconds.append(time.perf_counter()-started)
        return native_curve(*a,**kw)
    solver.res_on_curve=curve
    if native_save is not None:
        def save(*a,**kw):
            started=time.perf_counter();result=native_save(*a,**kw)
            save_seconds.append(time.perf_counter()-started);return result
        solver._save_weights=save
    try:
        solver.solve(cfg,distributed,sav_name='selected_r0_')
        if distributed:torch.distributed.barrier()
        if rank==0:
            weights=work/'training/selected_r0_weights.pt'
            if not weights.is_file():raise RuntimeError('Final weight export missing')
            json_write(work/'training_record.json',{'state':'complete','seed':arm['seed'],'repeat':0,'declared_repeat_time':arm['original_repeat_time'],
                 'selected_repeat_policy':'r0 only; no reseeding/replacement','source':str(source),
                 'config_sha256':sha256(args.config),'weights':{'bytes':weights.stat().st_size,'sha256':sha256(weights)},
                 'max_iter':arm['max_iter'],'expected_rows':arm['max_iter']+1,'host':socket.gethostname(),
                 'jobid':os.environ.get('SLURM_JOB_ID'),'world_size':arm['gpus'],'gpu':torch.cuda.get_device_name(local),
                 'weight_export_seconds':sum(save_seconds),'weight_export_clock':'separate from native rt_solve and rt_log',
                 'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'dtype':cfg.get('Environment','torch_dtype'),
                 'completed_utc':datetime.now(timezone.utc).isoformat()})
    finally:
        solver.parse_nets=native_parse;solver.res_on_curve=native_curve
        if native_save is not None:solver._save_weights=native_save
        if distributed:torch.distributed.destroy_process_group()

def official(args):
    import subprocess
    arm=json.loads(args.arm_json);torch=gpu_check(arm['gpus']);work=Path(args.work);out=work/'training';out.mkdir(exist_ok=True)
    cmd=[sys.executable,'-B',str(Path(args.source)/'run.py')]+arm['cli']+['--out',str(out),'--tag','selected_r0']
    subprocess.run(cmd,check=True)
    json_write(work/'training_record.json',{'state':'complete','cli':cmd,'jobid':os.environ.get('SLURM_JOB_ID'),
               'source':str(args.source),'seed':arm['seed'],'repeat':0,'clock':'native rt; keep distinct from DF rt_solve/rt_log',
               'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(0),'world_size':arm['gpus'],
               'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'dtype':'float32'})

def evaluate(args):
    torch=gpu_check();import numpy as np
    plan=json_read(args.spec);arms=plan['arms'];case=plan['case'];directory=Path(plan['case_dir']);out=directory/'evaluation'
    source_root=Path(plan['source_root'])
    first=arms[0]
    path_arms=arms[1:] if plan['version']=='paper_original' and case in ('hjb2_d10000','hjb3_d2000_w2010') else arms
    guide=path_arms[0]
    src=source_root/('corrected' if plan['version']=='corrected' else guide['source'])
    solver=use_source(src);import utils
    torch.set_default_device('cuda:0');torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    cfg=config(directory/guide['id']/'actual.ini');problem=solver.parse_problem(cfg,use_dist=False)
    d=problem.dim_x;N=int(round((problem.te-problem.t0)/problem.dt))

    def load(arm):
        acfg=config(directory/arm['id']/'actual.ini')
        # Model architecture is built by the guide source; network SHA/shape and
        # architectural INI keys must match. Training itself uses each arm's source.
        nets,_,_,_=solver.parse_nets(acfg,problem,use_dist=False)
        state=torch.load(directory/arm['id']/'training/selected_r0_weights.pt',map_location='cpu',weights_only=True)
        for key in ('u_alpha','v_theta'):
            nets[key].load_state_dict(state[key]);nets[key].eval()
            mod=getattr(nets[key],'module',nets[key])
            if hasattr(mod,'enable_autocast'):mod.enable_autocast=False
        return nets['v_theta'],nets['u_alpha']

    def batched(fun,x,block=16):
        pieces=[]
        with torch.no_grad():
            for lo in range(0,len(x),block):pieces.append(fun(x[lo:lo+block]).double().cpu())
        result=torch.cat(pieces).numpy()
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite prediction/reference')
        return result

    def origin(m):
        x=torch.zeros((m,d));x[:,0]=float(problem.t0);x[:,1:]=float(getattr(problem,'z0',0.0));return x

    def cost_walk(u,start,seed,save_states=False):
        gen=torch.Generator(device='cuda:0');gen.manual_seed(seed)
        x=start.clone();acc=torch.zeros((len(x),1));states=[]
        with torch.no_grad():
            for k in range(N):
                if save_states:states.append(x.cpu().numpy())
                ux=u(x)
                if not torch.isfinite(ux).all():raise RuntimeError('Nonfinite feedback')
                acc=acc+problem.running_cost(torch.zeros_like(ux[..., [0]]),x,ux)*problem.dt
                dw=torch.normal(0.,math.sqrt(problem.dt),x[:,1:].shape,generator=gen,device=x.device)
                x=problem.system_step(x,ux,dw=dw)
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite trajectory')
            acc=acc+problem.terminal_cost(problem.project_onto_boundary(x))
        costs=acc.double().cpu().numpy().reshape(-1)
        if not np.isfinite(costs).all():raise RuntimeError('Nonfinite cost samples')
        return costs,states

    def stable_cost(cost):
        mean=math.fsum(map(float,cost))/len(cost)
        variance=math.fsum((float(v)-mean)**2 for v in cost)/(len(cost)-1)
        return mean,math.sqrt(variance),math.sqrt(variance/len(cost))

    # S0 and S1 retain the source's curve definitions; 100+100 equispaced points
    # are additional diagnostics. Native final log RE is preserved separately.
    s0,x0=utils.t_diag_curve(d,100,problem.s_range,False)
    s1,x1=utils.t_manifold_curve(d,100,problem.s_range,False)
    grid=torch.cat([x0,x1]);torch.manual_seed(20260918)
    ref_grid=batched(problem.v,grid)
    # The plotted profile uses the original 101-point curves.
    curves=list(problem.x0_curves.items());profile={};curve_x={}
    for name,fn in curves:
        s,x=fn(d,101,problem.s_range,False);curve_x[name]=x
        profile['s_'+name]=s.cpu().numpy();profile['v_true_'+name]=batched(problem.v,x).reshape(-1)
    if plan['kind']=='hjb':
        # Figure 2's wide true contours used their own documented generator:
        # seed 20260914, 101 points [-2,2]; HJB1 d=1e4/M=2e5, HJB2/3
        # d=2000/M=1000. Retain that reference protocol, not the log grid.
        s=np.linspace(-2.,2.,101);rng=np.random.default_rng(20260914)
        if case=='hjb1_d10000':
            dim=10000;M=200000;chunk=5000;acc=np.zeros_like(s)
            for _ in range(M//chunk):
                b=rng.standard_normal((chunk,dim));r1=b.sum(axis=1);r2=np.einsum('ij,ij->i',b,b)
                q=dim*s[:,None]**2+2*np.sqrt(2)*s[:,None]*r1[None,:]+2*r2[None,:]
                acc+=(2/(1+q)).sum(axis=1)
            v0=-np.log(acc/M);vt=np.log(.5*(1+dim*s**2))
        else:
            # Advance the same shared stream through the HJB1 contour draws.
            for _ in range(200000//5000):rng.standard_normal((5000,10000))
            dim=2000;M=1000;chunk=250;delc=.3/np.pi
            def h(y):return np.sin(y-np.pi/2)+np.sin(1/(delc+y**2))
            for delta in ([.2] if case=='hjb2_d10000' else [.2,.1]):
                acc=np.zeros_like(s)
                for _ in range(M//chunk):
                    z=np.sqrt(2)*delta*rng.standard_normal((chunk,dim))
                    for k,sv in enumerate(s):acc[k]+=np.exp(-h(sv+z).mean(axis=1)).sum()
                v0=-np.log(acc/M)
            vt=h(s-1)
        profile['s_true_wide']=s;profile['v_true_wide']=v0;profile['v_terminal_wide']=vt

    # Table path protocol: 128 origin-started guide paths, all N preterminal
    # layers, seed 20260821, shared forced-M=1e6 reference, strict fp32 nets.
    states=None;ref_path=None
    if plan['kind']=='hjb':
        gv,gu=load(guide);costs,layers=cost_walk(gu,origin(128),20260821,True)
        states=np.concatenate(layers);m_ini=getattr(problem,'M_mc',None)
        problem.M_mc=1000000;torch.manual_seed(20260821)
        ref_path=batched(problem.v,torch.from_numpy(states).cuda())
        if m_ini is not None:problem.M_mc=m_ini
        np.savez(out/'shared_path.npz',x=states,v_true=ref_path,guide_costs=costs,n_paths=128,n_levels=N,seed=20260821,M_mc=1000000)
    np.savez(out/'grid_reference.npz',x=grid.cpu().numpy(),v_true=ref_grid,M_mc=int(getattr(problem,'M_mc',0)),seed=20260918)

    results=[]
    for ai,arm in enumerate(arms):
        v,u=load(arm);prefix=arm['id']
        pred_grid=batched(v,grid);delta=np.abs(pred_grid-ref_grid)
        re_grid=float(delta.sum()/np.abs(ref_grid).sum())
        for name,x in curve_x.items():profile['v_pred_'+name+'_'+prefix]=batched(v,x).reshape(-1)
        value0=float(batched(problem.v,origin(1)).reshape(-1)[0])
        costs,_=cost_walk(u,origin(int(problem.M_rc)),20260821+ai)
        jhat,std,se=stable_cost(costs)
        row={'arm':prefix,'seed':arm['seed'],'repeat':0,'re_t0_s0s1':re_grid,
             'jhat_origin_fp32':jhat,'jhat_mc_sd':std,'jhat_mc_se':se,'v_reference_origin':value0,
             'rc_signed_fp32':(jhat-value0)/abs(value0),'M_cost':int(problem.M_rc),'reference_semantics':'eps0 reference' if plan['kind']=='eps' else 'native reference'}
        np.savez(out/(prefix+'_grid.npz'),v_pred=pred_grid,v_true=ref_grid,x=grid.cpu().numpy())
        np.savez(out/(prefix+'_cost.npz'),per_path_cost=costs,M=int(problem.M_rc),seed=20260821+ai)
        if states is not None and arm in path_arms:
            pred=batched(v,torch.from_numpy(states).cuda());row['re_path']=float(np.abs(pred-ref_path).sum()/np.abs(ref_path).sum())
            np.savez(out/(prefix+'_path.npz'),v_pred=pred,v_true=ref_path,n_paths=128,n_levels=N)
            layer_rows=[]
            for k in range(N):
                sl=slice(k*128,(k+1)*128)
                layer_rows.append({'layer':k,'t':k*problem.dt,'n':128,'re_v':float(np.abs(pred[sl]-ref_path[sl]).sum()/np.abs(ref_path[sl]).sum())})
            write_csv(out/(prefix+'_layers.csv'),layer_rows)
        if plan['kind']=='k1':
            _,layers=cost_walk(u,origin(int(problem.M_rc)),20260920,True)
            # Only interior times enter the original QQ pool.
            xqq=torch.from_numpy(np.concatenate(layers[1:])).cuda()
            vt=batched(problem.v,xqq);vp=batched(v,xqq)
            ut=batched(problem.u_star,xqq);up=batched(u,xqq)
            np.savez(out/(prefix+'_qq.npz'),t=xqq[:,0].cpu().numpy(),v_true=vt,v_pred=vp,u_true=ut,u_pred=up)
            row['re_path']=float(np.abs(vp-vt).sum()/np.abs(vt).sum())
            row['re_u_path']=float(np.abs(up-ut).sum()/np.abs(ut).sum())
        if plan['kind']=='eps':
            tag_index={'eps1':0,'eps0p5':1,'eps0p25':2,'eps0p125':3,'eps0':4}[case]
            sp,xp=utils.t_diag_curve(d,101,problem.s_range,False);scan=[];cost_vectors=[]
            for point in range(len(xp)):
                seed=20260921+10000*tag_index+point
                c,_=cost_walk(u,xp[point:point+1].repeat(4096,1),seed)
                mean,sd,stderr=stable_cost(c);cost_vectors.append(c)
                pred=float(batched(v,xp[point:point+1]).reshape(-1)[0])
                scan.append({'s':float(sp[point]),'v_pred':pred,'jhat':mean,'jhat_std':sd,'jhat_se':stderr,'band2se':2*stderr,'gap':mean-pred,'seed':seed,'M':4096})
            write_csv(out/(prefix+'_eps_cost.csv'),scan)
            np.savez(out/(prefix+'_eps_costs.npz'),costs=np.stack(cost_vectors),s=sp.cpu().numpy(),M=4096)
        results.append(row);del v,u;torch.cuda.empty_cache()
    write_csv(out/'per_seed_eval.csv',results)
    np.savez(out/'profiles.npz',**profile)
    json_write(out/'protocol.json',{'kind':plan['kind'],'source':str(src),
          'grid':'100 S0 + 100 S1; native log RE separately','profile_points':101,'path_N':N,'path_M':128 if states is not None else None,
          'path_reference_M':1000000 if states is not None else None,'dtype':'float32 nets/dynamics; float64 statistics',
          'cost_statistics':'centred two-pass sample SD / sqrt(M); no second-moment subtraction',
          'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'jobid':os.environ.get('SLURM_JOB_ID'),
          'reference_cache_inputs':{'guide_weights_sha256':sha256(directory/guide['id']/'training/selected_r0_weights.pt'),
                                    'config_sha256':sha256(directory/guide['id']/'actual.ini'),
                                    'grid_bytes_sha256':sha256(out/'grid_reference.npz'),
                                    'path_bytes_sha256':sha256(out/'shared_path.npz') if states is not None else None}})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','train','official','evaluate'])
    p.add_argument('--gpus',type=int,default=1);p.add_argument('--source');p.add_argument('--config');p.add_argument('--work');p.add_argument('--arm-json');p.add_argument('--kind');p.add_argument('--spec')
    a=p.parse_args()
    if a.stage=='check':
        t=gpu_check(a.gpus);print(json.dumps({'gpu_count':t.cuda.device_count(),'gpu':t.cuda.get_device_name(0),'torch':t.__version__,'cuda':t.version.cuda,'host':socket.gethostname()}))
    elif a.stage=='train':train(a)
    elif a.stage=='official':official(a)
    else:evaluate(a)

if __name__=='__main__':main()
