"""Small mathematical/torch regressions for a declared Linux compute node.

These are synthetic unit fixtures, not reduced versions of paper experiments.
Never included in local unittest discovery; no numerical imports precede the
compute-node guard. No training, historical weights or paper INIs are changed.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def source_import_check(source):
    script="import sys;sys.path.insert(0,sys.argv[1]);import solver,sampling,loss_martnet,utils"
    subprocess.run([sys.executable,'-B','-c',script,str(source)],check=True,timeout=60)

def distributed_worker(rank,world,init_file,out_dir):
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    source=ROOT/'engines/runtime_4d30ed0';sys.path.insert(0,str(source))
    import utils,examples
    from loss_martnet import MartNetLoss
    from sample_budgets import split_global_budget
    dist.init_process_group('gloo',rank=rank,world_size=world,
                            init_method='file://'+str(Path(init_file).resolve()))
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    evidence={}
    try:
        # Unequal chunk/rank counts, and an exhausted rank with zero share.
        utils.get_safe_chunksize=lambda *a,**k:3
        for M in (17,1):
            counts=split_global_budget(M,world);offset=sum(counts[:rank]);consumed=0
            def chunk(count):
                nonlocal consumed
                ids=torch.arange(offset+consumed,offset+consumed+count,dtype=torch.float64)
                consumed+=count
                mean=ids.mean().reshape(1,1)
                return mean,(2*mean if M==17 else None)
            x=torch.zeros(1,2)
            mean,vector=utils.mc_mean_with_budget(x,chunk,M,use_dist=True,
                num_item_persize=2,with_gradient=M==17)
            assert consumed==counts[rank]
            assert abs(float(mean)-(M-1)/2)<1e-13
            if vector is not None:assert abs(float(vector)-(M-1))<1e-13
            evidence['reference_M'+str(M)]={'local_count':consumed,'global_count':M,'mean':float(mean)}

        # Public native-cost boundary must split once and weight unequal means.
        class CostFixture(examples.SOCProblem):
            def _initial_points(self,n):
                counts=split_global_budget(17,world)
                z=torch.arange(sum(counts[:rank]),sum(counts[:rank])+n,dtype=torch.float64)
                return torch.stack([torch.zeros_like(z),z],dim=1)
            def running_cost(self,acc,x,u):return acc
            def system_step(self,x,u):return torch.stack([torch.ones_like(x[:,0]),x[:,1]],dim=1)
            def terminal_cost(self,x):return x[:,1:]
            def project_onto_boundary(self,x):return x
        fixture=CostFixture(2,dt=1.)
        cost=fixture.estimate_cost(lambda x:torch.zeros(len(x),1),17)
        assert abs(float(cost)-8.)<1e-13
        evidence['global_cost_M17']=float(cost)

        # Exact dense stopped-gradient target versus the two-rank estimator.
        counts=(7,6);total=sum(counts);offset=sum(counts[:rank])
        ids=torch.arange(offset,offset+counts[rank]);first=ids[ids%2==0];second=ids[ids%2==1]
        ids=torch.cat([first,second]);p=torch.tensor(1.25,requires_grad=True)
        features=(ids.double()+1).reshape(-1,1)/7
        res=p*features;rho=torch.cat([features,.5-features],dim=1)
        loss=MartNetLoss.__new__(MartNetLoss);loss.use_dist=True;loss.world_size=world
        loss._row_w=None;loss.problem=SimpleNamespace(_batch_group_sizes=(len(first),len(second)),
            _batch_global_eval_count=total,_batch_point_counts=counts)
        value=loss.paired_mart_loss(res,rho);value.backward()
        grad=p.grad.clone();dist.all_reduce(grad);grad/=world
        q=torch.tensor(1.25,requires_grad=True);f=(torch.arange(total,dtype=torch.float64)+1).reshape(-1,1)/7
        rr=q*f;rh=torch.cat([f,.5-f],dim=1);both=rh*rr
        A=both[::2].mean(0);B=both[1::2].mean(0)
        dense=(A*B).mean()/4;dense.backward()
        expected_value=(A*B).mean()/2
        assert abs(float(value.detach()-expected_value.detach()))<1e-12
        assert abs(float(grad-q.grad))<1e-12
        evidence['paired_DDP']={'value':float(value.detach()),'dense_value':float(expected_value.detach()),
            'averaged_gradient':float(grad),'dense_gradient':float(q.grad)}

        # Real pilot sampler: independent path groups and exact branch budget.
        from sampling import refresh_pilot_batch,paired_point_layout
        problem=examples.HJB1(5,dt=1/32,batch_size=counts[rank],num_pilot_paths=6)
        problem._batch_point_counts=counts;problem._batch_rank=rank;problem.noise_sampler='amc'
        control=lambda x:torch.zeros(len(x),4)
        for iteration in range(8):
            batch=refresh_pilot_batch(problem,None if iteration==0 else object(),control)
            left,right=(set(x.tolist()) for x in problem._batch_path_groups)
            assert left.isdisjoint(right);assert problem.batch_size==counts[rank]
            assert len(batch)==paired_point_layout(problem)[0]
            assert problem._amc_num_pairs+len(batch)==counts[rank]
        evidence['pilot_path_groups']='disjoint; exact configured branch counts'
    finally:
        dist.destroy_process_group()
    (Path(out_dir)/('rank'+str(rank)+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')

def scalar_reference_check():
    import torch
    import examples
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    d,m=32,20;tau=.6
    problem=examples.HJBSparseLQK1(d+1,dt=1/32,batch_size=8,num_pilot_paths=8)
    problem.omega=1.;problem.delta=1.;problem.sparse_m=m;problem.sparse_a=1.;problem.c_weights=(1.,)
    q=torch.zeros(d);q[:2]=1/(2**.5)
    index=torch.arange(d,dtype=torch.float64);frequency=torch.arange(d,dtype=torch.float64).unsqueeze(1)
    amplitude=torch.full((d,1),(2/d)**.5);amplitude[0]=d**-.5
    cosine=(amplitude*torch.cos(torch.pi*(2*index+1)*frequency/(2*d))).T
    Q=torch.linalg.qr(torch.cat([q[:,None],cosine],dim=1)).Q[:,:m]
    # Independent RK4 of h'=L^T h, A'=2(q.h)^2, G'=|Q^T h|^2/A.
    def derivative(h,A,G):
        return h.roll(1)-h,2*(q@h)**2,(Q.T@h).square().sum()/A
    h=q.clone();A=torch.tensor(1.);G=torch.tensor(0.);step=tau/4096
    for _ in range(4096):
        k1=derivative(h,A,G)
        k2=derivative(h+step*k1[0]/2,A+step*k1[1]/2,G+step*k1[2]/2)
        k3=derivative(h+step*k2[0]/2,A+step*k2[1]/2,G+step*k2[2]/2)
        k4=derivative(h+step*k3[0],A+step*k3[1],G+step*k3[2])
        h+=step*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        A+=step*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        G+=step*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
    z=torch.stack([torch.zeros(d),torch.linspace(-1,1,d),torch.linspace(.2,.8,d)])
    x=torch.cat([torch.full((len(z),1),1-tau),z],dim=1).requires_grad_()
    expected=(z@h-1).square()/(2*A)+G
    expected_u=-(q@h)*(z@h-1)/A
    value=problem.v(x).reshape(-1);control=problem.u_star(x).reshape(-1)
    value_error=float((value.detach()-expected).abs().max())
    control_error=float((control.detach()-expected_u).abs().max())
    grad=torch.autograd.grad(value.sum(),x)[0][:,1:]
    projection_error=float((control.detach()+grad@q).abs().max())
    assert value_error<3e-5 and control_error<3e-5 and projection_error<1e-12
    xT=x.detach().clone();xT[:,0]=1.
    terminal_error=float((problem.v(xT)-problem.g(z)).abs().max());assert terminal_error<1e-12
    return dict(value_max_abs_error=value_error,control_max_abs_error=control_error,
                control_projection_max_abs_error=projection_error,terminal_max_abs_error=terminal_error)

def main():
    if sys.platform!='linux' or os.environ.get('DFM_COMPUTE_NODE')!='1':
        raise SystemExit('Run these numerical regressions only on a declared dedicated Linux compute node')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import torch
    for source in sorted((ROOT/'engines').glob('runtime_*')):source_import_check(source)
    sys.path.insert(0,str(ROOT/'engines/runtime_4d30ed0'))
    with tempfile.TemporaryDirectory() as d:
        torch.multiprocessing.spawn(distributed_worker,args=(2,str(Path(d)/'group'),d),nprocs=2,join=True)
        ranks=[json.loads((Path(d)/('rank'+str(r)+'.json')).read_text()) for r in range(2)]
    scalar=scalar_reference_check()
    result=dict(passed=True,scope='Synthetic CPU mathematical and two-process Gloo regressions; no paper training',
                torch=torch.__version__,python=sys.version.split()[0],world_size=2,device='CPU',ranks=ranks,scalar_reference=scalar)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
