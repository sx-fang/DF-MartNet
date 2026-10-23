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
    actual_world=torch.distributed.get_world_size() if distributed else 1
    if actual_world!=arm['gpus']:raise RuntimeError('Actual DDP worker count differs from the prescribed allocation')
    budget_record={}
    native_save=getattr(solver,'_save_weights',None);save_seconds=[]
    def capture(*a,**kw):
        problem=a[1] if len(a)>1 else kw['problem']
        budget_record.update(getattr(problem,'sample_budgets',{}))
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
                 'jobid':os.environ.get('SLURM_JOB_ID'),'world_size':actual_world,'gpu':torch.cuda.get_device_name(local),
                 'sample_budgets':budget_record,
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
    cmd=[sys.executable,'-B',str(Path(args.source)/'run.py')]+arm['cli']+['--gpus',str(arm['gpus']),'--out',str(out),'--tag','selected_r0']
    subprocess.run(cmd,check=True)
    json_write(work/'training_record.json',{'state':'complete','cli':cmd,'jobid':os.environ.get('SLURM_JOB_ID'),
               'source':str(args.source),'seed':arm['seed'],'repeat':0,'clock':'native rt; keep distinct from DF rt_solve/rt_log',
               'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(0),'world_size':arm['gpus'],
               'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'dtype':'float32'})

def evaluate(args):
    torch=gpu_check()
    from dfm_repro.evaluate import evaluate as evaluate_selected_runs
    evaluate_selected_runs(args.spec,torch,config,use_source)


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
