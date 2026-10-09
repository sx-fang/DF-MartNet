"""One-example entry points. Numeric imports exist only in compute workers."""
from __future__ import annotations
import argparse
import configparser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from .common import ROOT, json_read, json_write, sha256, package_digest, validate_files, fingerprints

HJB=['hjb1_d10000','hjb2_d10000','hjb3_d10000','hjb3_d2000_w2010','hjb3_d2000_w10010']
GROUPS={'hjb_table':HJB,'comparison':['compare_df_hjb2','compare_df_hjb3','compare_official_hjb2','compare_official_hjb3'],
        'rho_sweep':['rho_'+str(r) for r in [75,150,300,600,1200]],
        'epsilon_sweep':['eps1','eps0p5','eps0p25','eps0p125','eps0']}

def resolve_cases(name, cases):
    keys=sorted(k for k,v in cases.items() if v['kind']=='timing') if name=='performance_sweep' else GROUPS.get(name,[name])
    if any(k not in cases for k in keys): raise ValueError('Unknown or incomplete case: '+name)
    return keys

def command(cmd, output, env=None):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8') as log:
        log.write(shlex.join(map(str,cmd))+'\n');log.flush()
        result=subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
    if result.returncode: raise RuntimeError(f'Stage failed ({result.returncode}); see {output}')

def completed_stage(run, key, pins, callback, resume):
    state_path=run/'stages'/key/'stage.json';state_path.parent.mkdir(parents=True,exist_ok=True)
    if state_path.exists():
        old=json_read(state_path)
        if old['inputs']!=pins: raise ValueError('Cannot reuse stage with changed inputs: '+key)
        if old['state']=='complete' and resume:
            validate_files(old['artifacts'],run);return
        raise RuntimeError('Stage already exists; no automatic retry or partial numerical restart: '+key)
    state={'inputs':pins,'state':'running','started_utc':datetime.now(timezone.utc).isoformat()}
    json_write(state_path,state)
    try:
        artifacts=callback()
        state.update(state='complete',artifacts=fingerprints(artifacts,run))
    except BaseException as exc:
        state.update(state='failed',error=type(exc).__name__+': '+str(exc))
        json_write(state_path,state);raise
    state['finished_utc']=datetime.now(timezone.utc).isoformat();json_write(state_path,state)

def local_full(run, keys, cases, args, digest):
    if sys.platform!='linux': raise RuntimeError('Full numerical execution requires a Linux GPU compute node; CPU/Windows fallback is disabled.')
    # The resource probe runs on the allocation, before any solver import.
    gpus=max(a['gpus'] for k in keys for a in cases[k]['arms'])
    command([sys.executable,'-B','-m','dfm_repro.gpu_worker','check','--gpus',gpus],run/'raw/resource_check.log')
    source_snapshot=run/'source'
    if not source_snapshot.exists():
        for sub in ['paper_original','corrected','third_party','dfm_repro','configs']:
            shutil.copytree(ROOT/sub,source_snapshot/sub,ignore=shutil.ignore_patterns('__pycache__'))
        json_write(run/'source_files.json',fingerprints([p for p in source_snapshot.rglob('*') if p.is_file()],source_snapshot))
    validate_files(json_read(run/'source_files.json'),source_snapshot)
    for key in keys:
        case=cases[key]
        for arm in case['arms']:
            work=run/'cases'/key/arm['id'];work.mkdir(parents=True,exist_ok=True)
            source=source_snapshot/('corrected' if args.version=='corrected' and case['kind']!='official' else arm['source'])
            config=work/'actual.ini'
            if not config.exists() and case['kind']!='official':
                cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
                original=ROOT/arm['config']
                if sha256(original)!=arm['config_sha256']: raise ValueError('Config differs from approved original')
                cfg.read(original)
                cfg.set('Environment','output_dir',str(work/'training'))
                cfg.set('Environment','save_weights','True')
                # repeat_time remains in the INI. The adapter runs only selected r0;
                # the paper used r0, not a mean of this INI's five sequential repeats.
                with config.open('w',encoding='utf-8') as f:cfg.write(f)
            inputs={'package':digest,'version':args.version,'arm':arm,'config':sha256(config) if config.exists() else arm.get('cli_sha256')}
            train_key=key+'_'+arm['id']+'_train'
            def train_stage():
                cmd=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node',str(arm['gpus']),
                     str(source_snapshot/'dfm_repro/gpu_worker.py'),'train','--source',str(source),'--config',str(config),'--work',str(work),
                     '--kind',case['kind'],'--arm-json',json.dumps(arm)]
                if case['kind']=='official':
                    cmd=[sys.executable,'-B',str(source_snapshot/'dfm_repro/gpu_worker.py'),'official','--source',str(source),'--work',str(work),'--arm-json',json.dumps(arm)]
                command(cmd,work/'training_console.log')
                return [p for p in (work/'training').rglob('*') if p.is_file()]+[work/'training_console.log',work/'training_record.json']
            completed_stage(run,train_key,inputs,train_stage,args.resume)
        if case['kind'] not in ('timing','official') and not key.startswith('compare_'):
            # Shared state distribution comes from the first prescribed guide arm.
            # The reference is evaluated once and used by every selected final net.
            evaluation=run/'cases'/key/'evaluation';spec=evaluation/'eval_spec.json'
            evaluation.mkdir(exist_ok=True)
            plan={'kind':case['kind'],'case':key,'arms':case['arms'],'version':args.version,'case_dir':str(run/'cases'/key),'source_root':str(source_snapshot)}
            if not spec.exists():json_write(spec,plan)
            pins={'package':digest,'spec':sha256(spec),'train_states':[json_read(run/'stages'/(key+'_'+a['id']+'_train')/'stage.json')['artifacts'] for a in case['arms']]}
            def eval_stage():
                command([sys.executable,'-B',str(source_snapshot/'dfm_repro/gpu_worker.py'),'evaluate','--spec',str(spec)],evaluation/'console.log')
                return [p for p in evaluation.rglob('*') if p.is_file()]
            completed_stage(run,key+'_evaluate',pins,eval_stage,args.resume)
    from .collect import full_collect
    full_collect(run,keys,cases,args.case,args.version)

def slurm_submit(run, keys, cases, args, digest):
    """B-prime submission: execute an immutable private copy, never the live tree."""
    site=json_read(args.site)
    required={'partition','account','time','cpus_per_gpu','mem_gb'}
    if not required<=site.keys(): raise ValueError('Incomplete Slurm site configuration')
    job_record=run/'slurm_job.json'
    if job_record.exists():
        record=json_read(job_record)
        if not args.resume or record['package']!=digest: raise RuntimeError('Submission already recorded; refusing duplicate submission')
        jobid=record['jobid']
    else:
        freeze=run/'frozen_package';shutil.copytree(ROOT,freeze,ignore=shutil.ignore_patterns('.git','results','__pycache__'))
        jobscript=run/'run.slurm'
        command_line=[site.get('python',sys.executable),'-B',str(freeze/'experiments'/args.case/'run.py'),
                      '--backend','local','--output-root',str(run.parent.parent),'--run-id',run.name,'--version',args.version,'--resume']
        jobscript.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\ncd '+shlex.quote(str(freeze))+'\n'+shlex.join(command_line)+'\n',encoding='utf-8')
        gpus=max(a['gpus'] for k in keys for a in cases[k]['arms'])
        cmd=['bash',str(ROOT/'slurm/submit_job.sh'),str(freeze),'--parsable','--account',site['account'],'--partition',site['partition'],'--time',site['time'],
             '--gpus',str(gpus),'--cpus-per-gpu',str(site['cpus_per_gpu']),'--mem',str(site['mem_gb'])+'G',
             '--job-name','dfm_'+args.case,'--output',str(run/'raw/slurm_%j.out'),'--error',str(run/'raw/slurm_%j.err'),str(jobscript)]
        (run/'raw').mkdir(exist_ok=True)
        response=subprocess.check_output(cmd,text=True).strip();jobid=response.split(';')[0]
        if not jobid.isdigit(): raise ValueError('Invalid sbatch job identity: '+response)
        json_write(job_record,{'package':digest,'server':site.get('server',socket.gethostname()),'jobid':jobid,'frozen':str(freeze),'submitted_utc':datetime.now(timezone.utc).isoformat()})
    print('Submitted frozen end-to-end case; job '+jobid,flush=True)
    while True:
        pending=subprocess.check_output(['squeue','-h','-j',jobid,'-o','%i %T'],text=True).strip()
        if not pending:break
        time.sleep(30)
    accounting=subprocess.check_output(['sacct','-n','-P','-j',jobid,'--format','JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES'],text=True)
    (run/'raw/sacct.txt').write_text(accounting,encoding='utf-8')
    primary=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==jobid]
    if len(primary)!=1 or primary[0][1]!='COMPLETED' or primary[0][2]!='0:0':
        raise RuntimeError('Slurm terminal state not accepted; no automatic retry. See raw/sacct.txt')
    manifest=json_read(run/'run_manifest.json')
    if manifest.get('state')!='complete':raise RuntimeError('Job ended but output contract incomplete')
    validate_files(manifest['artifacts'],run)
    # Scheduler stdout may still grow after the compute child has written its
    # manifest. Pin these files only after Slurm confirms successful completion.
    manifest['artifacts'].update(fingerprints([run/'raw/sacct.txt']+list((run/'raw').glob('slurm_*.out'))+list((run/'raw').glob('slurm_*.err')),run))
    manifest['slurm_terminal_verified']=True
    json_write(run/'run_manifest.json',manifest)

def main(case=None):
    parser=argparse.ArgumentParser(description='Train, evaluate, collect and plot one paper case.')
    parser.add_argument('--case',default=case,required=case is None)
    parser.add_argument('--mode',choices=['full','archived'],default='full')
    parser.add_argument('--version',choices=['paper_original','corrected'],default='paper_original')
    parser.add_argument('--backend',choices=['local','slurm'],default='local')
    parser.add_argument('--site',type=Path,default=ROOT/'slurm/site.json')
    parser.add_argument('--output-root',type=Path,default=Path(os.environ.get('DFM_RESULTS','results')))
    parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cases=json_read(ROOT/'configs/cases.json');keys=resolve_cases(args.case,cases)
    digest=package_digest()
    if args.dry_run:
        print(json.dumps({'case':args.case,'version':args.version,'mode':args.mode,'case_keys':keys,
                          'selected_arms':sum(len(cases[k]['arms']) for k in keys),'max_gpus_per_stage':max(a['gpus'] for k in keys for a in cases[k]['arms']),
                          'stages':['collect saved final','plot','validate'] if args.mode=='archived' else ['train selected r0','evaluate','collect final','plot','validate'],'package_sha256':digest},indent=2));return
    if args.mode=='archived' and args.version!='paper_original': raise ValueError('Archived data are historical; do not relabel as corrected')
    if args.resume and not args.run_id:raise ValueError('--resume requires an explicit --run-id')
    run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('Invalid run-id')
    run=(args.output_root/args.case/run_id).resolve();run.mkdir(parents=True,exist_ok=True)
    (run/'raw').mkdir(exist_ok=True)
    manifest_path=run/'run_manifest.json'
    signature={'case':args.case,'version':args.version,'mode':args.mode,'package_sha256':digest}
    if manifest_path.exists():
        old=json_read(manifest_path)
        if not args.resume or old['signature']!=signature:raise RuntimeError('Existing run or changed protocol; choose a new run-id')
        if old['state']=='complete':
            validate_files(old['artifacts'],run)
            if args.mode=='full' and args.backend=='slurm' and not old.get('slurm_terminal_verified'):
                slurm_submit(run,keys,cases,args,digest)
            print(run/'report.md');return
    manifest={'signature':signature,'state':'in_progress','seed_pools':{k:cases[k]['arms'] for k in keys},'created_utc':datetime.now(timezone.utc).isoformat()}
    json_write(manifest_path,manifest)
    try:
        if args.mode=='archived':
            from .collect import archived_collect
            archived_collect(run,keys,cases,args.case)
        elif args.backend=='slurm':
            slurm_submit(run,keys,cases,args,digest);print(run/'report.md');return
        else:local_full(run,keys,cases,args,digest)
        required=[run/'tables/summary.csv',run/'report.md']
        if not all(p.is_file() for p in required) or not list((run/'figures').glob('*.pdf')):raise ValueError('Missing required table, report or figures')
        manifest['artifacts']=fingerprints([p for p in run.rglob('*') if p.is_file() and p!=manifest_path and not p.is_relative_to(run/'frozen_package')
                                           and not (p.parent==run/'raw' and p.name.startswith('slurm_') and p.suffix in ('.out','.err'))],run)
        manifest.update(state='complete',finished_utc=datetime.now(timezone.utc).isoformat());json_write(manifest_path,manifest)
    except BaseException as exc:
        manifest.update(state='failed',error=type(exc).__name__+': '+str(exc));json_write(manifest_path,manifest);raise
    print('Complete: '+str(run/'report.md'))

if __name__=='__main__':main()
