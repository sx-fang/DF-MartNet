"""Independent audit of existing saved arrays/CSV; no solver or network imports.

This is data analysis, not a numerical experiment or a training smoke test.
The printed paper values are used only for comparison, never as a substitute
for recomputed measurements. Writes a JSON audit to the requested output.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/archived'

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def nums(rs,col):return [float(r[col]) for r in rs]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    cases=json.loads((ROOT/'configs/cases.json').read_text())
    printed=json.loads((DATA/'TABLE1_PRINTED.json').read_text())
    report={'scope':'existing saved data only; no training, inference or Monte Carlo generation','table1':{},'checks':[]}
    def check(name,passed,**details):
        report['checks'].append(dict(check=name,passed=bool(passed),**details))
    for key,paper in printed.items():
        final=[]
        for arm in cases[key]['arms']:
            history=rows(DATA/key/(arm['id']+'_history.csv'))
            assert int(history[-1]['it'])==arm['max_iter']
            final.append(history[-1])
        re=nums(final,'rel_l1err');rc=[abs(x) for x in nums(final,'rc')];rt=nums(final,'rt_solve')
        with np.load(DATA/key/'path_predictions.npz',allow_pickle=False) as z:
            ref=z['v_star'];pred=z['v_hat_all']
            assert ref.shape==(4096,) and pred.shape[1:]==(4096,) and np.isfinite(pred).all() and np.isfinite(ref).all()
            path=[float(np.abs(p-ref).sum()/np.abs(ref).sum()) for p in pred]
            check(key+' path layers',all(np.count_nonzero(z['level_of']==i)==128 for i in range(32)),n_runs=len(path),n_states=4096,n_levels=32)
            pool=rows(DATA/key/'published_path_perseed.csv')
            check(key+' saved path metrics',max(abs(v-float(p['RE_v'])) for v,p in zip(path,pool))<1e-14)
        measured={'re_native':statistics.mean(re),'re_native_sd':statistics.stdev(re),'re_path':statistics.mean(path),
                  're_path_sd':statistics.stdev(path),'rc_abs':statistics.mean(rc),'rc_abs_sd':statistics.stdev(rc),'rt_solve':statistics.median(rt)}
        rounding={k:'.2e' if k=='re_native' else '.0f' if k=='rt_solve' else '.1e' for k in measured}
        matches={k:float(format(v,rounding[k]))==paper[k] for k,v in measured.items()}
        report['table1'][key]={'recomputed':measured,'printed':paper,'rounded_match':matches,'n_native':len(re),'n_path':len(path),
                              'rt_solve_MAD':statistics.median(abs(v-statistics.median(rt)) for v in rt)}
        check(key+' printed Table 1',all(matches.values()))
    timing=rows(DATA/'performance_sweep/timing.csv')
    timing_checks=[]
    for published in timing:
        tag=published['arm'];native=rows(DATA/'performance_sweep'/('timing_scpx_'+tag+'_s1_history.csv'))
        clock={int(r['it']):float(r['rt_solve']) for r in native}
        deltas=[clock[i]-clock[i-1] for i in range(51,201)];median=statistics.median(deltas);mad=statistics.median(abs(v-median) for v in deltas)
        memcol=next(k for k in native[0] if k.startswith('peak_memory_cuda'))
        memory=max(float(r[memcol]) for r in native)
        ok=len(native)==201 and len(deltas)==150 and min(deltas)>0 and float(f'{median:.6f}')==float(published['median_s']) and float(f'{mad:.6f}')==float(published['MAD_s']) and memory==float(published['peak_mem'])
        timing_checks.append(dict(arm=tag,passed=ok,median_s=median,MAD_s=mad,peak_mem_MiB=memory,n_deltas=len(deltas)))
    report['timing']=timing_checks;check('20 native timing histories',all(r['passed'] for r in timing_checks),configurations=len(timing_checks))
    origins=rows(DATA/'rho_sweep/origin_perseed.csv');groups=rows(DATA/'rho_sweep/origin_groups.csv');rho=[]
    for p in origins:
        g=[float(r['jhat']) for r in groups if r['case']==p['case'] and r['arm']==p['arm']]
        mean=statistics.mean(g);ref=float(p['v_reference_origin']);se=statistics.stdev(g)/math.sqrt(len(g))
        check(p['case']+'/'+p['arm']+' origin groups',len(g)==8 and abs(mean-float(p['jhat_origin_fp32']))<1e-14 and abs((mean-ref)/abs(ref)-float(p['rc_signed_fp32']))<1e-14 and abs(se-float(p['jhat_group_mc_se']))<1e-14)
    for p in rows(DATA/'rho_sweep/origin_cost_summary.csv'):
        vals=[abs(float(r['rc_signed_fp32'])) for r in origins if r['case']=='rho_'+p['r']]
        avg=statistics.mean(vals);sd=statistics.stdev(vals)
        matches=all(abs(float(p['final_s'+str(s)])-abs(float(next(r for r in origins if r['case']=='rho_'+p['r'] and int(r['seed'])==s)['rc_signed_fp32'])))<5.1e-7 for s in range(1,6))
        rho.append(dict(r=int(p['r']),mean=avg,sample_sd=sd,published_mean=float(p['mean_REc']),published_sd=float(p['SD_REc']),rounded_perseed_match=matches))
    report['rho_origin']=rho;check('rho origin published rounding',all(r['rounded_perseed_match'] for r in rho),runs=len(origins))
    comparisons=[]
    for eq in ['hjb2','hjb3']:
        for method,tag in [('df','df'),('official','reference')]:
            key='compare_'+method+'_'+eq;ts=[];ys=[]
            for arm in cases[key]['arms']:
                native=rows(DATA/key/(arm['id']+'_history.csv'))
                t=[float(r['rt']) if method=='official' else float(r['rt_total']) if r.get('rt_total') else float(r['rt_solve'])+float(r['rt_log']) for r in native]
                ts.append(np.array(t));ys.append(np.array(nums(native,'rel_l1err')))
            median=np.median([t[-1] for t in ts]);ts=[t*median/t[-1] for t in ts];grid=np.linspace(0.,min(t[-1] for t in ts),400)
            y=np.stack([np.interp(grid,t,v) for t,v in zip(ts,ys)]);published=rows(DATA/'comparison'/('band_'+tag+'_'+eq+'.csv'))
            diff={'time_max_abs':float(np.max(np.abs(grid-nums(published,'t')))),
                  'mean_max_abs':float(np.max(np.abs(y.mean(0)-nums(published,tag+'_mean')))),
                  'sd_max_abs':float(np.max(np.abs(y.std(0,ddof=1)-nums(published,tag+'_sd'))))}
            comparisons.append(dict(case=key,**diff))
            # Recompute the published band independently from saved histories.
            check(key+' historical band',diff['mean_max_abs']<2e-5 and diff['sd_max_abs']<2e-5,**diff)
    report['comparison']=comparisons
    profiles=[rows(DATA/'complex_dynamics'/(a['id']+'_profile.csv')) for a in cases['complex_dynamics']['arms']]
    published=rows(DATA/'complex_dynamics/published_perr.csv');diffs=[]
    for name,curve in [('S0','diag'),('S1','manifold')]:
        for i,p in enumerate(published):
            errors=[float(r[i]['vappr_'+curve])-float(r[i]['vtrue_'+curve]) for r in profiles]
            diffs.extend([abs(statistics.mean(errors)-float(p['err_5seed_mean_'+name])),abs(statistics.pstdev(errors)-float(p['err_5seed_sd_'+name]))])
    check('complex profile historical ddof0',max(diffs)<1e-12,max_abs_difference=max(diffs))
    for field in ['rc','rel_l1err_ualoop','rel_l1err_u_ualoop']:
        published=rows(DATA/'scalar_control'/('published_'+field+'_history.csv'));samples=[]
        histories=[rows(DATA/'scalar_control'/(a['id']+'_history.csv')) for a in cases['scalar_control']['arms']]
        for r in histories[0]:
            try:valid=math.isfinite(float(r[field]))
            except ValueError:valid=False
            if valid:samples.append(int(r['it']))
        maps=[{int(r['it']):r for r in h} for h in histories];diffs=[]
        for it,p in zip(samples,published):
            values=[float(h[it][field]) for h in maps]
            diffs.extend([abs(statistics.mean(values)-float(p['mean'])),abs(statistics.stdev(values)-float(p['sd']))])
        check('scalar '+field+' history',len(samples)==len(published) and max(diffs)<1e-12,max_abs_difference=max(diffs),samples=len(samples))
    for pref in ['u','v']:
        q=rows(DATA/'scalar_control'/('qq_'+pref+'.csv'));n=int(q[0]['n_pooled']);idx=nums(q,'index')
        check('scalar '+pref+' QQ pool',n==5*4096*31 and idx[0]==0 and idx[-1]==n-1 and all(a<b for a,b in zip(idx,idx[1:])),n_pooled=n,quantiles=len(q))
    report['all_required_checks_passed']=all(r['passed'] for r in report['checks'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(('PASS' if report['all_required_checks_passed'] else 'FAIL')+': '+str(len(report['checks']))+' saved-data checks; '+str(args.output))
    if not report['all_required_checks_passed']:sys.exit(1)

if __name__=='__main__':main()
