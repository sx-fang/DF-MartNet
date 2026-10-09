"""One data layer for archived and newly saved results; no solver imports."""
from __future__ import annotations
import configparser
import math
from pathlib import Path
import shutil
import statistics
from .common import ROOT, read_csv, write_csv, json_read, json_write, summary, finite, expression, sha256

HJB=['hjb1_d10000','hjb2_d10000','hjb3_d10000','hjb3_d2000_w2010','hjb3_d2000_w10010']
EPS={'eps1':'eps1','eps0p5':'eps2','eps0p25':'eps4','eps0p125':'eps8','eps0':'eps0'}

def final_row(rows, expected):
    if not rows or int(float(rows[-1]['it']))!=expected:
        raise ValueError('Missing prescribed final iteration '+str(expected))
    if not finite(rows[-1]['rel_l1err']):raise ValueError('Nonfinite final RE; retain failure, do not replace the seed')
    return rows[-1]

def full_history(rows, expected, require_rc=False):
    """Fresh native histories must contain every prescribed update label."""
    final_row(rows,expected)
    if [int(float(r['it'])) for r in rows]!=list(range(expected+1)):
        raise ValueError('Incomplete, duplicated or unordered fresh training history')
    if any(not finite(r.get('rel_l1err')) for r in rows):
        raise ValueError('Nonfinite fresh training RE; preserve the failed result')
    if require_rc and not finite(rows[-1].get('rc')):
        raise ValueError('Missing prescribed final native RC')

def aggregates(histories, field, absolute=False, ddof=1):
    """No smoothing/interpolation on iteration axes; use common saved labels."""
    maps=[]
    for rows in histories:
        maps.append({int(float(r['it'])):float(r[field]) for r in rows if finite(r.get(field))})
    if not maps or any(not m for m in maps):return []
    labels=sorted(set.intersection(*(set(m) for m in maps)))
    result=[]
    for it in labels:
        x=[abs(m[it]) if absolute else m[it] for m in maps]
        avg=statistics.mean(x)
        sd=statistics.stdev(x) if len(x)>1 and ddof==1 else statistics.pstdev(x) if len(x)>1 else None
        result.append({'it':it,'mean':avg,'sd':sd,'n':len(x),'ddof':ddof})
    return result

def timing_stat(rows):
    by_it={int(float(r['it'])):float(r['rt_solve']) for r in rows}
    if not all(i in by_it for i in range(50,201)):raise ValueError('Timing needs all saved iterations 50..200')
    deltas=[by_it[i]-by_it[i-1] for i in range(51,201)]
    if min(deltas)<=0:raise ValueError('Nonpositive timing increment')
    stats=summary(deltas)
    mem=[float(v) for r in rows for k,v in r.items() if k.startswith('peak_memory_cuda') and finite(v)]
    if not mem:raise ValueError('Missing actual peak allocated GPU memory')
    memcol=next(k for k in rows[0] if k.startswith('peak_memory_cuda'))
    # Original Figure 6/7 harvest selected the first memory column (rank0),
    # then maximised it across saved iterations. Export all-rank max separately.
    rank0_peak=max(float(r[memcol]) for r in rows if finite(r.get(memcol)))
    return dict(n_deltas=150,median_s=stats['median'],MAD_s=stats['mad'],max_it=200,
                rt_solve_final_s=by_it[200],peak_mem=rank0_peak,peak_mem_all_ranks=max(mem),memory_definition='maximum first native GPU-memory column over saved iterations')

def profile_rows(files, ddof=1):
    """Native 101-point curves, with per-run predictions preserved in CSV."""
    raw=[read_csv(f) for f in files]
    if not raw or any(len(rs)!=101 for rs in raw):raise ValueError('Native profile requires 101 points per run')
    names=[k.removeprefix('vappr_') for k in raw[0][0] if k.startswith('vappr_')]
    out=[]
    for name in names:
        for i in range(101):
            ss=[float(r[i]['s_of_'+name+'_for_vappr']) for r in raw]
            if max(ss)-min(ss)>1e-6:raise ValueError('Different profile coordinates across runs')
            vals=[float(r[i]['vappr_'+name]) for r in raw]
            refs=[float(r[i]['vtrue_'+name]) for r in raw]
            if not all(math.isfinite(v) for v in vals+refs):raise ValueError('Nonfinite native profile')
            sd=statistics.stdev(vals) if len(vals)>1 and ddof==1 else statistics.pstdev(vals) if len(vals)>1 else None
            signed=[v-t for v,t in zip(vals,refs)]
            esd=statistics.stdev(signed) if len(vals)>1 and ddof==1 else statistics.pstdev(signed) if len(vals)>1 else None
            row={'curve':name,'s':ss[0],'v_reference':statistics.mean(refs),'v_mean':statistics.mean(vals),
                 'v_sd':sd,'error_mean':statistics.mean(signed),'error_sd':esd,'n':len(vals),'ddof':ddof}
            for j,v in enumerate(vals):row['prediction_'+str(j)]=v
            out.append(row)
    return out

def path_archived(key, case, figure_dir):
    import numpy as np
    path=ROOT/'data/archived'/key/'path_predictions.npz'
    metrics=[];layers=[]
    with np.load(path,allow_pickle=False) as z:
        ref=z['v_star'];pred=z['v_hat_all'];times=z['t_all'];levels=z['level_of']
        if ref.shape!=(4096,) or pred.shape[1:]!=(4096,) or not np.isfinite(pred).all() or not np.isfinite(ref).all():raise ValueError('Invalid shared path arrays')
        for tag,p in zip(z['arms'],pred):
            seed=int(str(tag).removeprefix('s'))
            arm=next(a for a in case['arms'] if a['seed']==seed)
            metrics.append({'arm':arm['id'],'seed':seed,'re_path':float(np.abs(p-ref).sum()/np.abs(ref).sum()),'n_states':4096})
            for k in range(32):
                mask=levels==k
                if mask.sum()!=128:raise ValueError('Incorrect path time-layer count')
                layers.append({'arm':arm['id'],'seed':seed,'layer':k,'t':float(times[mask][0]),'n':128,
                               're_v':float(np.abs(p[mask]-ref[mask]).sum()/np.abs(ref[mask]).sum())})
    write_csv(figure_dir/(key+'_path_layers.csv'),layers)
    return metrics

def qq_quantiles(files, destination):
    import numpy as np
    stats=[];pairs={k:[] for k in ['v_true','v_pred','u_true','u_pred']}
    for arm,file in files:
        with np.load(file,allow_pickle=False) as z:
            row={'arm':arm}
            for pref in ['v','u']:
                t=z[pref+'_true'].reshape(-1);p=z[pref+'_pred'].reshape(-1)
                if not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('Nonfinite QQ pair')
                row['re_'+pref+'_path']=float(np.abs(p-t).sum()/np.abs(t).sum())
                pairs[pref+'_true'].append(t);pairs[pref+'_pred'].append(p)
            stats.append(row)
    for pref in ['u','v']:
        t=np.sort(np.concatenate(pairs[pref+'_true']));p=np.sort(np.concatenate(pairs[pref+'_pred']));n=t.size;stride=max(1,n//500)
        idx=np.unique(np.concatenate(([0],np.arange(stride,n-1,stride),[n-1])))
        write_csv(destination/('scalar_qq_'+pref+'.csv'),[{'index':int(i),'n_pooled':n,'q_reference':float(t[i]),'q_prediction':float(p[i])} for i in idx])
    return stats

def compare_band(histories, official=False):
    import numpy as np
    ts=[];ys=[]
    for rs in histories:
        pairs=[]
        for r in rs:
            time=float(r['rt']) if official else float(r['rt_total']) if finite(r.get('rt_total')) else float(r['rt_solve'])+float(r['rt_log'])
            if not finite(time) or not finite(r['rel_l1err']):raise ValueError('Invalid comparison clock/error')
            pairs.append((time,float(r['rel_l1err'])))
        t,y=np.asarray(pairs).T
        if (np.diff(t)<0).any():raise ValueError('Nonmonotone saved comparison clock')
        ts.append(t);ys.append(y)
    med=float(np.median([t[-1] for t in ts]));normal=[t*(med/t[-1]) for t in ts]
    grid=np.linspace(0.,min(t[-1] for t in normal),400)
    Y=np.stack([np.interp(grid,t,y) for t,y in zip(normal,ys)])
    mean=Y.mean(0);sd=Y.std(0,ddof=1)
    return [{'t':float(t),'mean':float(m),'sd':float(s),'n':len(ts),'clock':'official native in-loop' if official else 'DF rt_solve + rt_log',
             'normalisation':'per-run final time scaled to group median'} for t,m,s in zip(grid,mean,sd)]

def _collect(run, keys, cases, group, version, archived):
    tables=run/'tables';figures=run/'figures';tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    per_seed=[];summaries=[];histories={};data={};notes=[]
    saved_origin=read_csv(ROOT/'data/archived/rho_sweep/origin_perseed.csv') if archived and group=='rho_sweep' else []
    saved_timing={r['arm']:r for r in read_csv(ROOT/'data/archived/performance_sweep/timing.csv')} if archived and group=='performance_sweep' else {}
    def metric(key,name,values,definition,expected=None):
        if not values:return
        st=summary(values)
        summaries.append(dict(case=key,metric=name,**st,expected_n=expected or st['n'],definition=definition))
    for key in keys:
        case=cases[key];kind=case['kind'];rs_all=[];seed_rows=[]
        for arm in case['arms']:
            if archived:
                path=ROOT/'data/archived'/('performance_sweep' if kind=='timing' else key)/(arm['id']+'_history.csv')
            else:
                training=run/'cases'/key/arm['id']/'training'
                if kind=='official':
                    hits=[p for p in training.glob('*.csv') if p.name.endswith('_selected_r0.csv')]
                    if len(hits)!=1:raise ValueError('Expected one explicit official final CSV')
                    rows=read_csv(hits[0]);path=tables/(key+'_'+arm['id']+'_native_history.csv')
                    write_csv(path,[dict(it=r['iter step'],rt=r['rt'],rel_l1err=r['error']) for r in rows])
                else:path=training/'selected_r0_log.csv'
            rows=read_csv(path);last=final_row(rows,arm['max_iter']);rs_all.append(rows)
            if not archived:full_history(rows,arm['max_iter'],require_rc=key in HJB or kind in ('k1','complex'))
            item=dict(case=key,arm=arm['id'],seed=arm['seed'],repeat=0,final_it=arm['max_iter'],
                      re_native=float(last['rel_l1err']))
            if key.startswith('eps') and key!='eps0':item['reference_semantics']='difference to epsilon=0 reference; not true RE'
            if finite(last.get('rc')):item['rc_native_signed']=float(last['rc']);item['rc_abs']=abs(float(last['rc']))
            for col in ['rt_solve','rt_log','rt_total','rt','rel_l1err_ualoop','rel_l1err_u_ualoop']:
                if finite(last.get(col)):item[col]=float(last[col])
            if not archived:
                record=json_read(run/'cases'/key/arm['id']/'training_record.json')
                item['actual_jobid']=record.get('jobid');item['gpu']=record.get('gpu','see resource_check.log');item['world_size']=arm['gpus']
                item['actual_host']=record.get('host');item['actual_server']=json_read(run/'slurm_job.json')['server'] if (run/'slurm_job.json').is_file() else record.get('host')
                item['weight_export_seconds']=record.get('weight_export_seconds')
            seed_rows.append(item)
            # Preserve every saved training sample used by the plots.
            write_csv(figures/(key+'_'+arm['id']+'_history.csv'),rows)
        histories[key]=rs_all
        if kind=='timing':
            st=timing_stat(rs_all[0]);tag=key.removeprefix('timing_scpx_').removesuffix('_s1')
            meta={'gpu':'A100-SXM4-80GB','world_size':case['arms'][0]['gpus']} if archived else {'actual_jobid':seed_rows[0].get('actual_jobid'),'gpu':seed_rows[0].get('gpu'),'world_size':case['arms'][0]['gpus']}
            data.setdefault('timing',[]).append(dict(arm=tag,**meta,**st));per_seed+=seed_rows;continue
        if key in HJB:
            paths=path_archived(key,case,figures) if archived else read_csv(run/'cases'/key/'evaluation/per_seed_eval.csv')
            for p in paths:
                row=next(r for r in seed_rows if r['arm']==p['arm'])
                for col in ['re_path','re_t0_s0s1','jhat_origin_fp32','jhat_mc_sd','jhat_mc_se','rc_signed_fp32']:
                    if finite(p.get(col)):row[col]=float(p[col])
            expected_path=4 if archived and key in ('hjb2_d10000','hjb3_d2000_w2010') else 4 if version=='paper_original' and key in ('hjb2_d10000','hjb3_d2000_w2010') else len(case['arms'])
            if sum('re_path' in r for r in seed_rows)!=expected_path:raise ValueError('Incomplete prescribed path pool')
            if expected_path==4:notes.append(key+': Native table metrics n=5; archived path metrics n=4, guided by seed2.')
            if not archived:
                layers=[]
                for arm in case['arms']:
                    if not any(p['arm']==arm['id'] and finite(p.get('re_path')) for p in paths):continue
                    rows=read_csv(run/'cases'/key/'evaluation'/(arm['id']+'_layers.csv'))
                    if len(rows)!=32 or {int(r['layer']) for r in rows}!=set(range(32)) or any(int(r['n'])!=128 or not finite(r['re_v']) for r in rows):
                        raise ValueError('Invalid full HJB 32-layer path evaluation')
                    layers.extend(dict(arm=arm['id'],seed=arm['seed'],**r) for r in rows)
                write_csv(figures/(key+'_path_layers.csv'),layers)
        elif kind=='k1':
            if archived:
                for pref in ['u','v']:shutil.copyfile(ROOT/f'data/archived/scalar_control/qq_{pref}.csv',figures/f'scalar_qq_{pref}.csv')
                vals=read_csv(ROOT/'data/archived/scalar_control/qq_perseed.csv')
                for row in seed_rows:
                    v=next(v for v in vals if int(v['seed'])==row['seed']);row.update(re_path=float(v['re_v_path']),re_u_path=float(v['re_u_path']))
            else:
                vals=qq_quantiles([(a['id'],run/'cases'/key/'evaluation'/(a['id']+'_qq.npz')) for a in case['arms']],figures)
                for row in seed_rows:
                    v=next(v for v in vals if v['arm']==row['arm']);row.update(re_path=v['re_v_path'],re_u_path=v['re_u_path'])
        elif not archived and kind in ('complex','eps'):
            vals=read_csv(run/'cases'/key/'evaluation/per_seed_eval.csv')
            for row in seed_rows:
                v=next(v for v in vals if v['arm']==row['arm'])
                for col in ['re_t0_s0s1','jhat_origin_fp32','jhat_mc_se','rc_signed_fp32']:
                    if finite(v.get(col)):row[col]=float(v[col])
        if archived and key.startswith('rho_'):
            for row in seed_rows:
                v=next(v for v in saved_origin if v['case']==key and v['arm']==row['arm'])
                for col in ['rc_signed_fp32','jhat_origin_fp32','jhat_group_mc_se','v_reference_origin','M_cost','groups']:
                    row[col]=float(v[col])
        for name in ['re_native','rc_abs','re_path','re_u_path','rt_solve','rt_log','rt_total','rt','re_t0_s0s1','jhat_origin_fp32','rc_signed_fp32']:
            values=[r[name] for r in seed_rows if name in r]
            definition={'re_native':'final native logger RE; see grid definition in docs/metrics.md',
                        're_path':'pooled L1 over fixed deployment states; sample SD across available prescribed arms',
                        'rc_abs':'absolute native final signed RC; native denominator may be signed',
                        'rc_signed_fp32':'publication diagnostic (Jhat - Vref)/abs(Vref)',
                        'rt_solve':'native cumulative solver seconds, excluding rt_log',
                        'rt_total':'DF console rt_solve+rt_log, rounded to 0.01 s',
                        'rt':'official native in-loop clock; reference precomputation excluded'}.get(name,name)
            if key.startswith('eps') and key!='eps0' and name=='re_native':definition='final difference to epsilon=0 reference; not perturbed true RE'
            if key.startswith('eps') and key!='eps0' and name=='rc_signed_fp32':definition='relative policy-cost gap to epsilon=0 Vref; not perturbed true RC'
            metric(key,name,values,definition,len(case['arms']))
        per_seed+=seed_rows
        for field in ['rel_l1err','rc','rel_l1err_ualoop','rel_l1err_u_ualoop']:
            band=aggregates(rs_all,field,absolute=field=='rc' and key!='scalar_control')
            if band:write_csv(figures/(key+'_'+field+'_band.csv'),band)
        if key.startswith('compare_'):
            band=compare_band(rs_all,kind=='official');write_csv(figures/(key+'_rt_band.csv'),band)
            data[key]={'comparison':band};continue
        if key.startswith('rho_'):continue
        if key.startswith('eps'):
            tag=EPS[key]
            if archived:cost=read_csv(ROOT/'data/archived/epsilon_sweep'/('cost_'+tag+'.csv'))
            else:cost=read_csv(run/'cases'/key/'evaluation'/(case['arms'][0]['id']+'_eps_cost.csv'))
            write_csv(figures/(key+'_cost_profile.csv'),cost);data[key]={'cost':cost};continue
        files=[ROOT/'data/archived'/key/(a['id']+'_profile.csv') if archived else run/'cases'/key/a['id']/'training/selected_r0_resonline_data.csv' for a in case['arms']]
        profiles=profile_rows(files,ddof=0 if archived and key=='complex_dynamics' else 1)
        write_csv(figures/(key+'_profiles.csv'),profiles);data[key]={'profiles':profiles}
        if key in HJB[:3]:
            if archived:
                label={'hjb1_d10000':'hjb1','hjb2_d10000':'hjb3a','hjb3_d10000':'hjb3b'}[key]
                true=[{'s':r['s'],'v0':r[label+'_v0'],'vT':r[label+'_vT']} for r in read_csv(ROOT/'data/archived/hjb_table/true_profiles.csv')]
            else:
                import numpy as np
                with np.load(run/'cases'/key/'evaluation/profiles.npz',allow_pickle=False) as z:
                    true=[dict(s=float(s),v0=float(v),vT=float(t)) for s,v,t in zip(z['s_true_wide'],z['v_true_wide'],z['v_terminal_wide'])]
            write_csv(figures/(key+'_true_profile.csv'),true);data[key]['true']=true
    if any(cases[k]['kind']=='timing' for k in keys):
        for r in data['timing']:
            r.update(clock='delta rt_solve at it=51..200; 50 warm-up intervals',hardware='4 x A100 (historical)' if archived else 'see raw/resource_check.log',dtype='float32 + native bf16 autocast',repeat=1)
        write_csv(tables/'timing.csv',data['timing']);write_csv(figures/'performance_timing.csv',data['timing'])
        for r in data['timing']:summaries.append({'case':r['arm'],'metric':'per_outer_iteration_seconds','n':r['n_deltas'],'median':r['median_s'],'mad':r['MAD_s'],'definition':r['clock']})
    if group=='rho_sweep':
        rho=[]
        perf={int(r['r']):r for r in read_csv(ROOT/'data/archived/rho_sweep/final_runs.csv') if r['dim']=='scpx10k'}
        for key in keys:
            r=int(key.removeprefix('rho_'));re=next(x for x in summaries if x['case']==key and x['metric']=='re_native')
            cost=[abs(x['rc_signed_fp32']) for x in per_seed if x['case']==key];st=summary(cost)
            if st['n']!=len(cases[key]['arms']):raise ValueError('Missing prescribed origin-cost evaluation')
            rc_mean=st['mean'];rc_sd=st['sample_sd']
            if archived:
                rt=float(perf[r]['mean_rt_solve']);mem=float(perf[r]['mean_peak_mem_mb'])
            else:
                rt=next(x['mean'] for x in summaries if x['case']==key and x['metric']=='rt_solve')
                peaks=[]
                for h in histories[key]:
                    memcol=next(k for k in h[0] if k.startswith('peak_memory_cuda'))
                    peaks.append(max(float(row[memcol]) for row in h if finite(row.get(memcol))))
                mem=statistics.mean(peaks)
            rho.append({'r':r,'re_mean':re['mean'],'re_sd':re['sample_sd'],'rc_mean':rc_mean,'rc_sd':rc_sd,'mean_rt_solve':rt,'mean_peak_mem_mb':mem,'n':5})
        if archived:
            shutil.copyfile(ROOT/'data/archived/rho_sweep/origin_groups.csv',tables/'origin_mc_groups.csv')
            notes.append('Origin RC recomputed from full-precision saved group means, M=4096, eight groups. jhat_group_mc_se is the original across-group SE; it is separate from seed SD and fresh centred per-path SE.')
        write_csv(tables/'rho_sweep.csv',rho);write_csv(figures/'rho_sweep.csv',rho);data['rho']=rho
    write_csv(tables/'per_seed.csv',per_seed);write_csv(tables/'summary.csv',summaries)
    table=[]
    for key in keys:
        if key not in HJB:continue
        def st(name):return next((r for r in summaries if r['case']==key and r['metric']==name),{})
        cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());cfg.read(ROOT/cases[key]['arms'][0]['config'])
        row={'case':key,'d':cfg.getint('Example','dim_x')-1,'width_v':expression(cfg.get('Network','width_v')),'hidden_v':cfg.getint('Network','num_hidden_v')}
        for name in ['re_native','re_path','rc_abs']:row.update({name:st(name).get('mean'),name+'_sd':st(name).get('sample_sd'),name+'_n':st(name).get('n')})
        row['rt_solve_median']=st('rt_solve').get('median');row['rt_solve_MAD']=st('rt_solve').get('mad');table.append(row)
    if table:
        write_csv(tables/'table1.csv',table)
        tex=[]
        for r in table:
            def pair(n):return '--' if r[n] is None else f"{r[n]:.3e} ({r[n+'_sd']:.2e}; n={r[n+'_n']})" if r[n+'_sd'] is not None else f"{r[n]:.3e} (n=1)"
            tex.append(r['case'].replace('_',r'\_')+' & '+str(r['d'])+' & '+pair('re_native')+' & '+pair('re_path')+' & '+pair('rc_abs')+' & '+f"{r['rt_solve_median']:.0f}"+r' \\')
        (tables/'table1_rows.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    from .plots import generate
    generate(run,keys,cases,group,data)
    notes+=['Final iteration metrics only; no oracle best or replacement seeds.',
            'Iteration curves use unsmoothed saved samples. Comparison curves retain the historical time interpolation and median-speed normalisation.',
            'Single epsilon arms are separate parameter choices; each records its own sample count.']
    if archived:notes.append('Archived mode: tables and figures regenerated from the bundled saved data.')
    else:notes.append('Full mode: tables and figures use the current run\'s final measurements and recorded hardware.')
    figures_index=sorted(p.name for p in figures.glob('*.pdf'))
    (run/'report.md').write_text('# '+group+' reproduction\n\nMode: '+('archived' if archived else 'full')+'. Version: '+version+'.\n\n'
          +'Tables: `tables/per_seed.csv`, `tables/summary.csv`'+(', `tables/table1.csv`' if table else '')+'.\n\n'
          +'Figures: '+', '.join('`figures/'+p+'`' for p in figures_index)+'.\n\n'
          +'\n'.join('- '+n for n in dict.fromkeys(notes))+'\n',encoding='utf-8')

def archived_collect(run,keys,cases,group):_collect(run,keys,cases,group,'paper_original',True)
def full_collect(run,keys,cases,group,version):_collect(run,keys,cases,group,version,False)
