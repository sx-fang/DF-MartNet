"""Publication figures from accepted saved CSVs only. Never calls a solver."""
from __future__ import annotations
from .common import read_csv, write_csv

def generate(run,keys,cases,group,data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
    out=run/'figures'
    def arr(rs,col):return np.array([float(r[col]) if r.get(col) not in ('',None) else np.nan for r in rs])
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=160);plt.close(fig)
    def decorate(ax,x='Iteration',y='Relative error',log=False):
        ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.25,which='both')
        if log:ax.set_yscale('log')
    def lineband(ax,rows,color,label,x='it'):
        if not rows:return
        xx=arr(rows,x);m=arr(rows,'mean');s=arr(rows,'sd')
        ax.plot(xx,m,color=color,lw=1.3,label=label)
        if np.isfinite(s).any():ax.fill_between(xx,m,m+2*s,color=color,alpha=.2,lw=0)
    def history(ax,key,fields=(('rel_l1err','#1f77b4','RE'),('rc','#ff7f0e','|RC|'))):
        for f,c,l in fields:
            p=out/(key+'_'+f+'_band.csv')
            if p.exists():lineband(ax,read_csv(p),c,l)
        decorate(ax,log=True);ax.legend(fontsize=8)
    def profile(ax,key,error=False):
        rows=data[key]['profiles'];labels=list(dict.fromkeys(r['curve'] for r in rows))
        for ci,name in enumerate(labels):
            rs=[r for r in rows if r['curve']==name];s=arr(rs,'s');m=arr(rs,'error_mean' if error else 'v_mean');sd=arr(rs,'error_sd' if error else 'v_sd')
            c=['#d62728','#1f77b4'][ci%2];ax.plot(s,m,color=c,lw=1.3,label=('error ' if error else 'prediction ')+name)
            if np.isfinite(sd).any():ax.fill_between(s,m-2*sd,m+2*sd,color=c,alpha=.18,lw=0)
            if not error:ax.plot(s,arr(rs,'v_reference'),color=c,ls='--',lw=1.2,label='native reference '+name)
        decorate(ax,'s','signed point error' if error else 'Value');ax.legend(fontsize=7)
    def merged(ax,key):
        rs=[r for r in data[key]['profiles'] if r['curve']=='diag'];s=arr(rs,'s');m=arr(rs,'v_mean');sd=arr(rs,'v_sd')
        true=data[key]['true'];st=arr(true,'s');mask=np.abs(st)<=1.5+1e-12
        ax.plot(st[mask],arr(true,'v0')[mask],'k-',lw=1.4,label='wide reference at t=0')
        ax.plot(st[mask],arr(true,'vT')[mask],'k--',lw=1.4,label='wide reference at t=T')
        ax.plot(s,m,'o',color='red',ms=2.4,label='prediction at t=0')
        ax.fill_between(s,m-2*sd,m+2*sd,color='#ffcccc',alpha=.8,lw=0)
        ax.set_xlim(-1.5,1.5);decorate(ax,'s','Value');ax.legend(fontsize=7,loc='lower left')
    hjb=[k for k in keys if k.startswith('hjb')]
    for key in hjb:
        if key in data and 'true' in data[key]:
            fig,axs=plt.subplots(1,2,figsize=(10,3.7));merged(axs[0],key);history(axs[1],key);save(fig,key+'_figure2_row')
        fig,axs=plt.subplots(1,2,figsize=(10,3.7));profile(axs[0],key,error=True);history(axs[1],key);save(fig,key+'_point_error_and_history')
        layers=out/(key+'_path_layers.csv')
        if layers.exists():
            rs=read_csv(layers);ts=sorted(set(float(r['t']) for r in rs));band=[]
            for t in ts:
                vals=[float(r['re_v']) for r in rs if float(r['t'])==t]
                band.append(dict(t=t,mean=np.mean(vals),sd=np.std(vals,ddof=1) if len(vals)>1 else '',n=len(vals)))
            write_csv(out/(key+'_path_band.csv'),band)
            fig,ax=plt.subplots(figsize=(5,3.7));lineband(ax,band,'#1f77b4','deployment RE',x='t');decorate(ax,'t',log=True);save(fig,key+'_path_error')
    if group=='hjb_table':
        first=hjb[:3];fig,axs=plt.subplots(2,3,figsize=(14,7))
        for col,key in enumerate(first):merged(axs[0,col],key);history(axs[1,col],key);axs[0,col].set_title(key)
        save(fig,'figure2_hjb_d10000')
        fig,axs=plt.subplots(1,3,figsize=(14,3.8))
        for key,c,label in [(hjb[3],'#1f77b4','d+10'),(hjb[4],'#ff7f0e','5d+10')]:
            rs=data[key]['profiles'];s=arr(rs,'s');m=arr(rs,'error_mean');sd=arr(rs,'error_sd')
            axs[0].plot(s,m,color=c,label=label);axs[0].fill_between(s,m-2*sd,m+2*sd,color=c,alpha=.2)
            for ax,f in zip(axs[1:],['rel_l1err','rc']):lineband(ax,read_csv(out/(key+'_'+f+'_band.csv')),c,label)
        decorate(axs[0],'s','signed point error');decorate(axs[1],log=True);decorate(axs[2],y='|RC|',log=True)
        for ax in axs:ax.legend()
        save(fig,'figure3_hjb_d2000')
    if group=='comparison':
        fig,axs=plt.subplots(1,2,figsize=(10,3.8))
        for ax,eq in zip(axs,['hjb2','hjb3']):
            for method,c,label in [('df','#1f77b4','DF-MartNet'),('official','#ff7f0e','SOC-MartNet')]:
                key='compare_'+method+'_'+eq;lineband(ax,data[key]['comparison'],c,label,x='t')
            decorate(ax,'RT (s; distinct native clocks)',log=True);ax.set_title(eq.upper());ax.legend(fontsize=8)
        save(fig,'figure4_method_comparison')
    if group=='complex_dynamics':
        fig,axs=plt.subplots(1,2,figsize=(10,3.8));profile(axs[0],group);history(axs[1],group);save(fig,'figure5_complex_dynamics')
        fig,ax=plt.subplots(figsize=(5,3.8));profile(ax,group,error=True);save(fig,'complex_point_error')
    if group=='rho_sweep':
        rows=data['rho'];r=arr(rows,'r');fig,axs=plt.subplots(1,2,figsize=(10,3.8))
        for tag,c in [('re','#1f77b4'),('rc','#ff7f0e')]:
            m=arr(rows,tag+'_mean');s=arr(rows,tag+'_sd');axs[0].plot(r,m,'o-',color=c,label=tag.upper());axs[0].fill_between(r,m,m+2*s,color=c,alpha=.2)
        for field,c,label in [('mean_rt_solve','#1f77b4','RT'),('mean_peak_mem_mb','#ff7f0e','GPU memory')]:
            y=arr(rows,field);axs[1].plot(r,y/y[0],'o-',color=c,label=label)
        for ax in axs:ax.set_xscale('log',base=2);ax.set_xticks(r,labels=[str(int(x)) for x in r]);ax.legend();ax.grid(alpha=.25)
        axs[0].set_yscale('log');axs[0].set_xlabel('r');axs[0].set_ylabel('Relative error');axs[1].set_xlabel('r');axs[1].set_ylabel('Ratio to r=75')
        save(fig,'figure6_rho_sweep')
    if group=='performance_sweep':
        rows=data['timing'];fig,axs=plt.subplots(1,4,figsize=(16,3.6));ratios=[]
        for ax,prefix,label in zip(axs,['n','b','m','d'],['N','half-batch size','pilot paths M','state dimension d']):
            sub=sorted((r for r in rows if r['arm'].startswith(prefix)),key=lambda r:int(r['arm'][1:]));x=[int(r['arm'][1:]) for r in sub]
            for field,c,title in [('median_s','#1f77b4','RT'),('peak_mem','#ff7f0e','GPU memory')]:
                y=arr(sub,field);ratio=y/y[0];ax.plot(x,ratio,'o-',color=c,label=title)
                for r,q in zip(sub,ratio):ratios.append(dict(arm=r['arm'],quantity=field,ratio=float(q)))
            ax.set_xscale('log',base=2);ax.set_xlabel(label);ax.set_ylabel('Ratio to smallest value');ax.grid(alpha=.25);ax.legend(fontsize=8)
        write_csv(out/'performance_ratios.csv',ratios);save(fig,'figure7_performance_sweep')
    if group=='scalar_control':
        fig,axs=plt.subplots(1,3,figsize=(14,3.8));history(axs[0],group,[('rel_l1err_ualoop','#1f77b4','path RE(v)'),('rel_l1err_u_ualoop','#2ca02c','path RE(u)'),('rc','#ff7f0e','RC')])
        for ax,pref in zip(axs[1:],['u','v']):
            rs=read_csv(out/('scalar_qq_'+pref+'.csv'));x=arr(rs,'q_reference');y=arr(rs,'q_prediction');lo=min(x.min(),y.min());hi=max(x.max(),y.max())
            ax.plot([lo,hi],[lo,hi],'k--',lw=1,label='y=x');ax.plot(x,y,'o',mfc='none',mec='#1f77b4',ms=3,lw=0)
            ax.set_aspect('equal',adjustable='box');decorate(ax,'reference '+pref,'predicted '+pref);ax.legend(fontsize=8)
        save(fig,'figure8_scalar_control')
    if group=='epsilon_sweep':
        fig,ax=plt.subplots(figsize=(7,4.7))
        for key,c,eps in zip(keys,['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'],[1,.5,.25,.125,0]):
            rs=data[key]['cost'];s=arr(rs,'s');v=arr(rs,'v_pred');j=arr(rs,'jhat');ax.plot(s,v,color=c,label='epsilon='+str(eps))
            # Original markers are order-preserving sampled coordinates; every
            # point and its independent MC SE remain in the underlying CSV.
            idx=np.unique(np.concatenate((np.arange(0,len(s),5),[len(s)-1])))
            ax.plot(s[idx],j[idx],'o',color=c,ms=3,mfc='none')
        decorate(ax,'s','Value / policy cost');ax.legend(fontsize=8);save(fig,'figure9_epsilon_sweep')
        fig,ax=plt.subplots(figsize=(7,4))
        for key in keys:
            rs=read_csv(out/(key+'_rel_l1err_band.csv'));ax.plot(arr(rs,'it'),arr(rs,'mean'),label=key)
        decorate(ax,y='Difference to epsilon=0 reference',log=True);ax.legend(fontsize=8);save(fig,'epsilon_reference_difference_history')
