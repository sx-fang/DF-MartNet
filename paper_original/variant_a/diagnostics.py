"""Post-training diagnostics: curve data + summary statistics.

Every figure is paired with a CSV containing all plotted quantities (house
rule: figures must be reproducible from CSVs).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def _is_save_res():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def res_on_curve(v_theta, v_true_func, xcurve_gen, sav_prefix):
    """Save v_theta vs reference along the initial-point curves.

    Writes ``{sav_prefix}resonline_data.csv`` (s, v_true, v_theta for each
    curve) and a comparison PDF per curve.
    """
    curve_name, s_true, xs_true = xcurve_gen(num_points=101)
    _, s_appr, xs_appr = xcurve_gen(num_points=101)

    s_true = [s.detach().cpu().numpy() for s in s_true]
    s_appr = [s.detach().cpu().numpy() for s in s_appr]
    vtrue_xs = [v_true_func(x).detach().cpu().numpy() for x in xs_true]
    
    
    mod = getattr(v_theta, 'module', v_theta)      
    flag = getattr(mod, 'enable_autocast', None)
    if flag is not None:
        mod.enable_autocast = False
    try:
        with torch.no_grad():
            vappr_xs = [v_theta(x).detach().cpu().numpy() for x in xs_appr]
    finally:
        if flag is not None:
            mod.enable_autocast = flag

    if not _is_save_res():
        return

    res_header, res_col = [], []
    for cname, strue, vtrue, sappr, vappr in zip(curve_name, s_true,
                                                 vtrue_xs, s_appr, vappr_xs):
        res_header.extend([
            f's_of_{cname}_for_vtrue', f'vtrue_{cname}',
            f's_of_{cname}_for_vappr', f'vappr_{cname}'
        ])
        res_col.extend([strue, vtrue.squeeze(-1), sappr, vappr.squeeze(-1)])
    Path(sav_prefix).parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(res_col).transpose().to_csv(sav_prefix + 'resonline_data.csv',
                                             index=False,
                                             header=res_header)

    for strue, vtrue, sappr, vappr, cname in zip(s_true, vtrue_xs, s_appr,
                                                 vappr_xs, curve_name):
        plt.figure()
        plt.scatter(sappr,
                    vappr,
                    label='MartNet',
                    color='red',
                    marker='o',
                    facecolors='none')
        plt.plot(strue, vtrue, label='Reference', color='blue')
        plt.legend()
        plt.savefig(f'{sav_prefix}resonline_{cname}.pdf')
        plt.close()


def summary_hist(hist_list):
    """Mean and std across repeat runs, grouped by iteration."""
    df_list = [pd.DataFrame(hist) for hist in hist_list]
    hist_df = pd.concat(df_list, axis=0, keys=range(len(df_list)),
                        names=['round', 'index'])
    hist_gp = hist_df.groupby('it')
    summ_df = pd.concat([hist_gp.mean(), hist_gp.std()],
                        axis=1,
                        keys=['mean', 'std'])
    return summ_df.swaplevel(0, 1, axis=1).sort_index(axis=1)


def plot_hist_summary(summ_df, sav_prefix):
    """Log-scale relative-error and loss curves from the summary table."""
    it_arr = np.array(summ_df.index)
    for key, fname, yscal in [('rel_l1err', 'rel_l1err', 'log'),
                              ('rel_linferr', 'rel_linferr', 'log'),
                              ('rel_l1err_u', 'rel_l1err_u', 'log'),
                              ('rel_linferr_u', 'rel_linferr_u', 'log'),
                              ('pde_loss', 'pde_loss', 'log'),
                              ('ctr_loss', 'ctr_loss', 'linear')]:
        if (key, 'mean') not in summ_df.columns:
            continue
        mean = summ_df[key, 'mean'].to_numpy(dtype=float)
        std = summ_df[key, 'std'].to_numpy(dtype=float)
        if not np.isfinite(mean).any():
            continue
        plt.figure()
        plt.plot(it_arr, mean, color='blue')
        plt.fill_between(it_arr, mean - std, mean + std, alpha=0.3)
        plt.yscale(yscal)
        plt.xlabel('iteration')
        plt.ylabel(key)
        plt.savefig(f'{sav_prefix}summary_{fname}.pdf')
        plt.close()
