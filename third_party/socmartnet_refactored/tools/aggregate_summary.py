"""Aggregate per-seed engine CSVs into the authors' summary CSV format.

Usage:
    python tools/aggregate_summary.py -o summary.csv \
        outputs/SOCMartNet_HJBLQ_..._d100_te1.0_s0.csv [more per-seed CSVs]

Rows are aligned by the 'iter step' column (all inputs must share the same
iteration grid).  For each logged quantity we write a mean and a sample std
(n-1, pandas .std() convention) column, in the authors' savresult.py column
order:

    it, cost_mean, cost_std, g1_loss_mean, g1_loss_std, mart_loss_mean,
    mart_loss_std, mean_vtrue_t0, rel_l1err_mean, rel_l1err_std,
    rel_linferr_mean, rel_linferr_std, rt_mean, rt_std

Quantities missing from the input CSVs (e.g. cost when --cost-track was off)
yield empty fields, mirroring the archived summary files.
"""

import argparse
import csv
import statistics
import sys

# engine CSV column -> summary column stem
COL_MAP = [('cost', 'cost'),
           ('hami', 'g1_loss'),
           ('mart loss', 'mart_loss'),
           ('mean_vtrue_t0', 'mean_vtrue_t0'),
           ('error', 'rel_l1err'),
           ('rel_linf', 'rel_linferr'),
           ('rt', 'rt')]

HEADER = []
for _eng, _stem in COL_MAP:
    if _eng in ('mean_vtrue_t0',):
        HEADER.append(_stem)  # constant per run: mean only (authors' format)
    else:
        HEADER += [f'{_stem}_mean', f'{_stem}_std']
HEADER = ['it'] + HEADER


def read_history(path):
    with open(path, encoding='UTF8') as fh:
        rows = list(csv.DictReader(fh))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('csvs', nargs='+', help='per-seed engine CSV files')
    ap.add_argument('-o', '--out', required=True, help='output summary CSV')
    args = ap.parse_args()

    histories = [read_history(p) for p in args.csvs]
    it_grids = [[row['iter step'] for row in h] for h in histories]
    if any(g != it_grids[0] for g in it_grids[1:]):
        sys.exit('error: iteration grids differ across input CSVs')

    out_rows = []
    for i, it in enumerate(it_grids[0]):
        row = [it]
        for eng, stem in COL_MAP:
            if eng == 'mean_vtrue_t0':
                vals = [float(h[i][eng]) for h in histories
                        if h[i].get(eng) not in (None, '')]
                row.append(f'{statistics.mean(vals):.6e}' if vals else '')
                continue
            vals = [float(h[i][eng]) for h in histories
                    if h[i].get(eng) not in (None, '')]
            if not vals:
                row += ['', '']
            else:
                row.append(f'{statistics.mean(vals):.6e}')
                row.append(f'{statistics.stdev(vals):.6e}'
                           if len(vals) > 1 else '0.000000e+00')
        out_rows.append(row)

    with open(args.out, 'w', encoding='UTF8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(out_rows)
    print(f'wrote {args.out}: {len(out_rows)} rows x {len(HEADER)} cols '
          f'from {len(histories)} runs')


if __name__ == '__main__':
    main()
