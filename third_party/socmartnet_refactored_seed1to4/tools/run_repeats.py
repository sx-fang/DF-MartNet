"""Numerical implementation implementation."""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--aggregate', metavar='OUT_CSV', default=None,
                    help="aggregate the CSVs after '--' instead of emitting "
                         'commands')
    ap.add_argument('rest', nargs=argparse.REMAINDER,
                    help='after --: run.py args (emit mode) or per-seed CSV '
                         'glob (aggregate mode)')
    args = ap.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ['--'] else args.rest

    if args.aggregate:
        csvs = sorted(p for pat in rest for p in glob.glob(pat))
        if not csvs:
            sys.exit('error: no CSV matched')
        agg = ROOT / 'tools' / 'aggregate_summary.py'
        subprocess.run([sys.executable, str(agg), '-o', args.aggregate]
                       + csvs, check=True)
        return

    for seed in args.seeds:
        cmd = [sys.executable, str(ROOT / 'run.py')] + rest \
            + ['--seed', str(seed), '--tag', f's{seed}']
        print(' '.join(cmd))
    print(f'\n# then aggregate with:\n# {sys.executable} '
          f'{ROOT / "tools" / "aggregate_summary.py"} -o SUMMARY.csv '
          f'<the {len(args.seeds)} CSVs above>')


if __name__ == '__main__':
    main()
