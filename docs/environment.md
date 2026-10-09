# Environment and resources

The paper reports Python 3.12, PyTorch 2.5.1 and A100-SXM4-80GB GPUs.
Use a Linux CUDA compute node with the GPU allocation specified for each run
in `paper_manifest.csv`.

Install PyTorch using its official
[installation instructions](https://pytorch.org/get-started/locally/) with a
compatible NVIDIA driver and CUDA wheel. Install the remaining dependencies
from `requirements.txt`. Example scripts use the existing environment and do
not install packages. Python 3.12 is the numerical baseline.

Numerical execution uses distributed CUDA, autocast and dense neural networks.
The complete configurations and iteration counts are supplied in the package;
CPU execution is disabled. Group entries execute serial stages in one allocation.
Independent examples can be scheduled in separate allocations.

Bundled-data mode uses NumPy and Matplotlib. Numerical executors also use
Pandas, SciPy, psutil and PyTorch. The reference comparison uses its included
`run.py` and `socmartnet/` runtime.

The publication-file checker uses `pypdf` to inspect the bundled manuscript;
it does not execute numerical code.

Allow ample scratch space: d=10000 final weight files can be several gigabytes
per run. Use `--output-root` on scratch. Runners keep each selected run's weights,
logs and outputs. Configure the editable Slurm template with your partition,
account, interpreter and wall-time limit.
