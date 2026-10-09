# DF-MartNet paper reproduction package

This package provides the numerical examples for Table 1 and Figures 2–9 of
*Martingale deep neural network for very high-dimensional stochastic optimal
controls*. Each example script connects training, final-weight export,
evaluation, tables and figures. Bundled numerical data also support direct
regeneration of the paper's tables and plots.

The reproduction target is the [bundled manuscript](docs/paper.pdf), dated
October 9, 2026. The [arXiv preprint](https://arxiv.org/abs/2408.14395) is the
earlier public version. The manuscript's date, file hash and package version
are recorded in [snapshot metadata](docs/paper_snapshot.json).

**Version: 0.1.0-rc2.** DF-MartNet code, the bundled SOC-MartNet reference
implementations, documentation and curated data use the [MIT licence](LICENSE);
see [licence scope](LICENSE_STATUS.md).
The manuscript PDF and its figures have separate copyright and are excluded
from the package's MIT licence.

The default source choice, `paper_original`, uses the supplied source snapshots
and experiment configurations for paper reproduction. `--version corrected`
selects the alternative path-group sampling and DDP-statistics implementation
with the same supplied numerical settings. Run outputs record the source choice;
bundled-data mode uses the paper's saved numerical data.

## Run one example

Download this repository, or clone it:

```bash
git clone https://github.com/sx-fang/DF-MartNet.git
cd DF-MartNet
```

Install the Python environment once; see [environment](docs/environment.md).
On a Linux CUDA compute node with the prescribed GPU allocation:

```bash
python experiments/hjb1_d10000/run.py --output-root /your/scratch/results
```

For a dedicated compute node without Slurm, first set `DFM_COMPUTE_NODE=1`.
This declaration must not be used on a login node. There is no CPU or
reduced-GPU fallback. All selected runs execute sequentially within the case
allocation; a group entry also includes the dependent evaluations and figures.
Use separate entry points to schedule independent examples concurrently. Choose
the HJB group entry or the individual HJB entries; results in another entry's
run directory are not silently imported as new training.

For Slurm, copy `slurm/site.example.json` to `slurm/site.json` and fill in your
account, GPU partition, interpreter and wall-time limit. Then run:

```bash
python experiments/hjb1_d10000/run.py --backend slurm --site slurm/site.json --output-root /your/scratch/results
```

The submitter freezes a private package copy, calls `slurm/submit_job.sh`, waits
for the recorded job, and checks its terminal state and output hashes. Solver,
inference and Monte Carlo work run only in the compute allocation. No jobs are
cancelled automatically. Sites with additional archiving rules should connect
their epilogue when configuring this generic launcher.

Inspect a plan without numerical execution:

```bash
python experiments/hjb1_d10000/run.py --dry-run
```

To redraw the paper's **saved data**, on a workstation or compute node:

```bash
python experiments/hjb_table/run.py --mode archived --output-root ./redraw
```

Archived mode needs only NumPy and Matplotlib. It recomputes statistics from
the bundled saved data and draws figures; it never loads a solver or a network.
The small curated data are included: no private NAS or author account is
required.

## Example entry points

| Script under `experiments/` | Paper objects | Maximum GPUs per stage |
|---|---|---:|
| `hjb1_d10000/run.py` | Table 1 row 1; Figure 2 HJB-1 panels | 8 |
| `hjb2_d10000/run.py` | Table 1 row 2; Figure 2 HJB-2 panels | 8 |
| `hjb3_d10000/run.py` | Table 1 row 3; Figure 2 HJB-3 panels | 8 |
| `hjb3_d2000_w2010/run.py` | Table 1 row 4; Figure 3 narrow network | 8 |
| `hjb3_d2000_w10010/run.py` | Table 1 row 5; Figure 3 wide network | 8 |
| `hjb_table/run.py` | Complete Table 1 and Figures 2–3 | 8 |
| `comparison/run.py` | Figure 4; both methods, both equations | 8 |
| `complex_dynamics/run.py` | Figure 5; native final metrics | 4 |
| `rho_sweep/run.py` | Figure 6; five dimensions of the test network | 4 |
| `performance_sweep/run.py` | Figure 7; 20 short timing configurations | 4 |
| `scalar_control/run.py` | Figure 8; value/control path metrics and QQ | 4 |
| `epsilon_sweep/run.py` | Figure 9; one seed per epsilon | 4 |

The seeds, selected repeats, world sizes, configuration
hashes, source variants and public sample identities are in
[`paper_manifest.csv`](paper_manifest.csv). HJB-3 uses the supplied
seed pool 1, 2, 3, 5, 6. Official comparison seeds are 0–4. The manifest gives
the GPU allocation for each individual run. Selected runs are retained in the
output tables and figures.

## Outputs and continuation

Each invocation writes `results/<example>/<run-id>/` (or `--output-root`):

```text
cases/                   actual INIs, native histories and final weights
source/                  immutable numerical-source snapshot (full mode)
raw/                     resource check / Slurm / console evidence
stages/                  input pins, completion states and artifact hashes
tables/                  per_seed.csv, summary.csv, table1.csv and LaTeX rows
figures/                 PDF, PNG, and the CSVs used to draw them
run_manifest.json        version, mode, seed pools, state and output hashes
report.md                output index and metric definitions
```

Final values are used in tables; histories are unsmoothed. Sample SD across
seeds, timing MAD and Monte Carlo SE are separately named. Single-seed results
have no cross-seed SD. Read [metric definitions](docs/metrics.md), especially
the distinct comparison clocks and epsilon reference semantics.

`--resume --run-id <id>` reuses only completed, hash-verified stages. Changed
inputs or artifacts are rejected. Failed/incomplete numerical stages are not
automatically retried, and there is no equivalent mid-training restart without
optimizer/RNG state. An existing Slurm job record prevents duplicate submission.

## Program variables and paper notation

The table links the DF-MartNet configuration, numerical code and exported
CSV/NPZ fields to the notation in the paper's problem setup, Algorithm 1 and
numerical examples. Let $d$ be the state dimension, $m$ the control dimension,
and $q$ the Brownian dimension. Code inputs named `x` contain both time and
state; the paper's spatial variable $x$ contains only the state. Native log
fields and publication diagnostics are listed separately where their
evaluation sets or definitions differ. Unless explicitly labelled as a
percentage, RE/RC fields store ratios: multiply by 100 to display percent.

| Program/configuration/output field | Paper notation | Meaning and reading convention |
|---|---|---|
| `[Example] dim_x`, `problem.dim_x` | $d+1$ | Network input dimension, including time. For example, `dim_x=10001` means $d=10000$. |
| `problem.dim_z`; exported `d` | $d$ | Spatial state dimension. `dim_z` is not a general definition of $q$; diffusion may have lower rank. |
| `problem.dim_u` | $m$, $U\subset\mathbb{R}^m$ | Control output dimension; $m=1$ in the scalar-control example. |
| `x[...,0]`, `x[...,1:]`; saved NPZ `x` | $(t,x)$; state $x$ or $\bar X_n^j$ | First column is time; the remaining $d$ columns are the state. The same layout is used for saved deployment states. |
| `problem.t0`, `problem.te` | $t_0$, $T$ | Initial and terminal times. |
| `[Training] dt`, time-step index; `n_levels`, `t_levels` | $\Delta t$, $n$, $N$, $t_n$ | $N=(T-t_0)/\Delta t$. Evaluation files record their actual saved layers; QQ files may omit the initial/terminal layers. |
| `v_theta`, its parameters; `u_alpha`, its parameters | $v_\theta$, $\theta$; $u_\alpha$, $\alpha$ | Learned value function and feedback control. Code `kappa` is a control value $\kappa=u_\alpha(t,x)$. |
| `rho`, its parameters | $\rho_\eta$, $\eta$ | Adversarial test network in the martingale loss. |
| `problem.v`, `problem.u_star` | $v$, $u^*$ | Exact or case-specific reference value/control, when available; they are not learned predictions. |
| `problem.mu(x,kappa)`, `problem.sigma(x,dw)` | $\mu(t,x,\kappa)$, $\sigma(t,x,\kappa)\Delta B$ | The diffusion method returns the diffusion applied to an increment, rather than the diffusion matrix itself. |
| `problem.running_cost(acc,x,kappa)`, `problem.g`, `terminal_cost` | $c(t,x,\kappa)$, $g(x)$ | `running_cost` returns the accumulator plus the running cost; `terminal_cost` extracts the spatial part of its time/state input. |
| `[Training] num_pilot_paths` | $M$ | Global retained pilot-path count. `problem.num_pilot_paths` is the rank-local share in DDP. |
| `[Training] num_xpts_batch`, `problem.batch_size` | Batch sampling associated with $\mathbb A_1,\mathbb A_2$ | Global branch-point budget and its rank-local share. These count time/state branch points, not the path-index counts $\lvert\mathbb A_i\rvert$; do not identify them by dividing by $N$ without checking the sampler. |
| `world_size`, manifest `gpus` | Number of GPUs per run | DDP worker count; a resource setting, not a path or seed count. |
| `[Network] width_v`, `width_u`; table `width_v` | $n_{\mathrm{unit}}$ | Hidden-layer widths of the value/control inner networks. Read each separately if the case uses different widths. |
| `num_hidden_v`, `num_hidden_u`; table `hidden_v` | $n_{\mathrm{layer}}$ | Hidden-layer counts of the value/control inner networks. |
| `width_rho` | $r$ | Test-network output dimension, including when `num_hidden_rho=0`. |
| `scale_ub_rho`, `scale_lb_rho`, `rho_shell` | Scaling parameter $c$, scales $c_i$, sine shell of $\rho_\eta$ | The supplied sine-shell settings use scales from 1 to $c$. This $c$ is distinct from the running-cost function $c(t,x,\kappa)$. |
| `[Training] num_descent`, `num_ascent` | $J$, $K$ in Algorithm 1 | Number of value/control descent substeps and test-network ascent substeps per outer cycle; `2,1` corresponds to $J=2K=2$. |
| `max_iter`; native `it`, exported `final_it` | $I$; iteration index $i$ | `it=0` is recorded after the first optimisation cycle. The preserved code runs labels `0..max_iter`, hence `max_iter+1` cycles; the label must not be read as the completed-cycle count. |
| `[Optimizer] lr0_v`, `lr0_u`, `lr0_rho` | Initial $\delta_0$, $\delta_1$, $\delta_2$ in Algorithm 1 | Initial learning rates for $v_\theta,u_\alpha,\rho_\eta$, respectively. Algorithm 1's $\delta_0$ differs from the dimension-dependent prefactor also called $\delta_0$ in the parameter-settings paragraph. |
| `decay_stepgap`, `decay_rate` | Iteration-dependent learning-rate decay | Discrete scheduler interval and multiplier implementing the decay; consult the actual INI rather than assuming continuous decay at every cycle. |
| `[Example] M_rc`; diagnostic `M_cost`, cost-profile `M` | $M_{\mathrm{test}}$ for $\hat J$ | Number of policy-cost Monte Carlo paths per starting point. Distinct from the retained training paths $M$. |
| Evaluation `n_paths`; archived path identifiers | $M_{\mathrm{test}}$ for path RE | Number of deployment paths. HJB/complex path-value tests use 128, whereas the scalar-control test uses 4096; this need not equal the cost-path count. |
| `[Example] M_mc`; reference metadata `M_mc` | Reference Monte Carlo sample size | Reference-solution budget, distinct from both training $M$ and deployment $M_{\mathrm{test}}$. |
| `problem.eps`; epsilon-case `name` | $\epsilon$ | Perturbation parameter selected by the problem class: `CoupledDriftS0`, `CoupledDriftEps2S0`, `CoupledDriftEps4S0`, `CoupledDriftEps8S0`, `CoupledDriftEps0S0` give $1,1/2,1/4,1/8,0$. |
| Profile `curve`, `s`; curve names `diag`, `manifold` | $S_0$, $S_1$, $s$ | `diag` parameterises $x=s\mathbf 1_d$; `manifold` parameterises the paper's $x=\mathbf l(s)$. Use the named curve and its saved coordinates together. |
| NPZ `v_pred`, `v_true`; archived `v_hat_all`, `v_star`; profile `vappr_diag`, `vtrue_diag` | $v_\theta(t,x)$, $v(t,x)$ | Predictions and corresponding references. `v_hat_all` contains the selected run axis; reference identity and state coordinates must match before comparing. |
| Scalar-control NPZ `u_pred`, `u_true` | $u_\alpha(t,x)$, $u^*(t,x)$ | Learned and reference controls on paired deployment states. |
| Native `s_of_diag_for_vappr`, `s_of_diag_for_vtrue`; curated `v_mean`, `v_reference` | $s$; mean learned profile, reference $v(0,x)$ | Native prediction/reference coordinates are stored separately. Curated profiles align them before aggregating runs. |
| Profile `error_mean`, `error_sd` | Pointwise error $v_\theta(0,x)-v(0,x)$ and its SD | Signed point error; not an absolute error, relative error, or cost gap. |
| Native `rel_l1err`; per-run/table `re_native` | Native counterpart of $\mathrm{RE}_v^{t_0}$ | Final logger relative L1 error on the actual problem grid. For HJB this is the diagonal grid, not the paper's combined uniform grids on $S_0\cup S_1$. |
| Diagnostic `re_t0_s0s1` | $\mathrm{RE}_v^{t_0}$ | Relative L1 error on 100 equispaced points on each of $S_0,S_1$. It is separately named and does not replace Table 1's native field. |
| `re_path`, `re_v_path`; native `rel_l1err_ualoop` | $\mathrm{RE}_v^{\mathrm{path}}$ | Pooled absolute value error divided by pooled absolute reference over paired deployment states. Use the specified saved layers and path pool. |
| `re_u_path`; native `rel_l1err_u_ualoop` | $\mathrm{RE}_u^{\mathrm{path}}$ | Analogous pooled relative L1 control error; QQ sorting/subsampling is not used to compute it. |
| Native `rc`; exported `rc_native_signed` | Native counterpart of $\mathrm{RE}_{\mathrm c}$ | Signed native cost error; the native reference denominator may itself be signed. |
| `rc_abs` | $\lvert\mathrm{RE}_{\mathrm c}\rvert$ as reported in Table 1 | Absolute native final RC. Scalar-control Figure 8 instead preserves the mean of signed native RC. |
| Diagnostic `rc_signed_fp32` | $\mathrm{RE}_{\mathrm c}$ with the stated reference | $(\hat J-V_{\mathrm{ref}})/\lvert V_{\mathrm{ref}}\rvert$, using strict-float32 feedback. For nonzero epsilon it is a cost gap to the epsilon=0 comparator, not true perturbed RC. |
| `jhat_origin_fp32`, profile `jhat`; `v_reference_origin` | $\hat J(u_\alpha)$; $v(0,x_0)$ | Empirical cost at the origin or named profile point and the corresponding origin reference. |
| NPZ `per_path_cost`; epsilon profile `gap` | $\sum_n c_n^j\Delta t+g(\bar X_N^j)$; $\hat J-v_\theta$ | Individual rollout cost; `gap` is the difference between policy cost and learned value. |
| `jhat_mc_sd`, `jhat_std`; `jhat_mc_se`, `jhat_se`, `band2se` | Cost-sample SD; Monte Carlo SE | Fresh SE is centred sample SD divided by the square root of the cost-path count; `band2se` is twice that SE. Historical `jhat_group_mc_se` uses saved group means and is named separately. |
| Summary `mean`, `sample_sd`; Table 1 `*_sd`; profile/history `sd`, `ddof` | Mean and SD across the selected runs | Metric sample SD uses `ddof=1`; one run has no cross-seed SD. Historical complex-profile bands use the recorded `ddof=0`. These SDs are distinct from Monte Carlo SE. |
| `seed`, `repeat`, `arm`; summary `n`, Table 1 `*_n` | Independent-run identities and available-run count | Read counts by context: profile/summary `n` counts runs; path-layer `n` counts states; timing `n_deltas` counts intervals. None is training $M$. |
| Native `rt_solve`; table `rt_solve_median`, `rt_solve_MAD` | RT (seconds), with the published table clock | Cumulative solver time excluding metric/logging time. Table 1 uses the median across runs; MAD is the median absolute deviation. |
| Native `rt_log`, DF `rt_total`; official `rt` | Separate wall-clock components / Figure 4 time axis | DF comparison time uses `rt_solve+rt_log`; official `rt` is its native in-loop clock excluding reference precomputation. The clocks are not interchangeable end-to-end times. |
| Timing `median_s`, `MAD_s`, `n_deltas` | Figure 7 per-iteration runtime and dispersion | Median/MAD of 150 solver-time deltas at iterations 51–200, after 50 warm-up intervals; not cross-seed SD. |
| Native `peak_memory_cuda*_MB`; `peak_mem`, `peak_mem_all_ranks` | GPU memory in Figures 6–7 | Values are MiB despite the native `MB` suffix. Published harvest uses the peak of the first GPU column; the all-rank peak is exported separately. |
| QQ `q_reference`, `q_prediction`, `index`, `n_pooled` | Reference/prediction order statistics in Figure 8 | Independently sorted samples with the recorded stride and extremes retained; these rows are not paired statewise errors. |
| Native `pde_loss`, `ctr_loss`, `lam_mart` | Diagnostics associated with the empirical $L$ objectives | Implementation loss diagnostics, with residuals normalised by $\Delta t$ and configured weighting. They are not RE/RC; `lam_mart` is an implementation weight without a separate Algorithm 1 symbol. |

In the epsilon experiment, value references at nonzero epsilon are also the
epsilon=0 comparator, so a field called `v_true` or `rel_l1err` does not assert
an exact perturbed solution. For historical/fresh metric details, signed versus
absolute RC, SD bands and timing conventions, see [metric definitions](docs/metrics.md).
The source-class names `HJB1`, `HJB3a`, `HJB3b` correspond to the paper's
HJB-1, HJB-2, HJB-3, respectively.

## Check the package and saved-data outputs

```bash
python scripts/verify_package.py
python scripts/audit_publication.py
python -B -m unittest discover -s tests -v
python scripts/audit_archived.py --output ./audit/archived.json
```

These commands check file hashes, source/configuration structure, orchestration
and saved-data calculations. The archived-data check recomputes the table and
figure statistics from the bundled arrays and CSVs.

## Citation and publication

Citation metadata are in `CITATION.cff`. Use release/tag name `v0.1.0-rc2`;
Python represents this version as `0.1.0rc2`. DF-MartNet and the bundled
SOC-MartNet reference implementations use MIT; see `LICENSE_STATUS.md`.
When referring to the reproduced tables and figures, identify the bundled
manuscript snapshot as well as the code version.

Publish this directory alone into a new repository; see
[publication boundaries](docs/publishing.md). Local scheduling files and generated
results are excluded from the release inventory.
