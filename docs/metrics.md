# Metrics and evaluation contracts

`re_native` is the saved logger's final `rel_l1err`, evaluated on the grid
specified by the problem class and source snapshot.
Native `it=0` is after the first optimisation cycle, not an untrained initial
network. The preserved loop performs `max_iter+1` cycles, so label9000 means
9001 cycles. The release retains the original stopping condition and requires
every saved label0..max_iter in fresh native histories.
For HJB classes this grid is the diagonal curve. `re_t0_s0s1` names the
additional evaluation on 100 equispaced points on each S0/S1 curve.

`rc_native_signed` preserves the native logger, including its signed reference
denominator. `rc_abs` is its absolute value for the paper tables/positive
history curves. The publication diagnostic `rc_signed_fp32` uses
`(Jhat - Vref)/abs(Vref)`. It does not replace native RC in Table 1.

`re_path` pools absolute errors over all deployment states before dividing by
the pooled absolute reference. HJB uses 128 origin-started paths, 32
preterminal layers, a dedicated Brownian stream with seed 20260821, strict
float32 feedback/value nets, and a reference evaluated on each run's own deployment states (M=1000000); the per-run reference arrays are bundled. The historical
arrays contain all five seeds 1–5. Each reference row corresponds to the same
run's learned-control trajectories; no shared guide policy is used. The source
choice, weight hashes and sample counts are recorded with fresh outputs.

The scalar-control QQ plot pools the five prescribed runs and all 31 interior
layers, each containing 4096 states. True and predicted values are independently
sorted for QQ. The historical order-statistic stride is `max(1,N//500)`, with
both extreme order statistics retained. Per-seed path RE is computed on raw
paired values over all 32 preterminal layers including t0, not on sorted or
subsampled QQ data. Fresh QQ-only errors have separate `*_qq_interior` fields.

Epsilon choices 1, 1/2, 1/4, 1/8, 0 each use five independent runs. The value comparator is
the epsilon=0 reference. `rel_l1err` gives the relative difference to this
comparator. Origin `rc_signed_fp32` compares policy cost to the same reference.
Policy-cost profiles use 4096 paths per point, 101 S0 points, and seeds
20260921 + 10000*epsilon_index + point_index, where epsilon_index is
0, 1, 2, 3, 4 for epsilon=1, 1/2, 1/4, 1/8, 0 respectively.
The exact saved seed at every point is in `per_seed_profiles.csv`; all five
training runs use that same cost-evaluation stream at a given coordinate.
Monte Carlo SE is separate from seed SD; the published bands use across-run
sample SD. Value shading and cost error bars both show mean ± 2 SD.

Training curves use their saved samples without smoothing. Figure 4 preserves
its original exception: each run's time is normalised to the group median final
time and errors are linearly interpolated onto 400 time points. DF's time is
`rt_solve+rt_log`; official SOC-MartNet's is its native in-loop `rt`, excluding
reference precomputation. The underlying per-run curves and unnormalised times are also exported.

Table 1 RT uses the mean native `rt_solve`. Timing sweeps use 150 consecutive
outer-iteration deltas at iterations51–200, after the first50 intervals, and
report median and MAD. The original Figure 6/7 harvest uses the maximum of the
first native GPU-memory column (rank0) over saved iterations, in MiB. The
timing output also names the all-rank maximum separately; these are allocated
memory readings, not device utilisation. One timing repeat has
no between-repeat SD. Figure 6 uses the mean value error, mean absolute native RC, mean runtime and median GPU memory of the six-group r campaign. Hardware and dtype accompany timing output.

Sample SD uses ddof=1. The archived complex-profile generator used ddof=0 for
its profile band; that historical choice is preserved and labelled in CSV.
Fresh profiles use ddof=1. Table/path metrics never inherit that profile choice.
The scalar-control Figure 8 preserves the mean of signed native RC, including
negative per-seed readings. Table 1 uses absolute native RC, and the other
RC histories retain their respective historical positive-error convention.
Training-history bands show mean to mean+2SD; point/profile bands show
mean±2SD. These bands are not confidence intervals.

The archived rho origin-cost data contain eight saved group means per seed
(4096 total paths). Their original group SE is preserved as `jhat_group_mc_se`.
The mean and sample SD across five seeds are recomputed from full-precision
group values, rather than the rounded publication summary. Fresh runs store
individual path costs and use centred sample SD divided by sqrt(M); this
diagnostic is explicitly separate from the native logger and old group SE.
Figure 6 RC uses the final native signed RC's absolute value, averaged across
the five selected runs; the supplemental origin-cost diagnostic does not replace it.

Training branch points, pilot paths, policy-cost paths and reference paths are
global counts. Runtime workers split integers by quotient and remainder using
the actual DDP world size, then weight sums and gradients by the actual counts.
The startup log and training record include the global budget, every rank's
share and the budget protocol version. Invalid sampler minimum shares raise an
error rather than enlarging the configured budget. Reference path simulation
uses N=100; deployment paths use the submitted N=32.
