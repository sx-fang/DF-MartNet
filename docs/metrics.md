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
float32 feedback/value nets, and a shared M=1000000 reference. The historical
HJB-2/narrow pools contain seeds2–5 and use seed2 as guide; the other HJB pools
use their first selected seed as guide. The source choice and sample counts are recorded with each output.

The scalar-control QQ plot pools the five prescribed runs and all 31 interior
layers, each containing 4096 states. True and predicted values are independently
sorted for QQ. The historical order-statistic stride is `max(1,N//500)`, with
both extreme order statistics retained. Per-seed path RE is computed on raw
paired values, not on sorted or subsampled QQ data.

Epsilon choices 1, 1/2, 1/4, 1/8, 0 each use seed1. The value comparator is
the epsilon=0 reference. `rel_l1err` gives the relative difference to this
comparator. Origin `rc_signed_fp32` compares policy cost to the same reference.
Policy-cost profiles use4096 paths per point,101 S0 points, and seeds
20260921 + 10000*epsilon_index + point_index. Monte Carlo SE is separate from
seed SD. No across-epsilon SD is computed.

Training curves use their saved samples without smoothing. Figure 4 preserves
its original exception: each run's time is normalised to the group median final
time and errors are linearly interpolated onto 400 time points. DF's time is
`rt_solve+rt_log`; official SOC-MartNet's is its native in-loop `rt`, excluding
reference precomputation. The underlying per-run curves and unnormalised times are also exported.

Table 1 RT uses median native `rt_solve`. Timing sweeps use 150 consecutive
outer-iteration deltas at iterations51–200, after the first50 intervals, and
report median and MAD. The original Figure 6/7 harvest uses the maximum of the
first native GPU-memory column (rank0) over saved iterations, in MiB. The
timing output also names the all-rank maximum separately; these are allocated
memory readings, not device utilisation. One timing repeat has
no between-repeat SD. Figure 6 uses its historical mean runtime/memory ratios,
not Table 1's median runtime. Hardware and dtype accompany timing output.

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
