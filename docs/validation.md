# Reproduction validation, 2026-10-23

Version 0.2.0-rc2 repairs the executable package and aligns saved-data outputs
with the submitted five-run cohorts. The manuscript PDF is unchanged
(SHA-256 `97a48ed973b0a6eaa2d5e6dd699e5dfab24ac3ca97ef26ca39a5c54fb5937f7b`).

The package contains 41 cases and 125 selected runs. The original INIs and
historical numerical sources are retained. Twenty-five portable epsilon INIs
restore the actual submitted five-run inputs, changing only their output
directory. `configs/submitted/epsilon/SOURCE.json` pins both versions. Endpoint
terminal weights are 0.5, as in those original inputs; intermediate epsilon
weights remain 1.0. No scientific option was selected anew.

## Repairs

- Full-mode freezing includes the selected engines and their import closure.
- Case routing uses the recorded source family, including the scalar-control
  reference and all twenty timing configurations.
- HJB path evaluations use each run's own learned feedback, all five runs,
  and matching per-run reference arrays. The archived reader accepts their
  actual five-by-4096 schema.
- HJB plotting files use the submitted seeds 1–5, and the d=10000 HJB-2/3
  plotting references use their saved native 151-point grids. Terminal and
  initial references retain their separate coordinate grids.
- Rho collection uses the final native RC and no longer depends on an empty
  supplemental origin-cost list. Table 1 exports mean solver time.
- Figure 9 uses all five saved runs for both value and policy-cost profiles,
  and computes sample SD across runs separately from conditional MC SE.
- Paired-loss reductions clone detached storage before summation, preserve
  stopped-gradient scaling, and account for unequal rank counts. Sampling
  assigns the two groups to disjoint pilot paths. Global branch, pilot and MC
  budgets are split exactly, with no rounding-up of the N=128 timing batch.

## Checks and their limits

`verify_package.py` checks hashes, INIs, case routing and source import closure.
The standard test suite exercises orchestration and existing-data analysis.
`audit_archived.py` independently recomputes Table 1's five rows (including
printed rounding), timing statistics, comparison bands, scalar diagnostics
and five-run profile statistics. All twelve archived example entry points are
also exercised, including continuation and input/artifact validation.

`compute_validation.json` records the separate dedicated-node synthetic CPU
checks: imports of all four runtime families; two-process Gloo means and
gradients; exact and unequal budgets including a zero-share reference rank;
disjoint pilot sampling; and an independent scalar Riccati/transport ODE
reference with terminal and gradient identities. It records Python and PyTorch
versions. These tests do not train the paper networks or measure performance.

No full-scale GPU retraining is claimed by this repair. Correcting a loss,
sampler or reference can change a newly trained network even with the same
seed. Historical measurements, failures and source snapshots are not rewritten
as measurements of the repaired executable. Archived mode reproduces saved
numerical values; fresh mode produces new measurements under the frozen
scientific settings, with the implementation defects repaired.

The actual submitted runs also have more specific settings than the paper's
general algorithm prose: the INIs record terminal supervision and individual
learning rates; native Table 1 RE is measured on the logger's diagonal grid.
The additional S0/S1 diagnostic is separately named and never substituted for
that native field. Figure 4 retains its saved normalized time curves and
distinct native clocks. See [metrics](metrics.md). These historical conventions
are documented explicitly rather than changing scientific parameters or the
manuscript to conceal them.

The unchanged bundled PDF predates the final Figure 4 axis-layout adjustment.
The saved comparison curves retain the same numerical data. Reproduction here
concerns these values and definitions, rather than pixel-identical PDF layout.
