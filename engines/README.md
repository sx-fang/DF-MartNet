# Numerical sources

`4d30ed0` and `af4a053`, and every directory under `paper_original/`, are
unchanged historical snapshots. They identify the code behind the saved runs.
`ENGINE_PINS.csv` and each arm's `base_source` preserve that identity.

Fresh execution uses four portable repair copies:

| Runtime | Historical base | Cases |
|---|---|---|
| `runtime_4d30ed0` | `4d30ed0` | HJB, timing, scalar control, epsilon |
| `runtime_af4a053` | `af4a053` | Test-network dimension sweep |
| `runtime_complex` | `paper_original/variant_a` | Complex dynamics |
| `runtime_comparison` | `paper_original/variant_d` | DF method comparison |

The repairs provide the full import dependency set, disjoint pilot path
groups, an alias-safe stopped-gradient loss, and exact global sample budgets
with sample-weighted distributed reductions. The scalar-control reference is
the scalar-control implementation already present in `4d30ed0`. Configured
networks, coefficients, seeds, iteration counts and optimizers are retained.

Each `SOURCE.json` pins the historical base and every executable module.
Optional aligned/radial linear and fused optimizer modules are included so
conditional imports cannot fail. Full-mode runs freeze these sources, their
configurations and evaluation inputs before launching workers. See
[validation](../docs/validation.md) for test scope and historical limitations.
