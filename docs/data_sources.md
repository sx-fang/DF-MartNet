# Source and numerical data

`paper_manifest.csv` maps 125 selected runs in 41 configurations to public
sample IDs, seeds, repeats, GPU counts, configuration hashes and source
snapshots. The twelve entry points select these runs. HJB group and individual
entries share configurations. The package retains 145 original/revision DF INIs, adds
25 portable copies of the submitted epsilon INIs, and contains ten
reference-method CLI records. Labels such as `seed1_r0` identify samples.

Code pinned to a named commit lives under `engines/`. Fresh DF execution uses
its separate `runtime_*` repair copies; `base_source` identifies each immutable
historical snapshot. Official comparison code remains under `third_party/`
and runs with the explicitly prescribed eight GPUs. `SOURCE.json` records
module hashes; `ENGINE_PINS.csv` records case routing. Public hashes describe
the bundled files and remain distinct from their original-input hashes.

Adjacent `*_PROVENANCE.json` files record sample IDs and input hashes.
`data/ASSET_PROVENANCE.json` records content-source IDs and bundled-file
SHA256 values. `PACKAGE_FILES.json` pins the publication files; Git preserves
their bytes without line-ending conversion.

Saved histories contain the numerical columns used by the paper's plots.
HJB assets contain per-run prediction/reference arrays and a time-layer index;
`path_protocol.json` binds their five-row ordering and 128-by-32 layout.
`HJB_PLOT_SOURCES.json` pins the submitted final plotting cohort. Figure 2's
wide references have separate initial/terminal coordinates. Full mode measures
accuracy on new states and references; the wider display-only reference is a
frozen plotting input and is labelled as such in evaluation metadata.

Scalar-control QQ files contain pooled order statistics, both extremes and
per-seed raw-pair RE on the 31-layer QQ interior. Full path metrics remain
separate. Epsilon `per_seed_profiles.csv` exposes all 25 saved runs, including
pointwise predictions, costs and conditional MC SE; the aggregate uses
across-run sample SD. Archived epsilon training histories cover seed 1 only;
this subset is labelled and does not supply a five-seed history band.
Comparison inputs contain the selected histories and their
aggregate curves. Counts and statistic definitions accompany the data.

All bundled-data inputs are included. Full mode produces final weights during
training; it requires no historical checkpoint download or author account.
DF-MartNet code, the bundled SOC-MartNet reference implementations and curated
data use MIT; see `LICENSE_STATUS.md` for the manuscript's separate scope.

The release also carries a `raw-data.tar.gz` asset; `data/RAW_MANIFEST.json` records its members, sizes and hashes.
