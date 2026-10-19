# Source and numerical data

`paper_manifest.csv` maps 105 selected runs in 41 configurations to public
sample IDs, seeds, repeats, GPU counts, configuration hashes and source
snapshots. The twelve entry points select these runs. HJB group and individual
entries share configurations. The package includes 95 DF INIs and ten
reference-method CLI records. Labels such as `seed1_r0` identify samples.

Code paths pinned to a named engine commit live under `engines/` (Figure 6 uses `engines/af4a053`; the HJB tables and Figures 2-3 use `engines/4d30ed0`), each with a `SOURCE.json`; the `paper_original/` and `third_party/` directories carry the older original and public file hashes. These snapshots provide the numerical executors used by the example
entries. Public hashes describe the bundled files and remain distinct from
their original-input hashes.

Adjacent `*_PROVENANCE.json` files record sample IDs and input hashes.
`data/ASSET_PROVENANCE.json` records content-source IDs and bundled-file
SHA256 values. `PACKAGE_FILES.json` pins the publication files; Git preserves
their bytes without line-ending conversion.

Saved histories contain the numerical columns used by the paper's plots.
HJB assets contain prediction, reference and time arrays for the selected
samples. Figure 2's wide reference profiles are bundled separately from each
run's final-profile arrays. Full mode generates states, predictions, references
and per-path costs from the current run.

Scalar-control QQ files contain pooled order statistics, both extremes and
per-seed raw-pair RE. Epsilon cost files contain the saved points, mean, SD and
Monte Carlo SE. Comparison inputs contain the selected histories and their
aggregate curves. Counts and statistic definitions accompany the data.

All bundled-data inputs are included. Full mode produces final weights during
training; it requires no historical checkpoint download or author account.
DF-MartNet code, the bundled SOC-MartNet reference implementations and curated
data use MIT; see `LICENSE_STATUS.md` for the manuscript's separate scope.

The release also carries a `raw-data.tar.gz` asset; `data/RAW_MANIFEST.json` records its members, sizes and hashes.
