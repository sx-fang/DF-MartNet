# Licence scope

## Author-maintained materials

DF-MartNet code in `paper_original/` and `corrected/`, the publication adapters,
experiment scripts, author-maintained documentation, configurations and curated
experiment data are released under the [MIT licence](LICENSE). The root licence
uses the collective attribution "DF-MartNet and SOC-MartNet contributors".
Paper authors and the preferred citation remain in `CITATION.cff`.

## Manuscript

`docs/paper.pdf` is the manuscript snapshot accompanying this numerical
reproduction package. The manuscript and its figures are excluded from the
root MIT licence. Copyright remains with the respective rights holders;
including the PDF grants no additional licence to the manuscript or its figures.
The snapshot date and file identity are recorded in `docs/paper_snapshot.json`.

## SOC-MartNet reference implementations

`third_party/socmartnet_refactored/` and
`third_party/socmartnet_refactored_seed1to4/` contain the complete runtimes used
for the paper's SOC-MartNet comparison, with separate seed0 and seed1–4 entry
variants. Both directories are released under the root [MIT licence](LICENSE).
Their `SOURCE.json` records identify this licence and pin the execution files.
The comparison entry point uses these bundled runtimes directly.

## Data and weights

The release candidate contains no public download URL for large historical
weights. Fresh full runs produce their own final weights. Historical raw logs,
state matrices and full QQ pairs remain outside this package; the small curated
data and SHA provenance are sufficient for the included redraw mode.
