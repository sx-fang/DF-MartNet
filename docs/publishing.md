# Publication boundaries

Publish the files listed in `PACKAGE_FILES.json`, plus that inventory file,
as the root of a new repository with fresh Git history. Preparation reports,
backup copies, local scheduling settings, generated outputs and environments
are outside this package.

## Version and licence

Use release/tag name `v0.1.0-rc2`. README and `CITATION.cff` use `0.1.0-rc2`;
`dfm_repro.__version__` uses the equivalent Python form `0.1.0rc2`.
`cff-version: 1.2.0` identifies the citation-file schema. `paper_original` and
`corrected` identify numerical-source choices.

Add the actual publication date to `CITATION.cff` when releasing. Refresh
`PACKAGE_FILES.json`, repeat the checks below, then create the release tag.
DF-MartNet code, both bundled SOC-MartNet reference runtimes, documentation and
curated data use MIT; see [licence scope](../LICENSE_STATUS.md).

The numerical reproduction target is the bundled `docs/paper.pdf` snapshot.
Freeze that PDF and its `docs/paper_snapshot.json` metadata with the code tag;
both files are included in the inventory. The arXiv link identifies the earlier
public preprint. The manuscript and its figures are outside the root MIT licence.

## Local settings and output

`slurm/site.example.json` contains placeholders. Create `slurm/site.json` with
your local settings; it is excluded from the inventory. Generated logs and
results can contain local machine, task and path information. Check those
outputs separately before sharing them.

Before exporting the package, run:

```bash
python scripts/verify_package.py
python scripts/audit_publication.py
```

The checks verify the locked input files and inspect text, filenames and saved
array metadata. PDF inspection also checks extracted text, document metadata,
links and embedded objects, using `pypdf` from `requirements.txt`. Only the
manuscript authors' contact emails and the verbatim funding attribution recorded
in the snapshot metadata are treated as public scholarly information.
Other occurrences remain subject to the checks. Findings contain rule names and locations,
without credential values.
