# Engine snapshots

`4d30ed0` is the only engine version that the revised figures need on top of
the code paths already bundled in `paper_original/` and `corrected/`.

The nine core modules were taken from the engine repository with `git archive`
and each file was checked with `git hash-object` against the commit tree. They
were then compared byte for byte with the frozen copies archived next to the
runs: the twenty-five HJB runs, the twenty timing runs and the earlier seed-1
batch all carry the same core modules, so one snapshot serves every revised
figure.

```
engines/4d30ed0/
  SOURCE.json        commit, subject, sizes and SHA-256 of every module
  diagnostics.py  examples.py  loss_martnet.py  networks.py  runtask.py
  sampling.py     solver.py    train.py         utils.py
```

Scheduling scripts, calibration drivers and diagnostics are deliberately absent.
Which case uses which code path is recorded in `ENGINE_PINS.csv`.

Figures 4 and 5 come from the pre-October code paths that the released package
already pins under `paper_original/`; they are unchanged here.
