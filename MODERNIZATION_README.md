# CPI-3V processing — complete modernized snapshot


Key additions/updates:

- `python/proc/`: subprocess-oriented processing improvements, atomic MAT writes,
  safer argument handling, faster image-statistics hot path, timeseries fixes.
- `python/calibration/`: maintained PySide6 one-particle-at-a-time calibration GUI.
- `python/post/ChooseCalibrationParticles*.py`: compatibility launchers for the
  new GUI.
- `python/ml/cpi_ml/`: maintained modern ML package; the historical ML scripts
  remain present for reproducibility.
- Historical `~auxLoad` boolean bugs and the stray notebook magic in legacy ML
  scripts are corrected.
- `python/output/exportNetCDF.py`: modernized NetCDF exporter.
- `matlab/`: complete original MATLAB tree retained, with reviewed parity fixes
  overlaid in `matlab/proc/` and notes in `matlab/PARITY_NOTES.md`.

The MATLAB particle-statistics algorithm is intentionally not claimed to be
numerically equivalent to the Python implementation; see `PARITY_NOTES.md`.
