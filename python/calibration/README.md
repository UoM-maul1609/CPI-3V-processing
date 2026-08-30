# CPI calibration GUI

The calibration GUI is a PySide6 replacement for the unfinished MATLAB GUIDE/Tkinter prototypes.
It reviews **one particle at a time** and never edits the processed `.mat` files.

## Workflow

1. Open a directory containing processed CPI `.mat` files.
2. Optionally attach the calibration-stage position text file (`x y date time`).
3. Narrow the review set by length, width, focus, image x/y, time, or current decision.
4. Set the particle's distance from the object plane with the slider/spin box.
5. Mark the particle **Keep**, **Reject**, or **Undecided**.
6. Decisions auto-save to `calibration_selection.csv` in the data directory.

The selection file records the source MAT file and ROI index, all review metrics, stage position,
manual object-plane position, and decision. This gives a reproducible audit trail without modifying
raw/processed data.

## Shortcuts

- Left / Right: previous / next particle
- K: keep
- R: reject
- U: undecided
- + / -: move object-plane position by one slider step

## Memory behaviour

Catalog scanning is done with a fresh subprocess for every MAT file. The GUI then caches only the
currently displayed source MAT file. This follows the same process-lifetime strategy as the main CPI
processor and prevents repeated SciPy MAT-file loads from growing the long-lived GUI process RSS.
