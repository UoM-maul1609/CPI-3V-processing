# CPI-3V processing

Tools for turning raw **CPI-3V** (and CPI 1.0) `.roi` files into particle
images, particle properties and **concentration time series**, exporting those
to **NetCDF**, and choosing **calibration particles** interactively.

The Python code under `python/` is the reference implementation. The
`matlab/` folder holds the original MATLAB version for comparison.

Developed by Dr Paul Connolly, University of Manchester. Licensed under the
GNU LGPL v3 (see `LICENSE`). Please cite the original source when you use the
code in scientific work.

---

## Contents

1. [Repository map](#1-repository-map)
2. [Setting up Python](#2-setting-up-python)
3. [Processing ROI files — `python/proc/`](#3-processing-roi-files--pythonproc)
4. [Exporting to NetCDF — `python/output/`](#4-exporting-to-netcdf--pythonoutput)
5. [Batch scripts — `bash/`](#5-batch-scripts--bash)
6. [Calibration workbench — `python/calibration/`](#6-calibration-workbench--pythoncalibration)
7. [Other folders](#7-other-folders)
8. [Tests](#8-tests)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Repository map

```
.
├── python/
│   ├── proc/          ★ main processing pipeline: .roi → .mat → images + timeseries.mat
│   ├── output/        ★ timeseries.mat → CF-style NetCDF
│   ├── calibration/     interactive calibration workbench (cpi-calibrate) — own README
│   ├── post/            older calibration readers + launchers for the workbench
│   └── ml/              experimental machine-learning classification (cpi_ml = modern path)
├── bash/              ★ batch drivers for many flights/runs, PNG → PDF helper
├── matlab/              original MATLAB pipeline (proc/) and calibrator (cals/)
├── tests/               pytest suite
├── images/              diagram used by the ML image pre-processing
├── pyproject.toml       package definition and optional extras
└── requirements-*.txt   alternative pip requirement files
```

A typical run looks like this:

```
raw .roi files ──► python/proc/batchProcess.py ──► <file>.mat per ROI file (ROI_N, HOUSE, IMAGE1, BG, dat)
                                               ├─► full_backgrounds.mat
                                               ├─► mm_dd_yy_pytgt<min_len>/  PNG pages of particle images
                                               └─► timeseries.mat  (conc, conc2, conc2ar …)
timeseries.mat ──► python/output/exportNetCDF.py ──► <flight>.nc
<file>.mat     ──► cpi-calibrate (python/calibration) ──► calibration flags / sets / positions CSV
```

---

## 2. Setting up Python

Use a dedicated virtual environment. NumPy/SciPy, OpenCV, Qt, NetCDF and
TensorFlow are heavy dependencies, and you don't want them changing other
projects.

**Python version:** 3.12.14 is recommended. **Do not use 3.12.0**: its
import machinery clashes with the PySide6/`six` stack used by the GUI. Python
3.10+ is required.

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Then install what you need:

| You want to … | Install |
|---|---|
| Process ROI files (`proc/`) | `python -m pip install -e .` |
| …and use the calibration workbench | `python -m pip install -e ".[gui]"` |
| …and export NetCDF (`output/`) | `python -m pip install -e ".[gui,output]"` |
| …and run the ML classifier | `python -m pip install -e ".[ml]"` |
| …and run the tests | add `dev`, e.g. `".[gui,output,dev]"` |

The `requirements-core/gui/output/ml.txt` files are equivalent if you prefer
`pip install -r`.

Check the right interpreter is active with
`python -c "import sys; print(sys.executable)"`; it should point inside
`.venv`. Leave the environment with `deactivate`. Next time you only need
`source .venv/bin/activate`. `.venv/` is not committed to Git.

---

## 3. Processing ROI files — `python/proc/`

### 3.1 Quick start

```bash
source .venv/bin/activate
python python/proc/batchProcess.py /path/to/flight/roi_directory --cores 4
```

- **Input:** a directory containing CPI `.roi` files.
- **Output:** written into that same directory.
- **Path:** the command can be run from anywhere. The driver finds its helper
  scripts relative to itself.
- **`--cores`:** the number of ROI files processed in parallel during the
  particle-property stage (default 4).

Redirect the log if you want to keep it, as the `bash/` scripts do:

```bash
python python/proc/batchProcess.py /data/C298/ &> /data/C298/python.out
```

### 3.2 Settings (top of `batchProcess.py`)

Edit these variables before a run. They are not command-line options.

| Variable | Default | Meaning |
|---|---|---|
| `cpiv1` | `False` | `True` for **CPI 1.0** data. This changes the block markers, housekeeping layout and image-type codes. |
| `foc_crit` | `5` | Minimum **focus** for a particle to be counted in the time series or exported as an image. |
| `min_len` | `1` | Minimum particle length (µm) for **image export** only. |
| `dt` | `10` | Time-series bin width (s). |
| `ds` | `10` | Size-bin width (µm). Bins run 0–2300 µm. |
| `vel` | `100` | Air speed through the sample volume (m s⁻¹), e.g. aircraft TAS. It sets the swept volume. |
| `find_particle_edges` | `True` | Store each particle's 30-point boundary (needed for the calibration workbench overlay). |
| `process_sweep1_if_exist` | `True` | Re-extract `.roi → .mat` even if the `.mat` already exists. Set `False` to reuse existing extraction. |
| `process_roi_driver`, `process_image_stats`, `export_images`, `output_timeseries` | all `True` | Switch individual stages on or off (section 3.3), e.g. to redo only the time series. |
| `num_cores` | `4` | Default for `--cores`. |
| `outputfile` | `timeseries.mat` | Time-series file name. |
| `classifier`, `classifierFile`, `minClassSize` | `False`, *path*, `50.0` | Use a trained CNN/DEC model to split particles into classes (drops / ice / unclassified). This needs the `ml` extra and a model saved as `<classifierFile>.json` + `.h5`. |

### 3.3 What each stage does

**Stage 1 — extract ROI data** (`ROIDataDriver.py`). This runs two sweeps,
each using one fresh process per file.

1. **Sweep 1** reads the binary `.roi` file and finds its blocks:
   - image blocks (`0xA3D5`);
   - ROI blocks (`0xB2E6`);
   - housekeeping blocks (`0x484B`, or `0xA1D7` for CPI 1.0).

   It then:
   - decodes the header, image, ROI and housekeeping structures
     (`convertDataTo*SA.py`);
   - builds per-ROI arrays (`postProcess.py`);
   - saves `<file>.mat`;
   - adds any full-frame **background** images (image type 19, or 33795 for
     CPI 1.0) to `full_backgrounds.mat`, which also stores the overall time
     range `t_range`.
2. **Sweep 2** gives every ROI the matching crop of the nearest-in-time
   background (`associateBackgrounds.py`) and saves it as `BG`.

**Stage 2 — particle properties** (`imageStatsDriver.py` → `imageStats.py`).
Files are processed in parallel, one subprocess per file. Only particle ROIs
are processed (image type 89, or 33857 for CPI 1.0). For each one the code:

1. subtracts the background;
2. median-filters the result;
3. Otsu-thresholds it;
4. finds the particle contour;
5. measures it, with a pixel size of **2.3 µm**:
   - length (major axis);
   - width (minor axis);
   - area;
   - roundness = area / (π/4 · L²);
   - orientation and centroid;
6. computes **focus**, an edge-gradient sharpness metric evaluated along 30
   boundary points (higher = sharper);
7. stores the 30-point boundary.

The results go into `dat` inside `<file>.mat`. ROIs that fail a check (too
faint, too small, no contour) keep zero sizes and NaN focus. For CPI 1.0, the
centre-strip artefact is repaired first.

**Stage 3 — image export** (`exportImagesDriver.py` → `exportImages.py`).
Particles with length > `min_len` and focus > `foc_crit` are tiled onto PNG
pages:

- **Folder:** `mm_dd_yy_pytgt<min_len>/`, e.g. `03_21_02_pytgt1.0/`.
- **File names:** each page is named after the time of its first particle,
  `mm_dd_yy_HH_MM_SS.mmm.png`.
- **Page breaks:** a new page starts when the page is full or the day changes.
- **With the classifier:** the folder is `mm_dd_yy_class/` and particles are
  coloured grey (drops), blue (ice) or red (unclassified).

Use `bash/make_pdf.sh` to merge a folder of pages into one PDF (section 5).

**Stage 4 — concentration time series** (`calcTimeseriesDriver.py`). Particle
ROIs with focus > `foc_crit` are counted into time × size × roundness bins.
Each count is divided by the sampled volume in its time bin:

```
concentration = counts / ( (dt − deadtime) · vel · SA  +  nimages · SV )
SA = 1280·1024/√2 · (2.3 µm)²        (sample area)
SV = SA · √2 · 3 mm                  (volume of one image frame)
```

`deadtime` comes from the housekeeping packets and `nimages` is the number
of image frames in the bin. With `classifier = True`,
`calcTimeseriesClassifierDriver.py` writes `timeseries_class.mat` instead.
Its third axis is the **class** rather than roundness (see below).

### 3.4 Output files

| File | Contents |
|---|---|
| `<file>.mat` (one per `.roi`) | `ROI_N`, `HOUSE`, `IMAGE1`, `BG`, `dat`. Details in the next table. |
| `full_backgrounds.mat` | `FULL_BG` (all background frames and their times) and `t_range` (first/last time, rounded to `dt`). |
| `mm_dd_yy_pytgt<min_len>/*.png` | Pages of particle images. |
| `timeseries.mat` | Struct `timeser`. Details in the next table. |
| `timeseries_class.mat` | Same layout, classifier runs only. |

**Inside `<file>.mat`:**

| Variable | Field | Meaning |
|---|---|---|
| `ROI_N` | `Time` | Time of each ROI (MATLAB datenum). |
| | `IMAGE(i).IM` | Pixels of each ROI (uint8). |
| | `StartX`, `StartY`, `EndX`, `EndY` | Position of each ROI on the 1280 × 1024 camera. |
| | `imageType` | Image-type code of each ROI. |
| `HOUSE` | `Time`, `deadtime` | Housekeeping times and dead time. |
| `IMAGE1` | `Time`, `Time1`, `Timestr`, `imageType` | Per-frame times: instrument clock, datenum and string. |
| `BG` | `BG(i).BG` | Background crop matching each ROI. |
| `dat` | `len`, `wid`, `area` | Length, width (µm) and area (µm²). |
| | `round`, `orientation`, `centroid` | Roundness, orientation and centroid. |
| | `foc.focus`, `foc.boundaries` | Focus and the 30-point boundary. |
| | `Time`, `imageType` | Copies of the ROI times and types. |

**Inside `timeseries.mat` (`timeser`):**

| Field | Shape | Meaning |
|---|---|---|
| `Time` | nt | Bin centres (MATLAB datenum), spacing `dt`. |
| `size1`, `size2`, `midsize` | nl | Lower edge, upper edge and middle of each size bin (µm). |
| `ar1`, `ar2` | na | Roundness bin edges, 0–1.2 in steps of 0.2. |
| `conc` | nt | Total number concentration (m⁻³). |
| `conc2` | nt × nl | Concentration by size (m⁻³ per bin). |
| `conc2ar` | nt × nl × na | Concentration by size and roundness, or by **class** in `timeseries_class.mat`. |
| `deadtimes`, `nimages` | nt | Dead time and number of frames in each bin. |

In `timeseries_class.mat`, index 0 of the third axis is *unclassified* and
index *k* is cluster *k − 1*.

### 3.5 Memory and robustness

Each heavy step runs in its own short-lived subprocess, so memory is returned
to the operating system after every file. Native thread pools are pinned to
one thread per worker. MAT files are written atomically: a killed job never
leaves a half-written file. If an old run that used `multiprocessing` crashed
and left shared-memory segments behind, `python/proc/kill_ipcs.sh` removes
the System V IPC objects (shared memory, semaphores, message queues) owned by
you. Run it only when none of your processing jobs are running.

---

## 4. Exporting to NetCDF — `python/output/`

`exportNetCDF.py` converts a `timeseries.mat` / `timeseries_class.mat` into a
CF-style NetCDF4 file. Install the `output` extra first.

```bash
python python/output/exportNetCDF.py /data/C298/timeseries.mat /data/C298/c298.nc \
    --title "CPI-3V size distributions, flight C298" --comment "foc_crit=5"
```

With classifier output, sum classes into drop and ice products:

```bash
python python/output/exportNetCDF.py /data/C298/timeseries_class.mat \
    /data/C298/uman_faam_20220719_v002_c298.nc --drop-bins 2,4,6,9 --ice-bins 1,3,5,7,8
```

**Options:**

| Option | Meaning |
|---|---|
| `--drop-bins`, `--ice-bins` | Comma-separated indices along the third axis of `conc2ar`. For classifier output these are classes (0 = unclassified). For a normal `timeseries.mat` they would be roundness bins. |
| `--title`, `--institution`, `--source`, `--history`, `--comment` | Global attributes of the NetCDF file. |

**NetCDF contents:**

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `time` | time | seconds since the first bin | Bin times. |
| `size1`, `size2` | size | µm | Lower and upper edges of each size bin. |
| `conc` | time | m⁻³ | Total concentration. |
| `conc2` | time × size | m⁻³ | Concentration by size. |
| `conc2Drops`, `concDrops` | time × size, time | m⁻³ | Only with `--drop-bins`. |
| `conc2Ice`, `concIce` | time × size, time | m⁻³ | Only with `--ice-bins`. |

From Python, `export_netcdf(...)` does the same job, and `exportNetCDF(...)`
is kept for old scripts. Historical MATLAB v7.3 (HDF5) files are read through
the optional `hdf5storage` package.

`batchOutputNetCDFDCMEX.py` is the script used for the DCMEX campaign
(flights C298–C314, with hard-coded paths and class bins). Copy and edit it
for a new campaign.

---

## 5. Batch scripts — `bash/`

| Script | What it does |
|---|---|
| `batch.sh` | Runs `batchProcess.py` over the C072–C098 3VCPI flight folders. |
| `batch_dcmex.sh` | The same for DCMEX flights C298–C315. |
| `batch_mphase.sh` | The same for M-PHASE flights C318–C333. Most lines are commented out; only C328 is active. |
| `make_pdf.sh` | Merges every PNG in a folder into one PDF. |

### Running many flights (`batch*.sh`)

Each line processes one flight and writes its log to `python.out` in that
flight's folder. The paths are hard-coded to the original machines, so
**copy one of these scripts as a template** and edit the paths.

The scripts call `batchProcess.py` by a relative name, so **run them from
`python/proc`** with the virtual environment active:

```bash
source .venv/bin/activate
cd python/proc
bash ../../bash/batch_dcmex.sh
```

The scripts call `python3`, which is the venv interpreter once the venv is
active. Comment a line out (`#`) to skip a flight. Flights run one after
another.

### PNG pages → one PDF (`make_pdf.sh`)

```bash
bash bash/make_pdf.sh /data/C298/07_19_22_pytgt1.0
# → /data/C298/07_19_22_pytgt1.0.pdf
```

- Pass the folder **without** a trailing slash.
- It needs ImageMagick (`convert`) and Ghostscript (`gs`).
- Pages are converted one by one in `/tmp`, then merged in time order (the
  file names sort chronologically).
- **Caution:** the merge step takes *every* `*.pdf` in `/tmp`. Make sure
  `/tmp` holds no unrelated PDFs, or they will end up in the output.

---

## 6. Calibration workbench — `python/calibration/`

An interactive tool for choosing calibration particles from processed
`<file>.mat` files. You:

- box-select particles on the camera X–Y plot;
- refine the selection on focus and length vs stage-position plots, with
  median traces;
- accept or reject particles, labelling them with true sizes (bead types,
  ice analogues) from a library shared across datasets;
- save calibration sets with a best-focus X/Z position;
- export the signed distance of each particle from the object plane.

```bash
python -m pip install -e ".[gui]"
cpi-calibrate /path/to/processed/calibration
```

The workbench never modifies the MAT files. **Full instructions are in
[`python/calibration/README.md`](python/calibration/README.md).**

---

## 7. Other folders

**`python/post/`** holds older calibration helpers:

- `ReadCPIFiles.py` loads times, focus, length, images and boundaries from a
  folder of processed MAT files;
- `ReadPositionFile.py` reads legacy stage-position logs;
- `ReadAndInterp.py` matches the two in time.

`ChooseCalibrationParticles*.py` now simply launch the calibration
workbench.

**`python/ml/`** contains experimental classification of particle images:

- **`cpi_ml/`** is the maintained path, using Keras 3 and HDF5 datasets that
  record each particle's source file and ROI. See `python/ml/cpi_ml/README.md`.
- **`cnn/`, `sae/`, `saeMNIST/`, `postProcessImages0*.py`** are older Keras 2
  scripts, kept to reproduce historical models. The classifier option in
  `batchProcess.py` uses a model from `cnn/`.
- **`images/postProcess.png` / `.svg`** show the padding geometry used when
  `postProcessImages*.py` rotates particle images to a common orientation.

**`matlab/`** is the original MATLAB pipeline:

- `proc/batchProcess.m` mirrors the Python stages;
- `cals/calibrator.m` is the old calibration GUI.

`matlab/PARITY_NOTES.md` lists what has been ported between the two. In
particular, the particle segmentation differs, so lengths and focus from
MATLAB and Python are **not** numerically identical. New work should use the
Python pipeline.

---

## 8. Tests

```bash
python -m pip install -e ".[gui,dev]"
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python -m pytest tests
```

The tests cover:

- ROI time and full-frame handling;
- stage-position matching;
- calibration sets and exports;
- the label library;
- the signed object-plane distance;
- a headless run of the workbench's select → accept → unflag → undo workflow.

---

## 9. Troubleshooting

- **"No .roi files found".** Point `batchProcess.py` at the directory that
  directly contains the `.roi` files. Subfolders are not searched.
- **"No background images were found".** Sweep 2 needs at least one
  full-frame background in the run. Check that the `.roi` files contain
  background frames, and that `cpiv1` matches the instrument.
- **To redo only the time series** (e.g. a new `foc_crit` or `dt`), set
  `process_roi_driver`, `process_image_stats` and `export_images` to `False`,
  then rerun.
- **Out of memory.** Lower `--cores`. Each worker holds one ROI file's images.
- **CPI 1.0 data gives nonsense.** Set `cpiv1 = True`: the block markers and
  image-type codes are different.
- **NetCDF export fails on an old file** with "v7.3/HDF5". Install
  `hdf5storage`.
- **Workbench issues.** See section 15 of `python/calibration/README.md`.

`python/README` and `matlab/README` are older plain-text notes. This file
supersedes `python/README`.
