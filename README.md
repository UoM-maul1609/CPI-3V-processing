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
| `find_particle_edges` | `True` | Store each particle's boundary (30 points with `legacy`; as many as the shape needs with `transmittance`). Needed for the calibration workbench overlay. |
| `edge_options` | `{"method": "legacy", ...}` | Edge detection, see section 3.6. `legacy` reproduces the original processing exactly; `transmittance` is the improved method. The options used are written into every output file as `dat.proc_options`. |
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

### 3.6 Edge detection: legacy and transmittance methods

The particle edge decides the measured length, so it matters for every size
and concentration. `proc/edgeDetect.py` has two methods. The one used is set
by `edge_options` in `batchProcess.py`, or by `--edge-method` and
`--edge-level` on the command line. It is recorded in each file as
`dat.proc_options` (JSON: method, levels, passes, pixel size, code version,
date).

**`legacy`** is the original method, kept unchanged so results can be
compared:

1. Background difference.
2. 100 passes of a 3×3 median filter.
3. Otsu threshold, then 11 more median passes on the mask.
4. Contour, then the mask is grown by `contains_points(radius=2)`, which adds
   about 2 px to the diameter.
5. Length = ellipse major axis.
6. 30 boundary points.

Its edge sits outside the particle, so small particles are oversized most,
and heavy filtering rounds off corners. Some acceptance tests use fixed grey
levels, which depend on laser power.

**`transmittance`** works as follows:

1. Divide the image by its background, then by the median of the ROI border,
   giving `T` = the fraction of local light transmitted. Laser power and
   bright or dark parts of the background drop out.
2. Apply one median pass (`denoise`).
3. The particle is the dark region holding the darkest pixels. The bright
   centres of transparent beads and drops are filled in.
4. Its edge is traced sub-pixel at `T = edge_level`.
5. Length = maximum Feret dimension (the "maximum dimension" of the
   calibration papers). Width = minimum Feret dimension. Area, perimeter,
   equivalent diameter and roundness come from the traced outline.
6. Focus is recorded both as the legacy metric (`foc.focus`) and on the
   transmittance (`foc.focus_T`), which does not scale with illumination.
7. The boundary keeps as many points as the shape needs (simplified to
   0.25 px), so corners and necks are not cut off. `n_boundary` gives the
   count.
8. `edge_touch = 1` flags particles cut off by the ROI border.
9. `T_min` and `T_centre` record shadow depth and the bright centre, for
   future use.

**Edge mode: fixed or relative.** The edge can sit at a fixed
transmittance (`edge_mode: "fixed"`, at `edge_level`), or at a fixed
fraction of each particle's own shadow depth (`edge_mode: "relative"`, at
`T = 1 − edge_fraction·(1 − T_min)`).

* **Fixed:** small particles slightly out of focus have shallow shadows. They
  first come out too small, then are lost once their darkest pixel is
  brighter than the edge level. In synthetic tests a 10 µm particle measured
  9.7 µm when sharp, 7.9 µm when blurred, then disappeared. This biases the
  small-particle depth of field.
* **Relative:** keeps them, and measures them within about 1 px of true for
  10 µm and larger particles through the same blur. Very faint small
  particles come out somewhat large, but consistently, so the calibration
  can absorb it.
* **On real 100 µm beads** both modes agree to about 1 µm. Use the 15 µm
  beads to choose between them. The level actually used for each particle is
  stored as `edge_level_used`.

**Edge level.** For an ideal coherent imager, the edge of an opaque object
in focus sits near `T = 0.25`. On 100 µm beads from 21 March 2002 (file
`03211315.mat`), the in-focus median length was:

| Method | Median length |
|---|---|
| legacy | 107.5 µm |
| transmittance, `T = 0.25` | 92.0 µm |
| transmittance, `T = 0.35` | 96.3 µm |
| transmittance, `T = 0.5` | 101.3 µm (equivalent diameter 100.0 µm) |

The certified size is 99.4 µm. The level should be chosen with beads of
several sizes: the right level makes all sizes come out true at focus, with
no fixed offset.

**Tested on the 21/3/02 beads (project 210302_2).** The test used 13,747
of the 59,005 flagged particles: all 5,747 at 15 µm, and 4,000 each at 40
and 100 µm. Each method was scored by how well L_app/L for all three sizes
collapses onto one curve of the Fresnel number Zd = λ|d|/(L/2)². That is
what diffraction theory predicts, and what the correction relies on. Lower
is better in the first two columns.

| Method | Scatter about the common curve (log) | Spread between sizes | Detected 15 µm | Offset at focus 15/40/100 µm |
|---|---|---|---|---|
| legacy (subtract, Otsu, +2 px) | 0.048 | 0.117 | (all, by selection) | +10.1 / +9.9 / +4.0 |
| plain ratio, edge at 50% of shadow | 0.040 | 0.044 | 100% | +0.1 / +2.3 / −3.0 |
| ratio ÷ border median, fixed T = 0.5 | 0.031 | 0.041 | **72%** | +0.7 / +3.2 / −1.3 |
| **ratio ÷ border median, edge at 65% of shadow** | **0.032** | **0.040** | **100%** | **+0.6 / +1.8 / −2.9** |
| ratio ÷ border plane, edge at 65% of shadow | 0.031 | 0.039 | 100% | +0.6 / +1.8 / −3.1 |
| gain + offset fit on the border, fixed T = 0.5 | 0.047 | 0.059 | 85% | +3.3 / +5.3 / +0.3 |

* **Divide, don't subtract.** Then **correct for the laser intensity
  difference** between the background frame and the particle frame, using
  the ROI border (the median border ratio). This cut the scatter by about a
  quarter compared with the plain ratio or the subtraction, so the
  difference is real in this data.
* A tilted-plane fit to the border gains nothing measurable. A gain-and-offset
  fit is unstable, because the background is nearly flat across a small ROI.
* **Put the edge at a fraction of each particle's own shadow depth.** The
  optimum is broad, between 0.55 and 0.7 (0.55 is the default from v14.6, see below). A fixed
  transmittance level loses about 30% of the 15 µm beads as they go out of
  focus.
* With this, the 15, 40 and 100 µm beads lie on one curve (spread 0.040
  against 0.117 for legacy), and the curve is much flatter with defocus.

Recommended command:

```bash
python batchProcess.py <copy of processed data> --only-stats --cpiv1 \
       --edge-method transmittance --edge-mode relative --edge-fraction 0.55
```

**Edge fraction 0.55 (v14.6).** With the edge at 65% of the shadow depth,
100 µm beads come out slightly smaller on the negative side of the object
plane than at focus (about 0.94 of true at d ≈ −0.85 mm, against 0.96 at focus
and 0.99 at +0.85 mm), because the blurred edge of a defocused bead is
lop-sided on that side. Putting the edge a little further out, at 55%, makes
the measured size grow away from the object plane on both sides for all
three sizes, and is as consistent (equivalent-area diameter):

| Edge fraction | Scatter | Spread between sizes | 100 µm at −0.85 / 0 / +0.85 mm | 40 µm | 15 µm (±0.25 mm) |
|---|---|---|---|---|---|
| 0.65 | 0.027 | 0.035 | 0.94 / 0.96 / 0.99 | 1.14 / 1.01 / 1.17 | 1.21 / 0.96 / 1.29 |
| **0.55** | **0.026** | **0.038** | **1.01 / 0.98 / 1.03** | **1.22 / 1.06 / 1.28** | **1.34 / 1.07 / 1.42** |
| 0.45 | 0.028 | 0.046 | 1.06 / 1.00 / 1.08 | 1.32 / 1.10 / 1.37 | 1.46 / 1.19 / 1.54 |
| 0.35 | 0.031 | 0.059 | 1.10 / 1.01 / 1.12 | 1.40 / 1.16 / 1.44 | 1.59 / 1.32 / 1.66 |

Lower fractions make the growth with defocus clearer, but the sizes become
less consistent. 0.55 is the default from v14.6.

**Which particle in the ROI.** The CPI cuts each ROI around the particle
that triggered it, so a ROI can also contain parts of neighbours, usually cut
off at its border. The transmittance method scores each shadow by its
integrated darkness, reduced with distance from the ROI centre and strongly if
it touches the ROI border, and measures the best one. (Before v14 it took the
darkest shadow, which was sometimes a large neighbour in a corner.) A bright
spot at the centre of a bead can open a notch to the outside; notches up to
`close_px` (4 px) are bridged, but only over pixels that are still in the
particle's shadow, so real concavities of crystals are kept. Length (maximum
Feret diameter) is not affected by the notch; area and roundness are.

**Outline and size measure (v14.2).** By default (`outline = "radial"`) the
outline of a compact particle is found along rays from its centre: on each
ray, walking out from the darkest point, the edge is where T first rises above
the edge level, and the result is smoothed over about 5 px of arc. Dark rings
and a bright centre spot no longer make it ragged. Particles that are not
star-shaped from their centre, or not roughly round (raw roundness below 0.6: columns, needles, rosettes), keep the raw contour
(`dat.outline_radial` says which was used). On the 21/3/02 beads, the
equivalent-area diameter (`dat.deq`) is the most consistent size: scatter
about the common curve 0.027 against 0.030 for the maximum Feret length, with
the same spread between sizes. The workbench can calibrate on either (Fit size
& focus model ▸ Size measure; `cpi-fit --size deq`). An outline drawn at the
steepest rise of T, as you would draw by hand, looks better but sizes less
consistently (scatter 0.031–0.036), because it moves with the fringes as the
particle defocuses.

**Outline as drawn by hand (`--outline edge`, v14.5).** Round particles are
outlined where T rises most steeply between the shadow and the bright
surround, searched around the outer edge of the shadow and kept continuous
from ray to ray. This is the edge a person draws. On the 21/3/02 beads it sizes
less consistently than the default threshold outline: scatter 0.035 against
0.027 (equivalent-area diameter), and spread between sizes 0.057 against
0.035. The visible disc of a defocused bead is the blurred image, which grows
with defocus and with the fringes, so it is not a better estimate of the true
size; for example, it gives about 60 µm for the defocused 40 µm beads where
the default gives about 50 µm. Use it to compare runs (Compare processing
runs) or for display.

**Particle in its own background (v14.12).** Each ROI is divided by the
crop of the full-frame background nearest in time. With particles fixed on a
slide, a background taken during a run contains the particles, so a particle
cancels out of image / background and only a sliver is outlined. On the
21/3/02 beads this affects 0.1% of particles, but on the 7/11/03 ice analogues
(file 11071612) it affects 83%. The transmittance method now checks whether
the particle's dark core is also dark in its background crop. If it is, it
uses the nearest-in-time full-frame background (from `full_backgrounds.mat`)
in which that part of the frame is clear, or failing that a smooth background
fitted to the ROI's own clear pixels. `dat.bg_source` records which was used
(0 assigned, 1 another frame, 2 fitted) and `dat.bg_dt_s` the time to the
frame used. Turn it off with `"bg_check": false` in `edge_options`.

**Transparent columns (v14.19).** A transparent column focuses light into a
bright streak along its axis. At the edge level this splits its shadow into
two parallel dark bars, and only one bar used to be outlined, about half the
true width. Now all edge-level parts inside one particle's shadow are kept.
The gaps between them are bridged, and the footprint itself is closed by the
same radius, so the space between bars that are joined by grey end caps counts
as particle. Separate particles have background between them and are never
joined. For such merged particles only, small concavities that the shadow
nearly encloses are also filled: at least 65% enclosed and at most 8% of the
area, e.g. a bright spot at a column's end. Gaps between the arms of a rosette
are larger and stay out.

On the 6/11/03 col166 file, 22% of particles change: width +11 µm and deq
+15 µm (medians), with columns now about 52 µm wide instead of about 35. On
the 21/3/02 beads 0.1% change, all junk ROIs at the frame edge. Options:
`merge_parts`, `merge_max_gap`, `fill_bays`, `fill_bays_max_area`.

v14.20 handles columns far from focus. Some image as two thin parallel walls
with a wide bright gap, and end caps too faint to count as shadow, so the two
walls were separate particles. A second shadow piece is now joined to the
particle when both conditions hold:

- they connect at a fainter level (T < 1 − 0.5 × min_contrast, i.e. 0.925);
- both pieces are elongated (≥ 3:1), parallel (≤ 10°), of similar length and
  side by side.

Separate needles have background between them and don't connect. Effect:
ROI 1346 (11061538) 13 → 40 µm wide. 0.2% of that file changes, and none of
the 21/3/02 beads. Options: `join_faint`, `join_min_aspect`, `join_max_angle`,
`join_min_size`.

v14.21 fills shadow-covered bays. Out of focus, parts of a column's shadow can
be lighter than the edge level: an inner channel, or a big notch at a bright
end. Those bays used to be cut out of the outline. Concavities of the outline
that lie at least 70% inside the particle's own weak shadow (T < 0.85) are now
filled. This applies to elongated particles (≥ 2:1) and split-and-merged ones.
The gaps between a rosette's arms show background and are kept.

On 11061340, 39% of outlines change, with deq a median of +2 µm. None of the
1,845 accepted 21/3/02 beads change. Options: `fill_bays_shadow_cover`,
`fill_bays_min_aspect`.

**Cut-off by the ROI edge (v14.23).** Every particle now records
`dat.cutoff_pct`: the percentage of its outline within 1 pixel of the ROI edge
(0 = wholly inside), the same definition as CPIview's cut-off. A cut-off
particle's size is a lower limit.

- **Processing:** `--max-cutoff X` leaves particles more than X% cut off out of
  the concentration time series (CPIview's spectra use 6). Their statistics
  stay in the files, and the number left out is printed and stored as
  `timeser.max_cutoff_pct`.
- **Workbench:**
  - Filters has a "Cut off by ROI edge [%]" min/max row;
  - "Cut off by the ROI edge" is a plot metric;
  - the fit dialog has "Max cut-off by ROI edge" (`cpi-fit --max-cutoff X`).

  Particles processed before v14.23 have no value and are always kept.

On the 7/11/03 ice-analogue file, 824 of 5,009 particles are more than 6% cut
off (slide-mounted analogues often cross the ROI edge). With `--max-cutoff 6`,
740 particles that passed the focus test were left out and the summed
concentration fell by 17%.

**CPIview focus and edge quality (`--cpiview`, v14.22).** This option adds,
per particle, the focus measure and edge-quality variables of SPEC's CPIview 2,
ported from its IDL code (`proc/cpiviewMetrics.py`). Sizes still come from this
processing; CPIview's size correction and depth of field are not used. Stored
as `dat.cpv_*`:

- `cpv_focus`: the mean over 16 points round the boundary of
  100 × peak / FWHM / 110 of the gradient across the edge. The gradient is
  |Sobel| on the background-subtracted ROI after 12 median-filter passes. Also
  stored: `cpv_focus_sd`, `cpv_focus_min`, `cpv_focus_max`,
  `cpv_focus_median` and `cpv_samples`. A large spread means part of the edge
  is sharp and part blurred (tilted or partly cut-off particles).
- `cpv_fwhm_px`: the median edge width (FWHM of the gradient profile), and
  `cpv_peak`, the median peak gradient.
- `cpv_distinct`: the fraction of boundary pixels with a gradient above 40.
  CPIview keeps an object only above 0.4. A quick automatic junk filter.
- `cpv_cutoff_pct`: the percentage of boundary pixels on the ROI edge. CPIview
  spectra use particles under 6%.

This costs about 0.1 s per particle. In the workbench these appear as plot
metrics. The size-and-focus fit can use the CPIview focus instead of the
processed one ("Focus" in the fit dialog; `cpi-fit --focus cpv_focus`), and the
saved model records which focus it used. On 3,547 beads from 21/3/02 (15/40/100
µm, |d| ≤ 0.65 mm), the processed focus gave 6.7% sizing error (side of the
object plane unknown) and the CPIview focus 7.3%. The CPIview focus identified
the side better: 63% against 54%.

**Pixel size (`--pixel-um`, v14.18).** Lengths and areas are converted from
pixels at 2.3 µm unless `--pixel-um` is given. Edges, outlines and focus are
computed in pixels, so changing it scales every size by exactly the ratio and
changes nothing else. The value is recorded in each file (`dat.proc_options`).
The workbench fit reads it, so its "Pixel size: set" option rescales relative
to what was used in processing and never applies it twice. The concentration
time series reads it too, together with the real frame size from
`full_backgrounds.mat`: 1024 × 1022 px for CPI 1.0. The time series previously
assumed 1280 × 1024 px at 2.3 µm for every instrument, which made CPI 1.0
sample areas 25% too large before any pixel-size change.

**Choosing the backgrounds (`--backgrounds`, v14.13).** By default every
full-frame background is used (`--backgrounds all`). `--backgrounds choose`
opens a window after the backgrounds have been collected, before they are
assigned to the ROIs. It shows every background in time order with a tick box
and a count of *spots* (dark patches compared with the median of all
backgrounds, usually particles sitting in that background). Click one to see
it large, optionally as its difference from the median. The choice is saved in
the data folder as `background_selection.json`, keyed by background time, so it
still applies after reprocessing. The next time, the window first offers to
use the previous selection. `--backgrounds saved` reuses the saved choice
without the window. Only ticked backgrounds are used: for the nearest-in-time
assignment, and as the replacements in the own-background check above.
`full_backgrounds.mat` is not changed.

```bash
python batchProcess.py <folder> --cpiv1 --backgrounds choose ...         # full processing
python batchProcess.py <folder> --only-stats --cpiv1 --backgrounds choose ...   # already-converted MAT files
```

With `--only-stats`, giving `--backgrounds` (any value) re-assigns the
backgrounds to the ROIs before the statistics. Without it, the backgrounds
already stored in the MAT files are kept.

**ROI times (fixed in v14).** ROIs take the time of the full-frame IMAGE
block before them in the file. Blocks are found at both even and odd byte
offsets, and the old code searched that list without sorting it, so about
half the ROIs got the time of the wrong frame: typically 0.1–0.2 s out,
sometimes tens of seconds, and also the wrong image type. MAT files made
before v14 have these times. Reprocess the `.roi` files to correct them, then
use **Copy project onto reprocessed data** in the workbench, which matches
flagged ROIs by their images rather than their times.

**Comparing methods on the same particles:**

```bash
cp -r processed/Cal210302 processed/Cal210302_T050           # a copy; originals untouched
python batchProcess.py processed/Cal210302_T050 --only-stats --cpiv1 \
       --edge-method transmittance --edge-level 0.5
# or, edge at half of each particle's own shadow depth:
python batchProcess.py processed/Cal210302_R050 --only-stats --cpiv1 \
       --edge-method transmittance --edge-mode relative --edge-fraction 0.5
```

Then, in the workbench:

1. Open your project and use **File ▸ Copy project onto reprocessed data…**
   with the new folder. Flags, labels, batches and Xbest/Zbest carry over,
   because particles are matched by time and position.
2. **Calibration ▸ Compare processing runs…** with both projects (or run
   `cpi-compare a.json b.json`) tabulates, side by side:
   * length at focus per bead size;
   * the fitted edge offset, scale and L power;
   * the depth of field;
   * the leave-one-batch-out sizing errors.

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
- give each accepted cluster its own best-focus X/Z (handles tilted slides);
- export the signed distance of each particle from the object plane;
- record which CPI was calibrated; keep one project per calibration run and
  fit several runs together, read live from their projects (`cpi-fit
  run1.json run2.json …`), or export them to one CSV (`cpi-combine`);
- fit the size and focus calibration on each side of the object plane, check
  it out-of-sample, compute the depth of field, and retrieve true sizes for
  cloud data with the side of the object plane marginalised
  (Calibration menu, `cpi-fit`, `cpi-retrieve`).

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
- per-batch best focus and exports;
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
- **Workbench issues.** See section 16 of `python/calibration/README.md`.

`python/README` and `matlab/README` are older plain-text notes. This file
supersedes `python/README`.

### Calibration edge-method laboratory (read-only)

From **Review → Flag history**, Ctrl/Shift-click one or more saved batches and
**Test edges →** to open **④ Edge lab (temporary)**. Add multiple independently
coloured algorithm trials (relative/fixed transmittance, radial, steepest gradient
or raw contour) to the same size/focus plots. Select, hide and remove individual
trial curves; hover or right-click a point to inspect that trial's image and
boundary. Copy production processing flags from the settings or a selected trial.
CPIview 2 focus can be computed separately and compared with ordinary edge and
normalised transmittance gradients. The Gallery tab can overlay both saved and
experimental boundaries and show size, stage position and distance to the object
plane. Trial measurements are only held in memory and **never committed**; see
`python/calibration/README.md` for details and production CLI caveats.

Edge Lab update: adjustable fixed-transmittance contour, faint-particle detection independent of sizing threshold, and per-particle transmittance-profile diagnostics. See `python/calibration/README.md`.


### Calibration batch → NetCDF (raw ROIs and label data)

The calibration workbench now has **Review → Flag history → Export selected
batch(es) to NetCDF…**. It writes the original ROI and matching per-ROI
background for every selected particle, together with stage X/Z, signed
distance from the saved object plane, true label length/width, processed length,
and batch memberships. ROIs are losslessly packed at their original variable
sizes, with no boundary processing. The exporter accepts both structured MATLAB
arrays and the cell-array-of-structs format produced by CPI processing; original
MAT files are read-only and need no conversion. See `python/calibration/README.md` for the
NetCDF variables and a reconstruction example. Install `.[gui,output]` for
compressed NetCDF4 export; small exports work with the core SciPy install.

**Calibration workbench Edge Lab (October 2026):** Stored batches can now be
added as selectable, instant reference curves with their saved outlines (no
edge recalculation). New experimental **Signed shadow integral · fitted circle**
trials measure the net light deficit through an enlarged fitted circle and plot
the resulting *uncalibrated shadow-equivalent diameter* against signed position
or as a ratio to known bead diameter. Curve colours match the saved boundaries,
trial edges or integration apertures displayed on particle inspection. The Edge Lab trial remains temporary; **production processing** now offers
`--shadow-circle-padding 8` to store the circle measurement as an additional
size without changing ordinary morphology/focus. See
`python/calibration/README.md` for instructions and limitations.


### Fitting Bench: shadow size + normalised, size-dependent focus (2026-10-22)

The **Fit size & focus model** dialog has a fitting-recipe picker; **Original size and focus** is the initial selection:

- **Circle shadow size + normalised focus (experimental)**: signed-shadow circle diameter
  `dat.shadow_deq` (µm) fitted to labelled bead diameter by an affine
  calibration, independently of position. Original Feret length, width,
  roundness and contour are preserved as separate measurements.
- **Contour size + normalised focus**: conventional edge size calibration,
  but fit the **raw CPI focus divided by local illumination**.
- **Original size and focus**: unchanged existing model.

The focus model for the first two presets is
`Fraw/Ilocal = B + A0*(L/Lref)^a * exp[-(|d-d0|/(w_±0*(L/Lref)^k))^p]`.
Fit `p` (or set `p=2`), fit separate positive/negative widths, and enable or
fix diameter dependence of peak and width. The two size exponents are
constrained nonnegative when fitted. Change the focus/illumination threshold
(default 0.12) to see the predicted **depth of field versus true size**, the
acceptance-weighted depth of field, and size-error histograms from held-out
batches. Threshold 0.12 is a UI starting value, **not a physical instrument
cutoff**. The validation checkbox controls whether the slower held-out testing
is run. Signed-side ambiguity is retained in retrieval and evaluation.

**Use without reprocessing:** With a single currently open project, choose
"Compute circle sizes for this project (temporary)" in the fitting dialog,
choose the circle expansion in pixels, and run the fit. The image/background
processing happens in a worker and is never committed. For a durable extra
column in processed MAT files, run on a **copy** of the MAT data:

```
python python/proc/batchProcess.py /path/to/COPY --only-stats --shadow-circle-padding 8
```

Then reload and save the calibration project to export `shadow_deq_um` and
`shadow_radius_px` in its particle table. Fixed-radius circles that extend
outside the ROI are now integrated using the visible pixels only (the unseen
part is assigned transmittance T=1, hence zero signed deficit). The fraction of
the aperture seen is recorded in `shadow_coverage`; invalid/nonpositive integrals
remain NaN. A cropped circle may underestimate size if missing pixels would
have contained shadow or diffraction, so inspect low-coverage measurements. New processing also stores
`dat.local_illum_bg`, a local background-beam estimate corrected for
multiplicative laser brightness differences where the ROI border provides
an adequate reference; older processing uses the historical image-border
illumination. This correction cannot resolve a strongly contaminated border,
changing beam profile or uncorrected camera pedestal.

The same fitting options are exposed via `cpi-fit ... --fit-recipe circle`
(or `contour` / `original`), `--foc-crit 0.12`, `--fixed-gaussian`,
`--symmetric-focus`, `--no-peak-size` and `--no-width-size`.

**This is a calibrated shadow-equivalent size proxy, not a guarantee of the
geometric diameter for all ice crystal types.** The model's held-out diagnostics
should be checked separately for each material and particle habit.


## Production defaults: contour, shadow size and focus (23 October 2026)

`python/proc/batchProcess.py` now defaults to all of:

- **Contour / morphology:** transmittance edge, relative shadow fraction `0.55`,
  radial outline and one median-filter pass. This preserves conventional length,
  area, roundness, boundary and original gradient focus (`dat.foc.focus`).
- **Independent shadow size:** `dat.shadow_deq` from the signed transmittance
  deficit in a fitted circular aperture at **+8 px**. It does not replace the
  contour or focus. This uses a fixed margin, *not* automatic radius-to-plateau.
- **Illumination:** `dat.local_illum_bg` retains the stored background's spatial
  beam profile, multiplied by a median image/background ratio from ROI-border
  pixels. The fitting recipe uses original focus **divided by** local illumination
  (equivalent to multiplying by its reciprocal).

All three are calculated when processing new data or recomputing statistics.
Use `--shadow-circle-padding N` to override the circle margin,
`--no-shadow-circle` to disable the extra size measurement, or
`--edge-method legacy` to reproduce the older edge method. With `--only-stats`,
processing **writes to the MAT files**; use a copy. The fitting dialog **starts with the original size and focus fit** for all projects.
The new circle-size / normalised-focus recipe is an **explicit experimental option**;
for old projects without `shadow_deq`, select it and use the temporary
"Compute circle sizes" button. Processing may still calculate `shadow_deq`
without selecting it for fitting.

**Proposed, not yet default:** obtain the illumination scale from background
pixels outside masks for *all* detectable particles and diffraction rings,
instead of assuming the ROI border is clear. Estimating ring extent from a
plateau in the signed integral is still research work; masks can consume the
whole small ROI and leave no reliable illumination reference. Keep the current
method until the masked estimator has been tested against the calibration data.

### Circle-shadow sizing: Edge Lab / Fit workbench consistency (23 Oct 2026)

When **Circle shadow size + normalised focus (experimental)** is selected, the
upper-left fit diagnostic plots the **uncorrected signed-shadow equivalent
diameter divided by the labelled true diameter**, precisely the observable
shown as *Diameter / labelled true size (shadow = integral)* in Edge Lab.
The size estimator is a signed integral of `1 - I / B_corrected` within a fitted
circular aperture. The true label is used only for plotting and fitting the
calibration, never as an input to the particle measurement.

The lower-right panel displays **paired histograms** for the original contour
equivalent-area diameter and the signed-shadow diameter, each divided by the
same particle's labelled size. Only images with both measurements contribute.
If contour equivalent-area diameter is unavailable, maximum Feret length is
used and the legend states so. This panel does not depend on enabling the
expensive held-out validation. In quick-fit mode the lower-left histogram is
explicitly labelled as a raw fit-sample diagnostic, not held-out validation.

The workbench fits **saved** `dat.shadow_deq` when present; selecting an 8 px
spinbox value does not retroactively change saved results. To reproduce an
Edge Lab trial exactly, select the same image set and press **Recompute circle
sizes as in Edge Lab (temporary)**, matching aperture margin and processing
settings. This uses the **saved boundary** in each MAT ROI (as Edge Lab does),
which can differ from the freshly recalculated boundary used during batch
processing if contour algorithms/settings changed. No MAT files are changed.

### 2026-10-23 — experimental adaptive shadow-circle sizing and consistent backgrounds

- **No change to production sizing defaults**: relative-transmittance radial contour at
  fraction 0.55, existing gradient focus, and *fixed* signed-shadow integration
  with an 8-pixel margin still run as before.
- Edge Lab now offers **Circle radius → Automatic plateau (experimental)**. The
  extra-radius control becomes a **maximum search**. Compare with the fixed +8 px
  trial before accepting results. The same experimental method can be enabled in
  processing with `--shadow-circle-auto --shadow-circle-max-padding 30`.
- Automatic radius selection requires a sustained plateau across two consecutive
  four-pixel intervals (rather than a single zero crossing), using a relative
  1.5% stability criterion. If no trustworthy plateau exists, it reports no
  measurement. The result is **not** a true particle boundary.
- Both integration modes now assign **T=1 to unseen pixels outside the ROI**.
  Such pixels contribute zero signed deficit; no annular-mean extrapolation is
  applied. Fixed-radius circles are not rejected just because they cross the ROI.
  `shadow_coverage` reports the fraction of the integration circle observed, and
  `shadow_extrapolated` remains false. If there was shadow in the unseen area,
  the reported diameter may be underestimated.
- **Automatic plateau selection remains conservative**: to avoid a false plateau
  caused by reaching the image boundary, it requires at least 85% coverage of
  each annulus used to establish convergence; otherwise it returns no reliable
  plateau. The fixed-radius result can still be used with its coverage warning.
- **Important: current historical bead ROIs may be too closely cropped to show
  a signed-shadow plateau.** A trial on the uploaded 15, 40 and 100 µm bead
  datasets found that the ROI limits were encountered before a robust plateau
  could be demonstrated. Keep fixed +8 px for current production analysis;
  the automatic method is for controlled comparison and larger ROIs.
- When the processor substitutes a clean full-frame background or a fitted
  background, it now saves that actual replacement as a *sparse* `BG_USED`
  MATLAB cell array (only changed ROI entries). The original `BG` remains
  intact. Edge Lab uses `BG_USED` where available, and applies the CPI 1.0
  centre-strip correction consistently. This takes effect **after reprocessing
  with this version**; older MAT files cannot recover replacements that were
  not stored. Existing records without `BG_USED` use `BG`.
- In Edge Lab right-click a shadow-trial particle and select **Cumulative shadow
  integral vs circle radius…** for a per-particle convergence and ROI-coverage
  diagnostic. No preview changes source MAT files.

Example to trial adaptive radius on a **copy** of processed CPI 1.0 MAT data:

```bash
python python/proc/batchProcess.py "/path/to/calibration_COPY" --only-stats --cpiv1 \
  --shadow-circle-auto --shadow-circle-max-padding 30 --pixel-um 2.46
```

The automatic method may legitimately return NaN for many historical ROIs.
For established processing, omit the two `--shadow-circle-*auto/max*` flags.

#### Microscope and SEM reference images (new)

The **Labels** tab now provides **Microscope / SEM reference images…** for any
bead or ice-analogue label. Attach multiple images per label, automatically
suggest boundaries, choose the intended object or trace it with an editable
circle or straight-line polygon, draw a reference scale and enter its length,
then fit a minimum enclosing circle. Measurements include maximum dimension,
object area, enclosing-circle area and their area ratio. The full-resolution
**Save annotated image as PNG…** export includes the label and the overlays.

Microscopy images and editable annotations are stored beside the shared labels
library in `microscopy/images/` and `microscopy/index.json` and do not modify
MAT files, CPI contours or the labels' nominal bead sizes.


### Shared CPI reference library and stable Edge Lab focus

The calibration GUI now recognises `CPI_cals/ReferenceLibrary/` from either
`CPI_cals/CPICalibration/...` or `CPI_cals/ICEAnalogues/CPI/...`. Labels live
in `ReferenceLibrary/labels.json`; microscopy images and editable annotations
live in its `microscopy/` folder. The Labels page allows automatic discovery or
a persistent manually selected folder, with consent before copying a legacy
library. Bright, black-outlined annotation contours remain legible on light
and dark particle images. Edge Lab shadow sizing changes the size measurement
only: both conventional focus and transmittance-gradient focus always reflect
the saved processed ROI values rather than the selected Edge Lab size trial.

### Area ratio and reference-corrected Edge Lab shadow sizing (experimental)

Processing now stores `dat.area_ratio_circle` = projected CPI contour area / area of its minimum enclosing circle. This is a per-ROI **CPI morphology ratio**, not the independent optical/SEM area ratio.

In Edge Lab, Size → **Shadow diameter / true size · area-ratio corrected** uses two explicit reference inputs: **Microscopy area ratio** `R_A` (default 0.30, change for your label) and **Reference Dshadow/Ltrue** `R_ref` (default 0.40, change for your particle type). The displayed curve is each particle's signed-shadow diameter / labelled true maximum size, multiplied by `sqrt(R_A)/R_ref`. For the example 0.30 and 0.40 the factor is 1.3693. These are user-configured example references, not automatically assigned microscopy measurements. They are applied across the selected curves, so mixed types should be compared separately. The original raw trial and focus values remain available; no MAT labels or ROI images are modified in Edge Lab. The reported correction is empirical and should not be called extinction efficiency.

### Shadow / shape validation (experimental)

The fitting dialog has a **Shadow / shape check** tab for the circle-shadow
recipe. It plots `(D_shadow / labelled_length)^2` against the CPI contour
area divided by the Feret-circle area, and shows their ratio as a histogram.
These are matched-particle *diagnostics*, not corrections. No SEM/microscopy
information is required for actual CPI images; microscopy reference
measurements must never be treated as if available on unknown particles.
The existing circle sizing fit, default established fit, and production
processing are unchanged by this diagnostic. The older Edge Lab manually
specified microscopy correction remains experimental and is not used in fits.
