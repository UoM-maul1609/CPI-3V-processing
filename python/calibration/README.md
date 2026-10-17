# CPI Calibration Workbench

An interactive tool for choosing calibration particles (beads, ice-crystal
analogues …) from processed CPI-3V / CPI 1.0 `.mat` files. You select
particles on a camera X–Y plot, refine the selection on focus and size plots,
then accept or reject them. Accepted particles can carry a label with their
true size, and you can save named calibration sets with a best-focus stage
position.

**The original MAT files are never modified.** Everything you decide is
written to small CSV/JSON files alongside them.

---

## Contents

1. [Install and launch](#1-install-and-launch)
2. [Loading data](#2-loading-data)
3. [The window at a glance](#3-the-window-at-a-glance)
4. [What the colours mean](#4-what-the-colours-mean)
5. [The basic workflow](#5-the-basic-workflow)
6. [Selecting with the mouse](#6-selecting-with-the-mouse)
7. [Accept, reject, unflag and undo](#7-accept-reject-unflag-and-undo)
8. [Checking images: Gallery and Particle tabs](#8-checking-images-gallery-and-particle-tabs)
9. [Median traces](#9-median-traces)
10. [Labels with true sizes](#10-labels-with-true-sizes)
11. [Best focus, calibration sets and which side of the object plane](#11-best-focus-calibration-sets-and-which-side-of-the-object-plane)
12. [Filters](#12-filters)
13. [Files the workbench writes](#13-files-the-workbench-writes)
14. [Keyboard and mouse reference](#14-keyboard-and-mouse-reference)
15. [Tips and troubleshooting](#15-tips-and-troubleshooting)
16. [Code layout](#16-code-layout)

---

## 1. Install and launch

From the repository root, in your virtual environment (Python 3.12.14 is
recommended; do **not** use 3.12.0):

```bash
python -m pip install -e ".[gui]"
```

Then launch with any of:

```bash
cpi-calibrate
cpi-calibrate /path/to/processed/calibration          # load a folder straight away
cpi-calibrate /path/to/folder --position-file stage.csv
python -m calibration
```

---

## 2. Loading data

| To … | Do this |
|---|---|
| Load MAT files | Drag files or folders onto the window, or use **+ Add MAT files** / **+ Add folder** (sidebar). Folders are searched recursively. |
| Add another dataset | Use **+ Add folder** again (or **File → Add another dataset…**). Particles from every file are combined. |
| Attach stage positions | **Stage CSV…** (sidebar). Choose the series (`beads` / `analogue`) and time basis (`roi` / `full_frame` / `stage`). |
| Import true sizes from a spreadsheet | **File → Import calibration spreadsheet…** (`.csv` or `.xlsx`). Labels (section 10) are usually easier. |
| Reopen earlier work | **File → Open project…** |

`timeseries.mat` and `full_backgrounds.mat` are skipped. Files that cannot
be read are listed in a single message at the end rather than stopping the
import.

**Stage CSV format.** The CSV needs the columns `start_time` (ISO date-time),
`stage_x_original` and `stage_z_original`. It may also have `series` and
`time_basis` columns. Each particle takes the most recent stage position at
or before its `ROI_N.Time` (zero-order hold, no look-ahead). Legacy
whitespace-separated position logs (`X Z dd/mm/yy HH:MM:SS`) are also
accepted.

When stage positions load, the X axis of the focus/size plots switches to
**Stage X** automatically.

---

## 3. The window at a glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ◉ 111 selected · 111 unreviewed   Accept as [label▾] ✓Accept ✕Reject ↺Unflag │  ← selection bar
├──────────────────────────────────────────────────────────┬───────────────────┤
│ ① Select on plots │ ② Gallery │ ③ Particle               │ + Add MAT / folder│
│ Mouse [Select|Zoom|Pan|Xbest line]  Box [New|Add|Remove| │ Stage CSV / Save  │
│ Keep only|Reject]   New/Add boxes pick [Unreviewed ▾]    ├───────────────────┤
│ ┌─────────────────┐ ┌──────────────────────────────────┐ │ Review │ Labels │ │
│ │  Camera X–Y     │ │  Focus   vs  Stage X / d / time  │ │ Sets/X–Z│Filters│ │
│ │                 │ ├──────────────────────────────────┤ │                   │
│ │                 │ │  Length  (shared X axis)         │ │ counts, history,  │
│ └─────────────────┘ └──────────────────────────────────┘ │ labels, sets …    │
│ Show: ☑Unreviewed ☑Accepted ☑Rejected   legend           │                   │
├──────────────────────────────────────────────────────────┤                   │
│ Browse [Selection▾]  ◀ 12/111 ▶  [thumb] ROI 7 · Accepted │ Open in viewer   │  ← browse bar
└──────────────────────────────────────────────────────────┴───────────────────┘
  status messages …                                        ✓ Flags saved 16:42
```

- **Selection bar (top).** Shows how many particles are selected and has the
  actions that apply to the whole selection. It is visible on every tab.
- **Tab ① Select on plots.** Camera X–Y on the left. On the right, two
  stacked plots (by default Focus on top, Length below) share the same X
  axis. Pick the variables with **X**, **Top** and **Bottom**; set Bottom to
  *(none)* for a single plot.
- **Tab ② Gallery.** Thumbnails of the particles you are browsing.
- **Tab ③ Particle.** One particle shown large, with its measurements.
- **Browse bar (bottom).** Chooses which particles the Gallery, the Particle
  tab and ◀ ▶ step through: *Selection (all if none)*, *All particles*,
  *Selection only*, *Unreviewed*, *Accepted* or *Rejected*. It also shows the
  current particle (☆).
- **Sidebar tabs.** **Review** (counts, how-to, flag history), **Labels**,
  **Sets / X–Z**, **Filters**, and **Advanced** (hidden; turn it on with
  *View → Show experimental calibration tools*).
- **Status bar.** Confirms each action. The bottom-right corner shows when
  the flags file was last saved.

---

## 4. What the colours mean

The same colours are used on the plots, gallery frames, buttons and badges.

| Mark | Meaning |
|---|---|
| grey ● | **Unreviewed.** The default state. |
| **blue ◉** | **Selected.** A temporary working selection, not saved as a decision. |
| **green ●** | **Accepted.** Saved, and skipped by new selection boxes. |
| **red ✕** | **Rejected.** Saved. |
| teal ◎ ring | Member of a named calibration set (section 11). |
| orange ☆ | **Current particle**, i.e. the one shown in the browse bar and Particle tab. |
| orange ▲ | Peak of the median focus trace (section 9). |

Use the **Show** check-boxes under the plots to hide unreviewed, accepted or
rejected particles. Selected particles are always shown.

**Colour** (above the camera plot) can instead colour points by **Label** or
by camera X/Y position.

---

## 5. The basic workflow

1. **Select a cluster on the camera.** On tab ①, drag a box on
   *Camera X–Y*. Unreviewed particles inside it turn **blue**.
2. **Look at the cluster on the focus/length plots.** The same particles are
   highlighted on the plots to the right. Tick **Show only the selection
   here** to hide everything else on those plots.
3. **Drop or reject outliers.**
   - **Ctrl/⌘-drag** a box around outliers to *remove them from the
     selection*. They go back to grey and stay unreviewed.
   - Or click **✕ Reject** in the *Box* row, then drag around outliers to
     *flag them rejected* (red). With a selection active, only selected
     particles inside the box are rejected.
4. **(Optional) choose a label** in **Accept as**, e.g. *PSL 20 µm beads*.
5. **Accept.** Click **✓ Accept selection** or press **K**. The particles
   turn **green**, receive the label and its true size, and the flags file is
   saved immediately.
6. **Repeat for the next cluster.** New boxes skip green and red particles,
   so the same particles are never picked twice.
7. **Spot-check images** in the Gallery (section 8).
8. **Save a project** (**Save project** in the sidebar) to keep calibration
   sets, X/Z references and flag history as well as the flags.

---

## 6. Selecting with the mouse

### Mouse mode (tab ①, *Mouse* row)

| Mode | Drag does |
|---|---|
| **▭ Select** | Draws a selection box (see box modes below). |
| **🔍 Zoom** | Zooms to the rectangle. The selection is unchanged. |
| **✋ Pan** | Moves the view. |
| **↔ Xbest line** | With X = Stage X or Stage Z, drags the orange best-focus line (section 11). |

The **mouse wheel** zooms around the cursor in every mode. **⤢ Fit view**
resets all plots.

### Box modes (*Box* row, or hold a key while dragging)

| Button | Key held while dragging | Effect |
|---|---|---|
| **▭ New** | – | Replace the selection with what is inside the box. |
| **＋ Add** | **Shift** | Add what is inside the box. |
| **− Remove** | **Ctrl** (Windows/Linux) / **⌘** (Mac) | Drop what is inside the box from the selection. No flags change. |
| **∩ Keep only** | **Alt** / **⌥** | Keep only selected particles that are inside the box. |
| **✕ Reject** | – | Flag particles inside the box as rejected (only selected ones, if there is a selection). |

While you drag, the box outline is coloured by the mode it will apply: blue
for select, orange for remove, red for reject.

### Which particles a New/Add box can pick

The drop-down **New/Add boxes pick** controls this:

- **Unreviewed only** *(default)*: accepted and rejected particles are
  skipped, and the status bar tells you how many were skipped.
- **Unreviewed + accepted**
- **Accepted only**: use this to grab green particles, e.g. to unflag them.
- **Rejected only**
- **Everything**

Remove, Keep only and Reject boxes ignore this setting.

### Clicking

| Action | Effect |
|---|---|
| Click a dot | Makes it the **current particle** (☆). The selection is unchanged. |
| Shift-click | Adds the dot to the selection. |
| Ctrl/⌘-click | Toggles the dot in or out of the selection. |
| Double-click | Opens the particle in tab ③. |
| Hover | Shows a small card with the particle's **size in µm (length × width)**, status, label and focus. The image appears too if its MAT file is already in memory. |

Hover previews can be turned off under **View → Hover preview on plots**.

Other ways to select:

- **Review** tab: **Select** next to Unreviewed, Accepted, Rejected or
  *In calibration sets*.
- **Labels** tab: **Select particles with label**.
- **Flag history**: **Select batch**.
- **Edit → Select all visible** (Ctrl+A).

---

## 7. Accept, reject, unflag and undo

| Button (selection bar) | Key on tab ① | Effect on the selection |
|---|---|---|
| **✓ Accept selection** | **K** | Accepted (green). Applies the **Accept as** label. The selection is cleared. |
| **✕ Reject selection** | **R** | Rejected (red). The selection is cleared. |
| **↺ Unflag selection** | **U** | Back to **unreviewed** (grey) and the label is removed, i.e. the default. |
| **Clear selection** | **Esc** | Deselect everything. Flags are unchanged. |
| **↶ Undo** | **Ctrl+Z** | Undo the last selection, flag, label or set change (up to 60 steps). |

- Every flag change is **saved immediately** to the flags CSV (section 13).
- Each Accept or Reject of a group is recorded in **Review → Flag history**
  as a batch, e.g. *Accepted batch 3 · 111 · 16 Oct 16:42*. **Select batch**
  re-selects those particles. **Undo batch…** returns them to unreviewed,
  unless they have been re-flagged since or are also in another batch with
  the same decision.
- To unflag accepted particles, do either of the following, then press
  **↺ Unflag** (U):
  - set *New/Add boxes pick* to **Accepted only** and box them;
  - use **Review → Accepted → Select**.

**Keys act on what you are looking at:**

- **Tab ①:** K/R/U act on the **selection**.
- **Gallery:** K/R/U act on the **picked thumbnails**.
- **Particle tab:** K/R/U act on the **displayed particle**.

---

## 8. Checking images: Gallery and Particle tabs

### Gallery (tab ②)

The Gallery shows whatever the **Browse** drop-down selects. By default that
is the current selection, or all particles if nothing is selected. Each
thumbnail frame shows the particle's state:

- blue: selected;
- green ✓: accepted;
- red ✕: rejected;
- thin grey: unreviewed;
- orange corner: the current particle.

To act on particular thumbnails:

1. **Pick** them: click, Ctrl/⌘-click, Shift-click, or drag a rubber band.
   Picked thumbnails turn amber. **Pick all on page** picks the whole page.
2. Act on them with the buttons below the gallery or the keys:

| Button | Key | Effect on picked thumbnails |
|---|---|---|
| − Remove from selection | **Del** / Backspace | Drop them from the selection (not flagged). |
| + Add to selection | **A** | Add them to the selection. |
| ✓ Accept | **K** | Accept them (with the *Accept as* label). |
| ✕ Reject | **R** | Reject them. They also leave the selection. |
| ↺ Unflag | **U** | Back to unreviewed, with no label. |

Double-click (or **Enter**) opens a thumbnail in the Particle tab. **Esc**
clears the picks. Pages hold 48 thumbnails. Images are loaded one MAT file
at a time and cached, so the first page from a large file takes a moment and
later pages are quick.

### Particle (tab ③)

This tab shows the current particle large, with its stored boundary (toggle
with **B**) and a coloured status badge. You can:

- **Accept (K) / Reject (R) / Unflag (U)** it. Choose a **Reject reason** if
  you like. With *Go to next particle after flagging* ticked, it moves on
  automatically.
- **Add to / Remove from selection** (A / Del).
- Set its **Label** from the drop-down.
- Read its measurements: time, length, width, focus, stage X/Z, the signed
  distance *d* to the object plane (section 11), true size and source file.

**←/→** step through the browse list from any tab except the Gallery.

---

## 9. Median traces

With **Median ± IQR trace** ticked, every focus/size plot shows:

- a dark line: the median of the plotted **non-rejected** particles at each
  stage position (or in equal-count bins along time), with a grey
  inter-quartile band;
- a dashed blue line: the median of the **selection** only;
- on **Focus vs Stage X or Stage Z**, an orange ▲ at the peak of the median
  trace. **Sets / X–Z → Snap to median focus peak** copies its position into
  Xbest or Zbest.

Whether the peak is a maximum or a minimum follows *Advanced → Focus
optimum*, which defaults to maximum. The traces are descriptive only: no
model is fitted and no values are changed.

---

## 10. Labels with true sizes

Labels record *what* you accepted (e.g. *PSL 20 µm*, *Ice analogue column A*)
and its **true size** in µm.

### Creating and using labels

1. **Labels** tab → **＋ New…** (or choose *＋ New label…* in **Accept as**).
2. Enter:
   - a name;
   - a type: *bead, ice crystal, ice analogue, droplet* or *other*;
   - the true length or diameter in µm;
   - the true width (or tick *Width = length* for spheres);
   - notes such as supplier or lot number;
   - a plot colour.
3. Choose the label in **Accept as** before accepting. Every particle you
   accept from then on (selection, gallery or viewer) is tagged with it.

Other actions on the **Labels** tab:

- **Select particles with label**, **Add them to selection**, **Browse them**;
- **Tag selection with label** gives the label and size without changing
  accepted/rejected flags;
- **Remove labels from selection**.

### Shared across datasets, edited once

- **There is one label library per computer**: `~/.cpi_calibration/labels.json`.
  Every dataset uses it.
- **To share it with your group**, put it on a shared drive and either set
  the environment variable `CPI_CALIBRATION_LABELS=/path/to/labels.json` or
  use **Labels → Use another library file…**.
- **Particles store only the label name.** Their true size comes from the
  library, so **Edit / resize…** updates every particle carrying that label
  in every dataset.
- **Renaming is safe.** The old name is kept as an alias, so older datasets
  still match.
- **Projects save a copy of the labels they use.** When a project is opened
  on another computer, any missing labels are added to that computer's
  library. Existing labels there are not overwritten.
- **Deleting a label** removes it from the library and strips it from the
  loaded particles. Their flags are kept.

The **Measured / actual length** plot metric and the *Actual L × W* readout
use the label sizes. **Colour: Label** colours accepted particles by label.

---

## 11. Best focus, calibration sets and which side of the object plane

### Setting the best-focus position

1. Plot **Focus** against **Stage X** (or **Stage Z**). An orange dashed line
   marks the draft *Xbest* (or *Zbest*).
2. Move it in any of these ways:
   - drag it in **↔ Xbest line** mouse mode;
   - type the value in **Sets / X–Z**;
   - click **Snap to median focus peak**.
3. The value stays a **draft** until you save it with a set.

### Calibration sets

A calibration set is a named snapshot of particle IDs **plus** its Xbest/Zbest.

- **Save selection as set + X/Z** saves the current selection with the
  current Xbest/Zbest. A handy way to build one: **Review → Accepted →
  Select**, then save.
- Click a set to make it **active**. Its X/Z then loads into the editor and
  applies to its members on the plots.
- **Save X/Z changes** corrects a set's X/Z without changing its members.
  **Clear X/Z** removes the reference.
- **Select set / Add to selection / Delete set… / Unflag members…** do what
  they say.
- **Add current / Remove current** add or remove the ☆ particle from the set.

Sets are stored in the project file, so **save a project** to keep them.

### Which side of the object plane? The signed distance *d*

Three position measures are available on the X axis. Only one of them keeps
the sign and is measured from the plane itself:

| X axis option | Formula | Signed? | Problem |
|---|---|---|---|
| Zp = √(ΔX² + ΔZ²) | distance from the best-focus point | no | Loses which side you are on. |
| Zp radial difference *(legacy)* | √(Xbest²+Zbest²) − √(X²+Z²) | yes | Measures distance from the stage **origin**, not the plane. |
| **Signed distance to object plane d** | **(X−Xbest)·cos θ + (Z−Zbest)·sin θ** | **yes** | — |

*d* is the offset from best focus projected onto the object-plane normal
n = (cos θ, sin θ), with θ measured in the stage X–Z frame from +X towards
+Z. *d* > 0 and *d* < 0 are **opposite sides of the plane**, and *d* = 0 is
on it. Units are those of the stage log.

To set θ, go to **Sets / X–Z → Object plane**:

- **Normal θ** defaults to **45°**. **Set this for your jig and instrument**
  (the CPI 1.0 object plane is flipped relative to CPI-3V).
- **Estimate θ from scan direction** fits the direction the stage moved (of
  the selection, if any). This is valid when the jig moves particles straight
  through the plane.
- **Flip side** adds 180°, swapping which side counts as positive.
- **Check θ:** plot X = **Offset within object plane**. For a straight scan
  through the plane it should stay close to 0 if θ is right.

The exported `*_relative_positions.csv` (section 13) contains
`plane_normal_deg`, `d_signed_plane` and `d_in_plane` alongside the older
columns.

---

## 12. Filters

The **Filters** tab hides particles from every view without changing them:

- minimum and maximum length, width, focus, image x/y, and measured/true
  length ratio;
- an optional time window;
- image type: *Cropped ROIs only* (default), *All images*, or *Full frames
  only*.

**Show data ranges** shows the minimum and maximum of the loaded data as
hints. Press **Apply** or Enter to apply.

---

## 13. Files the workbench writes

| File | Contents | When written |
|---|---|---|
| `<project>.json` | Source file list, stage CSV, calibration sets with X/Z, flag-history batches, the selection, plot settings, θ, a copy of the labels used | **Save project**, and on every set change once a project exists |
| `<project>_flags.csv` | One row per particle: `status` (undecided/keep/reject), `reject_reason`, `label`, true sizes, object-plane position, plus the measured values | **Immediately**, on every flag or label change |
| `calibration_selection.csv` | Same as above, used when **no project** is open. It sits next to the first data file and is reloaded automatically next time. | Immediately |
| `<project>_relative_positions.csv` | One row per (calibration set, particle): stage X/Z, Xbest/Zbest, `x_minus_xbest`, `z_minus_zbest`, `z_p_radial`, `z_p_from_offsets`, `plane_normal_deg`, `d_signed_plane`, `d_in_plane` | On project save |
| `~/.cpi_calibration/labels.json` | The shared label library | When labels are created or edited |

The status bar shows **✓ Flags saved hh:mm:ss** after each save, or a red
warning if saving failed. **File → Save flags as…** writes the flags to
another CSV.

When you close the window, it offers to save a project if you have
calibration sets or flag history that is not yet in one. Flags are already
safe either way.

---

## 14. Keyboard and mouse reference

| Key | Tab ① plots | Gallery | Particle tab |
|---|---|---|---|
| **K** | Accept selection | Accept picked | Accept particle |
| **R** | Reject selection | Reject picked | Reject particle |
| **U** | Unflag selection | Unflag picked | Unflag particle |
| **A** | Add current ☆ to selection | Add picked to selection | Add to selection |
| **Del / Backspace** | Remove current ☆ from selection | Remove picked from selection | Remove from selection |
| **Esc** | Clear selection | Clear picks (then selection) | Clear selection |
| **Enter** | Open current ☆ | Open picked | – |
| **← / →** | Previous / next particle | (move between thumbnails) | Previous / next particle |
| **B** | – | – | Toggle boundary overlay |
| **Ctrl+Z** | Undo | Undo | Undo |
| **Ctrl+A** | Select all visible (pickable) | | |

| Mouse on plots | Effect |
|---|---|
| Drag | Box in the current box mode |
| Shift / Ctrl(⌘) / Alt(⌥) + drag | Add / Remove / Keep only |
| Click | Inspect (make current ☆) |
| Shift-click / Ctrl(⌘)-click | Add / toggle one particle |
| Double-click | Open in Particle tab |
| Wheel | Zoom around cursor |
| Hover | Size (µm), status, label, focus |

---

## 15. Tips and troubleshooting

- **"Box changed nothing … particles inside are not 'Unreviewed only'".**
  Everything in the box is already accepted or rejected. Change *New/Add
  boxes pick* if you meant to grab them.
- **The focus/size plots say "No stage X/Z positions yet".** Import the
  Stage CSV. If none match, check the series (beads/analogue), time basis
  and clock conventions.
- **"No best-focus X/Z reference".** Relative axes (*d*, X − Xbest, …) need a
  draft or saved Xbest/Zbest. Select particles, or activate a set with a
  reference.
- **Plots look empty after filtering.** Check the **Filters** tab, the
  **Show** toggles, and *Show only the selection here*.
- **Large files.** Each MAT file can be ~200 MB. The workbench holds one file
  in memory at a time and caches thumbnails. Hover images appear only for
  the file already in memory, to avoid freezing; click the dot to load it.
- **Zoom got stuck.** Press **⤢ Fit view**.
- **Lost a selection.** **Ctrl+Z** restores it.

---

## 16. Code layout

| File | Purpose |
|---|---|
| `gui.py` | The PySide6 workbench window (`cpi-calibrate`). |
| `data.py` | Reading MAT files (`ROI_N`, `dat`), the particle catalog, the flags CSV, stage-position logs. |
| `selection_state.py` | Box/selection set operations, pick scopes, named sets and review batches. |
| `labels.py` | The shared label library and applying true sizes to particles. |
| `reference_sets.py` | Best-focus references, Zp variants, the **signed object-plane distance**, scan-direction estimate, CSV export rows. |
| `plot_utils.py` | Data limits, median scan curves and median ± IQR traces. |
| `focus_surface.py` | Experimental tilted best-focus surface fit (Advanced tab). |
| `metadata.py` | Calibration truth-spreadsheet import. |

Tests live in `tests/` at the repository root. Run them from the repository
root:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python -m pytest tests
```

`test_gui_workflow.py` drives the window headless through select → refine →
accept → skip → unflag → undo.
