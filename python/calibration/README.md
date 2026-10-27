# CPI Calibration Workbench (v12)

An interactive tool for choosing calibration particles (beads, ice-crystal
analogues …) from processed CPI-3V / CPI 1.0 `.mat` files. You select
particles on a camera X–Y plot, refine the selection on focus and size plots,
then accept or reject them. Accepted particles can carry a label with their
true size, and each accepted cluster (batch) is saved with its own best-focus
stage position.

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
11. [Best focus and which side of the object plane](#11-best-focus-and-which-side-of-the-object-plane)
12. [Sizing fit, retrieval and depth of field](#12-sizing-fit-retrieval-and-depth-of-field)
13. [Filters](#13-filters)
14. [Files the workbench writes](#14-files-the-workbench-writes)
15. [Keyboard and mouse reference](#15-keyboard-and-mouse-reference)
16. [Tips and troubleshooting](#16-tips-and-troubleshooting)
17. [Code layout](#17-code-layout)

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
| Remove MAT files | **File → Remove MAT files from project…**, then tick the files. Their particles leave the project and batches lose them (empty batches are removed). Their decisions and labels are set aside in `<flags>_removed.csv` and come back automatically if you add the file again. The MAT files themselves are never deleted. |
| Attach stage positions | **Stage CSV…** (sidebar). By default every row is used — you do **not** have to say whether the run was beads or analogues; that is decided later by flagging and labelling particles. You are only asked to choose when the CSV mixes several series or time bases (then *All series* and *roi* are the defaults). |
| Import true sizes from a spreadsheet | **File → Import calibration spreadsheet…** (`.csv` or `.xlsx`). Labels (section 10) are usually easier. |
| Reopen earlier work | **File → Open project…** |

`timeseries.mat` and `full_backgrounds.mat` are skipped. Files that cannot
be read are listed in a single message at the end rather than stopping the
import.

**Stage CSV format.** The CSV needs the columns `start_time` (ISO date-time),
`stage_x_original` and `stage_z_original`. Optional `series` and `time_basis`
columns let one CSV hold several runs. If the same timestamp appears with
different positions (overlapping series), you get a warning so you can
restrict the import. Each particle takes the most recent stage position at
or before its `ROI_N.Time` (zero-order hold, no look-ahead). If that row has
an `end_time_exclusive` and the particle is at or after it, the stage was
between positions, so the particle gets no position (v14.16). A blank end
means "until the next row starts". Legacy whitespace-separated position logs
(`X Z dd/mm/yy HH:MM:SS`) are also accepted.

**Which time basis.** Since v14, an ROI's time is the time of the camera frame
it came from. So use stage times taken from the frames (`full_frame`), not the
older `roi` times. The import checks this for you. Particles are recorded in
bursts after each stage move, so each burst should get one position. If
bursts are split between two positions, you are warned. That is the usual
sign of the wrong time basis, or of a time rounded into a burst.

On 21/3/02 the `roi` basis split 93 of 157 bursts and gave 19 whole bursts
(11,456 flagged particles) the neighbouring step's position. The `full_frame`
basis split 1 (a stage move logged during a long burst).

When stage positions load, the X axis of the focus/size plots switches to
**Stage X** automatically.

---

### Projects: new, open, close

* **File ▸ New project…** asks for a file name, saves the project you have
  open, then starts an empty one.
* **File ▸ Open project…** saves the open project, then loads the chosen one.
* **File ▸ Close project** saves and closes, leaving an empty workbench.
* **Starting over with different MAT files:** use **File ▸ New project…** (it starts empty), then add the new files. Or use **Remove MAT files from project…** to drop some files from the project you have open.
* Closing the window also saves.
* **File ▸ New window** opens another workbench window, so you can have
  several projects open side by side, one per window.
  * Opening a project that's already open in another window switches to that
    window instead. Two windows on one project would overwrite each other's
    saves.
  * All windows share one label library, so a label edited in one updates the
    others.
  * To fit several projects together you don't need them open: use
    Calibration ▸ Fit … from several projects (section 12).
* **File ▸ Merge projects into a viewing copy…** makes one new project
  containing everything from two or more projects, for looking at several
  runs together (e.g. focus against d across days). The originals are not
  changed.
  * It brings in their data, flags, labels and batches (renamed
    `<project> · …`) and every stage CSV.
  * It is a **copy**: the title says so, and changes made in it don't reach
    the originals. Label in the original projects, and fit with
    **from several projects**, not by fitting a merged copy.

The window title shows which project is open. A workbench with no project
asks whether to create one before closing it, if you've made batches.

## 3. The window at a glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ◉ 111 selected · 111 unreviewed   Accept as [label▾] ✓Accept ✕Reject ↺Unflag │  ← selection bar
├──────────────────────────────────────────────────────────┬───────────────────┤
│ ① Select on plots │ ② Gallery │ ③ Particle               │ + Add MAT / folder│
│ Mouse [Select|Zoom|Pan|Xbest line]  Box [New|Add|Remove| │ Stage CSV / Save  │
│ Keep only|Reject]   Boxes pick [Unreviewed ▾]            ├───────────────────┤
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
  **Best focus**, **Filters**, and **Advanced** (hidden; turn it on with
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
   highlighted on the plots to the right. **Only selection / current batch**
   is on by default: while you have a selection (or a batch shown), the
   focus/size plots show only that. With nothing selected they show
   everything. Press **S** (or untick it) to see all particles at any time,
   and S again to go back.
3. **Drop or reject outliers.**
   - **Ctrl/⌘-drag** a box around outliers to *remove them from the
     selection*. They go back to grey and stay unreviewed.
   - Or click **✕ Reject** in the *Box* row, then drag around outliers to
     *flag them rejected* (red). With a selection active, only selected
     particles inside the box are rejected.
4. **(Optional) choose a label** in **Accept as**, e.g. *PSL 20 µm beads*.
   **One batch from several areas:** after the first box, hold **Shift** while
   dragging (or click **＋ Add**). Each box adds to the selection, and Accept
   makes them all one batch.
5. **Accept.** Click **✓ Accept selection** or press **K**. The particles
   turn **green**, receive the label and its true size, and the batch is saved
   with the cluster's own best focus (the orange line, which follows the
   cluster's focus peak; section 11). The flags file is saved immediately.
   With *Only selection / current batch* ticked, the focus/size plots keep
   showing **the batch you just accepted** (in green), so you can check it.
   The line above the plots says what is shown. Your next box releases it,
   and **Esc** shows everything.
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
for select, orange for remove, red for reject. **Changed your mind mid-drag?
Press Esc (or right-click) before releasing** — the box is discarded and
nothing changes.

### Which particles a New/Add box can pick

The drop-down **Boxes pick** controls this (New and Add boxes only):

- **Unreviewed only** *(default)*: in a box that mixes states, accepted and
  rejected particles are skipped, and the status bar says how many. **If the
  box contains only flagged particles, they are selected anyway**, so you can
  go back to an accepted cluster (to unflag, re-label or save it as a set)
  without changing this setting.
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
| Hover | Shows a small card with the particle's **size in µm (length × width)**, status, label, focus, local illumination, **which batch it is in and the X/Z it uses** (*own*, or borrowed from a nearby batch), and its **image**. If the particle's MAT file isn't in memory yet, rest on the dot for a moment: the file loads in the background ("loading image…") and the image appears. Clicking the dot also loads it. |

Hover previews can be turned off under **View → Hover preview on plots**.

Other ways to select:

- **Review** tab: **Select** next to Unreviewed, Accepted or Rejected.
- **Labels** tab: **Select particles with label**.
- **Flag history**: **Select batch**.
- **Edit → Select all visible** (Ctrl+A).

---

## 7. Accept, reject, unflag and undo

**Reject or remove?**

- **Reject** (R, ✕ Reject box) is for particles that are **not usable at
  all**: debris, overlapping particles, clipped images, artefacts. They turn
  red and are excluded everywhere. Pick a reason (e.g. *debris*) in the
  Particle tab if you like.
- **Remove from selection** (Ctrl/⌘-drag, Del, − Remove) is for good
  particles that just **don't belong in this batch** (e.g. a different bead
  size). Nothing is flagged; they stay grey and unreviewed for a later batch.


| Button (selection bar) | Key on tab ① | Effect on the selection |
|---|---|---|
| **✓ Accept selection** | **K** | Accepted (green). Applies the **Accept as** label. The selection is cleared. |
| **✕ Reject selection** | **R** | Rejected (red). The selection is cleared. |
| **↺ Unflag selection** | **U** | Back to **unreviewed** (grey) and the label is removed, i.e. the default. |
| **Clear selection** | **Esc** | Deselect everything. Flags are unchanged. |
| **↶ Undo** | **Ctrl+Z** | Undo the last selection, flag, label or best-focus change (up to 60 steps). |
| **↷ Redo** | **Ctrl+Shift+Z** / **Ctrl+Y** | Redo what you just undid. Any new action clears the redo steps. |

- Every flag change is **saved automatically** within about 1.5 s, in the background (section 14).
- **Flag history shows each accepted batch's label** (🏷 *100 micron glass*,
  *no label*, or *mixed: 2 labels*); hover over a batch for the full count
  per label. To label a whole batch: double-click it in Flag history, then
  **Labels → choose the label → Tag selection with label**.
- Each Accept or Reject of a group is recorded in **Review → Flag history**
  as a batch, e.g. *Accepted batch 3 · 111 · 16 Oct 16:42*. **Select batch**
  re-selects those particles. **Undo batch…** returns them to unreviewed,
  unless they have been re-flagged since or are also in another batch with
  the same decision.
- **Taking particles out of a batch:** select the batch (Flag history →
  double-click), then **Reject** (R) the bad ones, or **Unflag** (U) ones that
  don't belong. They leave the batch at once: its count drops, *Select batch*
  no longer includes them, and a batch left empty disappears. Everything is
  saved immediately (✓ Flags saved). Ctrl/⌘-drag and Del only change the
  temporary selection and save nothing. Don't press K again on the rest: they
  are already accepted, and K would create a duplicate batch. Rejected
  particles are excluded everywhere; untick **Show ✕ Rejected** to hide them
  from the plots too. The MAT files themselves are never changed.
- To unflag accepted particles, do either of the following, then press
  **↺ Unflag** (U):
  - box them (a box holding only accepted particles selects them);
  - use **Review → Accepted → Select**.

**Keys act on what you are looking at:**

- **Tab ①:** K/R/U act on the **selection**.
- **Gallery:** K/R/U act on the **picked thumbnails**.
- **Particle tab:** K/R/U act on the **displayed particle**.

---

### Trimming a batch after Accept

After **Accept**, the focus/length plots keep showing the batch you just
accepted. Clicking a batch in **Flag history** shows that batch instead. To
take particles out of it on the plots:

* **Drag a box** around them (nothing needs to be selected). Only that
  batch's particles inside the box are selected; other batches are never
  touched. Then:
  * **U** returns them to unreviewed and out of the batch;
  * **R** rejects them as debris.
* Or use the **✕ Reject** box mode and drag. With a batch on show and
  nothing selected, it rejects that batch's particles inside the box.

Both leave the rest of the batch accepted, and Ctrl+Z undoes. If you take out
a lot of particles, re-centre the batch's Xbest/Zbest with **Set batch X/Z
from their focus peaks** (section 11).

### Changing labels, and working on several batches at once

**Change a batch's label:** click it in **Flag history**, then press
**Change label…** and pick the right label (or *New label…*). Every particle
in the batch gets the new label and its true size. The accept/reject flags
don't change, and Ctrl+Z undoes it.

**Change the label of particles you've selected:** use **Edit ▸ Change label
of selection…** or press **L**. For example, select a few accepted particles
that really belong to another bead size.

**Several batches together:** **Ctrl/⌘-click** or **Shift-click** batches in
Flag history.

* The focus/length plots then show all the chosen batches together, each at
  its own saved Xbest/Zbest. The line above the plots says
  "Showing: N batches".
* **Select batch** selects all of their particles.
* **Change label…** relabels all of them in one go.
* The X/Z buttons (Save, Clear) still act on one batch at a time, the one
  last clicked.
* Esc shows everything again.

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

### Measuring on images (Tools ▸ Measure images…, Ctrl+M)

This opens a separate window for measuring distances by hand. Use it for a
graticule, to check a particle's size, or for particles whose size isn't known.
The **📏 Measure on this ROI / full frames…** button on the Particle tab opens
it on the current particle.

- **What to view.** Choose one of the project's MAT files, or open another
  one. The list on the left can show:
  - the *full frames* in that file (full-size images and ROIs at least
    512 × 512 px);
  - *all ROIs* in that file (or type a number under **Go to ROI**);
  - the *backgrounds* from the same folder's `full_backgrounds.mat`;
  - *camera frames rebuilt*. The CPI saves only the ROIs from each camera frame
    (plus occasional full backgrounds). This option puts each frame's ROIs (all
    ROIs with the same time) back at their positions on the background nearest
    in time, with the ROI boxes and stored outlines drawn on. **Whole camera
    frame of this ROI** jumps from an ROI to its frame. Without a
    `full_backgrounds.mat`, the ROIs are drawn on grey.

  **Open image file…** opens a PNG, TIFF, BMP or JPEG instead.
- **Orientation.** All images in the workbench (Particle tab, gallery, hover
  previews and this window) are drawn the way the camera sees them: camera X
  across and Y down, the same way round as the Camera X–Y plot. (Since v14.16.
  Before that they were drawn transposed, i.e. mirrored about the diagonal.)
  Inside an ROI, x and y count from its corner. The cursor readout and the CSV
  also give absolute camera coordinates (the ROI corner plus the position in
  the ROI).
- **Measuring.**
  - Left-drag to draw a line at any angle. Hold **Shift** to keep it at 0°,
    45° or 90°.
  - Drag either end of a line to adjust it. Click a line to select it, and
    press **Delete** to remove it.
  - The wheel zooms about the pointer, and right- or middle-drag pans. Pixels
    stay sharp squares, so ends can be placed to a fraction of a pixel when
    you zoom in.
  - Coordinates are in pixels of the image as shown: a pixel's centre is at
    (column + 0.5, row + 0.5).
- **What's shown.**
  - Each line's length in px, its angle, and its length in µm when *Also show
    µm* is ticked with a pixel size.
  - The mean ± sd of the lines on that image.
  - The brightness profile along the selected line, with a little either side.
    It helps you put the ends on edges consistently.
  - *Show stored outline* draws the outline from the processing over the ROI.
  - *Stretch contrast* helps with faint graticules.
- **Pixel size from a graticule.** Draw a line across a known distance (e.g. N
  divisions), select it, and use **Pixel size from selected line…**. Enter the
  true length in µm and the pixel size is set from it.
- **Export.**
  - **Export image as PNG** writes the image enlarged with sharp pixels (auto:
    about 1000 px across), with the lines and their lengths, the outline if
    shown, a scale bar if a pixel size is set, and a caption naming the file,
    image and time.
  - **Export all measurements as CSV** writes every line measured in the
    session, on every image: file, image, coordinates, length in px and µm,
    and angle.

Measurements are kept per image while the window is open, so you can move
between images and come back to them. They are not saved in the project:
export the CSV to keep them.

It can also run on its own: `python -m calibration.measure_window file.mat image.png …`.

---

## 9. Median traces

With **Median ± IQR trace** ticked, every focus/size plot shows:

- a dark line: the median of the plotted **non-rejected** particles at each
  stage position (or in equal-count bins along time), with a grey
  inter-quartile band;
- a dashed blue line: the median of the **selection** only;
- on **Focus vs Stage X or Stage Z**, an orange ▲ at the peak of the median
  trace. **Best focus → Snap to the selection's focus peak** copies its position into
  Xbest or Zbest.

Whether the peak is a maximum or a minimum follows *Advanced → Focus
optimum*, which defaults to maximum. The traces are descriptive only: no
model is fitted and no values are changed.

---

### Illumination and focus

The focus metric measures edge sharpness, so to first order it scales with
how brightly the particle is lit. Laser power drifts in time and the beam is
not uniform across the camera. The workbench therefore measures each ROI's
**local illumination**: the median brightness of the ROI's outer 2-pixel
frame. That frame is the background immediately around the particle, at the
moment and place it was imaged. It does not use the frame's overall image
mean.

- **Plot it:** choose *Local illumination* as the X axis (or Top/Bottom Y) of
  the focus/size plots, or as the camera plot's **Colour**. *Focus* against
  *Local illumination* shows the dependence directly.
- **Remove it:** *Focus ÷ local illumination* is focus × (median
  illumination / ROI illumination), i.e. focus rescaled to typical lighting.
  Plot it against Stage X/Z like focus.
- **Use it for best focus:** tick **Best focus → Find peaks with focus ÷
  local illumination** so each cluster's peak, the line it moves, and the
  one-click batch fill use normalised focus.
- **Exclude badly lit ROIs:** **Filters → Local illumination** min/max. Very
  low edge brightness usually means the particle fills the ROI, so the frame
  isn't background and the normalisation is unreliable.

The hover card, the Particle tab and the positions CSV (`local_illum`,
`focus_normalised`) all show these values.

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

- **One label library per `CPI_cals` tree:** `CPI_cals/ReferenceLibrary/labels.json`.
  Both `CPICalibration/` and `ICEAnalogues/CPI/` use this shared location.
- **To choose a different location**, use **Labels → Choose reference library folder…**.
  The choice is remembered. **Use CPI_cals/ReferenceLibrary automatically**
  restores portable root discovery. An advanced `CPI_CALIBRATION_LABELS`
  environment override remains supported. The legacy home library is only a
  migration source (copying it requires confirmation).
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

## 11. Best focus and which side of the object plane

### Setting the best-focus position

1. Plot **Focus** against **Stage X** (or **Stage Z**). An orange dashed line
   marks *Xbest* (or *Zbest*). When you select a cluster it jumps to the
   cluster's own focus peak.
2. Override it in any of these ways:
   - drag it in **↔ Xbest line** mouse mode;
   - type the value in the **Best focus** tab;
   - click **Snap to the selection's focus peak**.
3. **Accept (K)** saves it with the new batch. For an existing batch, click it
   in **Review → Flag history**, adjust, and press **Save Xbest/Zbest to
   batch** (in Flag history or the Best focus tab).

The project stores every batch's X/Z, and the editor's current values are
restored when the project is reopened.

### Best focus per batch (tilted slides)

If the slide is not exactly parallel to the object plane, particles on
different parts of the slide come into focus at different stage positions.
So **each accepted batch, i.e. each cluster you boxed, gets its own
Xbest/Zbest**, and this happens as part of accepting.

1. **Box a cluster** on the camera plot. The focus plot shows the cluster's
   own median trace (dashed blue) and its peak (blue ▲). **The orange
   Xbest/Zbest line jumps to that peak** (*Line follows focus peak*). The
   peak is found along the object-plane normal, so d = 0 lands on the focus
   maximum. Xbest/Zbest are the median stage X and Z of the particles at the
   peak step, so an X-only scan keeps its own Z.
2. **Adjust if needed.** Drag the line (↔ Xbest line mode) or type a value.
   Your value then stays put while you refine the cluster with more boxes.
   It is released when you box a different cluster.
3. **Accept (K).** The tick box next to *Accept as* says exactly what will
   be saved. For example, **+ save Xbest 10.8 · Zbest 5** (in orange) means
   the new batch gets those values; the status bar confirms them after the
   accept. Other states:
   * **no peak found — not saved** (red): no X/Z is saved, and the batch
     borrows one (below). Move the line to set it.
   * **kept from their batches** (grey): every selected particle already
     belongs to a batch with its own X/Z. Accepting again never changes it.

**Check the tilt:** set *Colour* to **Batch Xbest** (or **Batch Zbest**). A
tilted slide shows a smooth gradient across the camera. A batch whose colour
doesn't fit its neighbours probably has a wrong X/Z: click it in **Flag
history** to see its peak, adjust the line, and press **Save Xbest/Zbest to
batch**.

**Set batch X/Z from their focus peaks (d = 0 at max focus)…** (Review →
Flag history) puts each batch's d = 0 at the peak of its own median focus
trace, in one step. It offers two choices:

* **All batches (replace existing):** re-centres every accepted batch,
  including ones whose X/Z you set before.
* **Only those without X/Z:** for batches from older projects.

Ctrl+Z undoes it. Batches too small to show a peak are listed and left
unchanged.

**Which X/Z a particle uses** (plots and exported CSV), in order:

1. **Its own batch's or set's X/Z**, exactly as saved. This is true even
   when you select all accepted particles, so *Accepted → Select* shows every
   batch at its own d = 0.
2. **Otherwise, the nearest calibrated batch on the camera in the same
   session.** A session is a run of acquisition with no gap of 30 minutes or
   more. On a tilted slide, the nearest cluster is the best guess.
3. **Otherwise, the most recent X/Z defined earlier in time.**
4. **While editing,** the Xbest/Zbest editor is previewed on the batch you
   clicked in Flag history, and on selected particles that have no saved X/Z
   yet (the new cluster). It is never previewed on particles that already
   have a batch X/Z.

The particle readout and the exported CSV say which rule applied
(`reference_kind` = own / nearest / step) and which batch it came from.

**Groups from older projects.** Earlier versions also had named
"calibration sets". When such a project opens, its sets appear in Flag history
(marked ◆) and behave like batches, X/Z included. *Undo batch…* on one simply
removes the group. To name groups of particles now, use **labels**.

Saving the project writes `<project>_particle_positions.csv`: one row per
accepted particle, with its label, true and measured size, focus, stage X/Z,
the Xbest/Zbest it uses and where that came from, X−Xbest, Z−Zbest and the
signed distance *d*. This is the table to fit calibrations from.

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

To set θ, go to **Best focus → Object plane**:

- **Normal θ** defaults to **45°**. **Set this for your jig and instrument**
  (the CPI 1.0 object plane is flipped relative to CPI-3V).
- **Estimate θ from scan direction** fits the direction the stage moved (of
  the selection, if any). This is valid when the jig moves particles straight
  through the plane.
- **Flip side** adds 180°, swapping which side counts as positive.
- **Check θ:** plot X = **Offset within object plane**. For a straight scan
  through the plane it should stay close to 0 if θ is right.

The exported `*_relative_positions.csv` (section 14) contains
`plane_normal_deg`, `d_signed_plane` and `d_in_plane` alongside the older
columns.

---

## 12. Sizing fit, retrieval and depth of field

Once particles are accepted, labelled with a true size and have a best-focus
X/Z (so the signed distance *d* is known), **Calibration ▸ Fit size & focus
model…** fits the calibration and tests it. The same thing runs from the
command line with `cpi-fit` and `cpi-retrieve`. The fit writes one JSON file
with everything the retrieval needs.

### Why measured beads come out larger than 100 µm at best focus

Mostly it is not the pixel size. `proc/imageStats.py` builds the particle
mask with `Path(contour).contains_points(points, radius=2)`, which grows the
outline by about a pixel on every side. Running known disks through exactly
those processing steps shows a constant +2.0 px on the diameter:

| true diameter | `radius=2` (current) | `radius=0` |
|---|---|---|
| 40 µm (17.4 px) | 44.4 µm | 40.2 µm |
| 100 µm (43.5 px) | 104.6 µm | 100.5 µm |
| 150 µm (65.2 px) | 154.5 µm | 150.4 µm |

That is +4.6 µm whatever the size: +4.6% at 100 µm, +46% at 10 µm. A
pixel-size error would instead scale with size. The fit therefore uses
`L_app = s·L·(oversize) + b`, with *b* = 4.6 µm by default and *s* fitted.

- *s* ≈ 1 means the 2.3 µm pixel is right and the excess is the mask pad
  (plus edge blur).
- With beads of a single size, *s* and *b* cannot be told apart. Tick **fit it**
  only when two or more true sizes are labelled.

The independent check is **bead tracking**. When a stage step moves the slide
within the image plane (`d_in_plane` changes), every bead stuck to the slide
shifts by the same vector. Image shift in pixels against the move in µm gives
the pixel size directly, with no diffraction or edge detection involved. It
appears in the report and figure whenever the scan provides such steps.

### The model

Each side of the object plane is fitted separately, because the papers and
the data both show the oversizing is asymmetric.

* **Length:** `L_app = s·L·(1 + A±·Zd^p±·(I/I_ref)^κ) + b`, where
  `Zd = λ|d − d0|/(L/2)²` and λ = 0.85 µm. Using the Fresnel number
  (Korolev; Connolly et al. 2007) is what lets a fit made from one size
  extrapolate to other sizes. That extrapolation is theory, not tested, until
  a second bead size is labelled.
* **Focus:** `foc = F0·(I/I_ref)^γ·exp(−(|d − d0f|/w±)^q) + floor`. The
* **Testing the Z/L² scaling.** With one bead size, Z and Z/L² can't be
  told apart (L is constant), so Z/L² is assumed. With **two or more true
  sizes**, three things are fitted instead:
  * the power of L, `Zd ∝ |d|/L^m` (m = 2 is the Fresnel number);
  * the size dependence of the focus peak, `× (L/L_ref)^c`;
  * the size dependence of the focus width, `× (L/L_ref)^k`.

  m ≈ 2 confirms Z/L²; c ≈ k ≈ 0 means focus depends on Z alone (Connolly et
  al. 2007). The first panel then plots all sizes against signed Zd: if the
  scaling holds they fall on one curve. Small beads sit higher at focus
  because the edge offset *b* is a larger fraction of them. The depth of
  field is reported per size, and the validation gives the error per bead
  size, marking any size where the correction does not help.
* **Small particles:** oversizing grows as 1/L², so near focus a small
  particle slightly out of focus can look like a larger one in focus. Size
  and focus alone may not separate them, and the per-size table shows
  whether they do for your data. A size dependence of focus (c ≠ 0) is what
  helps.
  linear dependence on local illumination is the first-order laser-power
  effect.
* **Scatter about both models** (log-normal, widening away from focus). The
  scatter is what makes the retrieval's uncertainties honest.
* *d0* (size minimum) and *d0f* (focus peak) are fitted separately. A
  difference tells you that "best focus" and "smallest image" are not the
  same place.

### Retrieval: side known vs side unknown

For each particle the posterior over a grid of (L, d) is evaluated from its
measured length, focus and local illumination. Priors are uniform in *d*
across the calibrated range and uniform in log *L*.

* **Side known** (`--side +` / `-`): for the calibration test.
* **Side unknown** (`--side both`, the default): for cloud data. Both sides
  stay in with equal prior weight. The reported size is the posterior median
  with a 16–84 % interval, plus `p_plus` = P(d > 0).

**Recommendation for real cloud data: marginalise over the side.** The
alternatives either pick a side (the old least-squares solve, which jumps
between two solutions and hides the ambiguity) or use a symmetric |d| model
(which throws away the asymmetry that is really there).

Marginalising is unbiased when the sides are genuinely ambiguous. Its
intervals widen exactly where the two sides disagree, and it costs little
near focus, where both sides give nearly the same size. The validation shows
how much it costs: the "side unknown" against "side known" rows, and the
fraction of particles whose side the data actually pins down (`p_plus` > 0.9
or < 0.1). On synthetic beads with the asymmetry of the papers:

* side known gives 5.7% mean error;
* side unknown gives 7.7%;
* uncorrected is 18.7%;
* size and focus alone identify the side only ~60% of the time.

If the bead data show that the image itself distinguishes the sides
(Poisson spot, ring contrast), a side classifier trained on these labelled
beads can replace the 50/50 prior. Adopt it only if it beats the
"side identified" figure in held-out batches.

### Which processing, and comparing edge methods

Each particle records the edge method that measured it (`processing` in the
particle table: `legacy`, or e.g. `transmittance T=0.5 denoise=1`). The fit
report lists it. Particles cut off by the ROI border (`edge_touch = 1`, new
method only) are left out of the fit, because their size is only a lower
limit.

To show what a new edge method changes, reprocess a copy of the data (see the
main README, section 3.6). Then:

1. **File ▸ Copy project onto reprocessed data…** gives the same particles,
   flags, labels, batches and X/Z on the new measurements. Each flagged ROI is
   found in the new MAT file by its raw pixels and corners, not by its time, so
   the flags carry across even if reprocessing renumbers the ROIs or changes
   their times. It reports how many it matched. This needs the original MAT
   files to still be where the project points; otherwise it falls back to
   matching by time and camera position.
2. **Calibration ▸ Compare processing runs…** puts the two calibrations side
   by side.

**Focus on transmittance (new edge method)** is available as a plot metric,
as are the **equivalent-area diameter** and its ratio to the actual size.

**Normalised distance** Zd = λd/(L/2)² (λ = 0.85 µm, L = the label's actual
size) is available as an X axis. It is signed like d and defined only for
labelled particles. Diffraction theory predicts that apparent size / actual
size is a function of Zd alone, so the three bead sizes should lie on one
curve against it.

**Pixel size.** Fit size & focus model ▸ Pixel size: *As processed* (default)
uses the sizes as they are, and the fitted scale s absorbs any pixel-size error.
*Set to…* rescales every measured size to a pixel size you know (for example
from a graticule). *Fit from the bead sizes* finds the factor that makes the
beads' sizes right; it needs two or more true sizes, so that the scale and the
edge offset can be told apart. With either, s is fixed at 1, so the pixel size
alone carries the scale. The model records it (`length_factor`, `pixel_mode`),
retrieval applies the same factor, and the report gives the pixel size used.
On the command line: `cpi-fit --pixel-um 2.25` or `cpi-fit --fit-pixel`. A
fitted pixel size is an effective one: it also absorbs any proportional bias
of the edge rule, so check it against a graticule.

**Size measure.** Fit size & focus model ▸ Size measure chooses what is
calibrated: the length (maximum Feret) or the equivalent-area diameter (needs
the transmittance edge method). The saved model remembers it, and
`cpi-retrieve` reads the same measure.

### Particle shape (Park et al. 2026) and errors by particle type

**Particle shape terms** in the fit dialog (`cpi-fit --shape`) add the
effect of shape, with Ar = measured roundness (projected area ÷ circle of the
same length):

- **Size:** the defocus variable becomes Zd / Ar^α. With α = 1 this is Park et
  al.'s Z/(L²·Ar); here α is fitted.
- **Focus:** the focus peak is multiplied by Ar^c.

This needs particles of different shapes (beads and ice analogues) and is
skipped if roundness hardly varies. Retrieval with a saved model uses each
particle's roundness.

On 21/3/02 beads plus col166 (Ar ≈ 0.36), α came out at 0 and the
cross-validated error did not improve (7.0% → 7.3%). col166 is large, so its Zd
is small. Small non-round analogues (Park's cross 20, rosette 41, bullet 45 and
column 50 µm) are the real test: label them before deciding.

For ice, calibrate on **Length (maximum Feret)**, Park's L. Equivalent-area
diameter ÷ length depends on habit (0.56 for col166).

The validation report also lists the error **by particle type** (label), with
each type's median roundness, so a habit-dependent bias is not hidden in the
average.

The fit window now gives the figure most of the space. Settings are in a
column on the left, figure and report are in tabs, the figure has a zoom/pan
toolbar, and it redraws to fit when the window is resized.

### Validation is out-of-sample

The papers validate on the particles they were fitted to. Here the fit is
repeated leaving out one **batch** at a time, and only the held-out batch is
retrieved. The report gives:

* mean |ΔL|/L, bias and spread for uncorrected, side-known and side-unknown;
* whether the 68 % interval really contains the true size about 68 % of the
  time;
* retrieved *d* against measured *d*;
* **sizing error against the processing focus threshold**, with the
  `foc_crit` that keeps the mean error ≤ 5 % and the depth of field at that
  threshold.

Near the edge of the depth of field the focus is close to its floor and stops
constraining *d*. A smaller particle further out then explains the image as
well as a larger one nearer, so particles there size poorly whatever the
method. The table shows where that starts.

### Depth of field

The depth of field is evaluated at the same `foc_crit` that
`proc/batchProcess.py` uses (default 5), as a function of local illumination:

* **total:** where the mean focus curve crosses the threshold on each side;
* **acceptance-weighted:** ∫ P(foc > foc_crit | d) dd. Scatter lets some
  particles in from beyond the crossing and loses some inside it, and this is
  the width the sample volume really needs.

"≥" in the report means the scan did not reach the threshold on that side.
Scan further out on that side to measure it.

`proc/calcTimeseriesDriver.py` currently uses a fixed 3 mm depth for every
size (`sv = sa·√2·3e-3`). The fitted DOF, or the DOF at a stricter
`foc_crit`, is what should replace it.

### Using it

1. Accept and label the beads; make sure every batch has its own X/Z
   (section 11).
2. **Calibration ▸ Fit size & focus model…** Untick labels you don't want,
   then set **Min local illumination** (e.g. 90 — focus is unreliable below
   it), `foc_crit` and the edge offset. Press **Fit & validate**. With ~2,000
   beads, expect up to about 10 s.
3. Check the six panels:
   * L_app/L against d with the model;
   * focus against d with the DOF;
   * DOF against illumination;
   * held-out sizing errors;
   * distance retrieval;
   * pixel size from tracking.
4. **Save model (JSON)…** writes `<project>_sizing_model.json`.

**Fitting several runs together.** Keep **one project per calibration
run**: one session on the rig, with its own MAT files, stage CSV, batches and
Xbest/Zbest. Label and Save each run in its own project, then use
**Calibration ▸ Fit size & focus model — from several projects (runs)…**
and pick the runs.

* The fit reads each project's **saved** particle table at the moment you
  press it. There are no copies, so a label or X/Z you fix in a run is in the
  next fit.
* It **refuses** projects from different CPIs, or a project that has never
  been saved.
* It **warns** about tables older than their project (open and Save it), tables
  from an older version, and merged copies (which would double-count).
* The saved model lists **which projects it came from**, how many particles
  each gave and when each was saved, so a model always says what went into
  it.
* Batches are tagged with their project, so the held-out-batch validation
  never mixes runs.

**Calibration ▸ Fit size & focus model — this project…** saves the open
project and fits it the same way.

Command line equivalents:

```bash
cpi-fit run_0302.json run_0309.json -o sizing_model.json --figure fit.png \
        --label "100 micron glass" --illum-min 90          # projects (recommended), or one particle-table CSV
cpi-retrieve sizing_model.json beads_particle_positions.csv -o check.csv --side +   # test, side known
cpi-retrieve sizing_model.json /path/to/processed/flight -o flight_sizes.csv      # cloud, side unknown
```

`cpi-retrieve` accepts a CSV (columns `length_um`, `focus`, `local_illum`) or
processed `.mat` files directly. It adds `retr_L_median`, `retr_L_p16`,
`retr_L_p84`, `retr_abs_d_median`, `retr_d_median`, `retr_p_plus`,
`retr_extrapolated` and `retr_in_dof`.

The model knows only the sizes and distances it was calibrated on.
`retr_extrapolated` marks retrieved sizes outside 0.7–1.4× the calibrated
range, or |d| at the edge of the scanned range.

`<project>_particle_positions.csv` now has a `batch` column; the
leave-one-batch-out validation uses it.

## 13. Filters

The **Filters** tab hides particles from every view without changing them:

- minimum and maximum length, width, focus, image x/y, and measured/true
  length ratio;
- an optional time window;
- image type: *Cropped ROIs only* (default), *All images*, or *Full frames
  only*.

**Show data ranges** shows the minimum and maximum of the loaded data as
hints. Press **Apply** or Enter to apply.

---

### Saved time windows (e.g. one bead size per run)

When one dataset contains several runs, e.g. 15, 40 and 100 µm beads one
after another, save each run as a named **time window** in **Filters ▸ Saved
time windows**. Windows are stored in the project.

There are three ways to create one:

* **Suggest from gaps…** makes one window per run, split wherever
  acquisition paused for longer than the gap you give (default 5 min).
* **From selection…** spans the selected particles' times. Box a run on a
  plot with X = Acquisition time, then press it.
* **From time filter…** uses the start and end typed above.

Then:

* **Edit…** gives a window its name ("15 µm beads") and optionally a
  **label**. Showing that window also sets *Accept as* to that label, so the
  beads accepted there get the right true size.
* **Show** (or double-click), or the **Window** box above the plots, filters
  everything to that run. **All times** removes the filter.
* Windows that overlap are pointed out, and each line shows how many
  particles fall in it.

## 14. Files the workbench writes

### Which CPI was calibrated

**Review → Calibrated instrument** records the instrument:

- **CPI type** is detected from the particle image-type codes (89 = CPI-3V,
  33857 = CPI 1.0), or you can choose it yourself. You are warned if the loaded
  files come from different instruments, or don't match your choice.
- **Name / serial** is free text, e.g. *FAAM CPI-3V s/n 12*.

Both are saved in the project, shown in the window title, and written as
`cpi_type` and `cpi_id` on every row of the particle table.

### Exporting several projects to one CSV

For fitting, use **Calibration ▸ Fit … from several projects** (section 12),
which reads the projects directly. To get one table for your own analysis,
use **File → Export several projects' particles to one CSV…**. It stacks the
`_particle_positions.csv` tables of the projects you choose into one CSV. The
leading columns are `project`, `cpi_type` and `cpi_id`, followed by the usual
columns (label, true and measured size, focus, illumination, stage X/Z, batch
Xbest/Zbest, *d* …). Labels come from the shared library, so the same bead
type has the same name and size in every project. Projects themselves are not
changed. A project that has never been saved has no table yet and is listed,
not skipped silently. From a terminal:

```bash
cpi-combine combined.csv beads_2021.json ice_analogues_2002.json
```


### Reprocessing is safe

You can reprocess the raw data with improved criteria and keep every flag,
label, batch and best-focus value. Particles are identified by **ROI time (to the
millisecond) and camera position (StartX+EndX, StartY+EndY)**, which come
straight from the raw data. ROI numbers, lengths and focus can all change.
When you open the project (or load the folder) against the reprocessed MAT
files:

- flags and labels are re-attached by time + position;
- flag-history batches (with their X/Z) and the selection are remapped the
  same way;
- you get a summary such as *"630 of 630 saved particles matched; 630
  re-identified by time + camera position (ROI numbering changed)"*;
- new ROIs found by the new processing start unreviewed;
- particles that no longer exist are reported as *not in the loaded files*.

The new length/focus values are used in the plots. Flags made with earlier
versions of the workbench match the same way, because every flags CSV stores
each particle's time and ROI centre.


| File | Contents | When written |
|---|---|---|
| `<project>.json` | Source file list, stage CSV, flag-history batches with their X/Z, the selection, plot settings, θ, a copy of the labels used, time windows, and `particle_keys` (time + position of every particle in a batch). Format 4 stores particle IDs grouped by file, so it is about a tenth of the old size. Older projects still open. | About 1.5 s after the last change (in the background), and on **Save project**, close or switching project |
| `<project>_flags.csv` | One row per particle: `status` (undecided/keep/reject), `reject_reason`, `label`, true sizes, and what is needed to re-find the particle after reprocessing (ROI time, `start_x`/`end_x`/`start_y`/`end_y`, camera position). Measurements are not repeated here; they are in the MAT files and the particle-positions table. | About 1.5 s after the last flag or label change (in the background), and on save/close |
| `calibration_selection.csv` | Same as above, used when **no project** is open. It sits next to the first data file and is reloaded automatically next time. | Immediately |
| `<project>_particle_positions.csv` | One row per **accepted** particle: CPI type and name, label and true size, measured size and focus, stage X/Z, the X/Z it uses and where from (own / step), X−Xbest, Z−Zbest, `d_signed_plane`, `d_in_plane`, processing | On **Save project**, close, and before any fit or export (not after every click: it is large) |
| `<project>_relative_positions.csv` | One row per (batch with its own X/Z, particle): stage X/Z, Xbest/Zbest, `x_minus_xbest`, `z_minus_zbest`, `z_p_radial`, `z_p_from_offsets`, `plane_normal_deg`, `d_signed_plane`, `d_in_plane` | Same as above |
| `CPI_cals/ReferenceLibrary/labels.json` | Shared calibration + ice-analogue labels | When labels are created or edited |

The status bar shows **● saving…** while a save is pending, then **✓ Saved
hh:mm:ss**, or a red warning if saving failed (the save is then retried).
Saving is batched: several quick clicks give one write, done on a background
thread so the window stays usable. Closing the window, opening or creating
another project, and **Save project** always write everything first.

**Rejects are plain flags, not batches.** Ctrl+Z undoes a reject, and
**Select** next to *Rejected*, then **Unflag**, undoes rejects later. Reject
batches from older projects are hidden from Flag history; **View ▸ Show
reject batches** shows them. **File → Save flags as…** writes the flags to
another CSV.

When you close the window, it offers to save a project if you have
flag history or best-focus values that are not yet in one. Flags are already
safe either way.

---

## 15. Keyboard and mouse reference

| Key | Tab ① plots | Gallery | Particle tab |
|---|---|---|---|
| **K** | Accept selection | Accept picked | Accept particle |
| **R** | Reject selection | Reject picked | Reject particle |
| **U** | Unflag selection | Unflag picked | Unflag particle |
| **A** | Add current ☆ to selection | Add picked to selection | Add to selection |
| **Del / Backspace** | Remove current ☆ from selection | Remove picked from selection | Remove from selection |
| **Esc** | Cancel the box being drawn; otherwise clear selection | Clear picks (then selection) | Clear selection |
| **Enter** | Open current ☆ | Open picked | – |
| **← / →** | Previous / next particle | (move between thumbnails) | Previous / next particle |
| **B** | – | – | Toggle boundary overlay |
| **S** | Plots: only selection / current batch ↔ all particles | same | same |
| **L** | Change label of the selection… | same | same |
| **Ctrl+Z** | Undo | Undo | Undo |
| **Ctrl+Shift+Z** / **Ctrl+Y** | Redo | Redo | Redo |
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

## 16. Tips and troubleshooting

**A project opens with no particles.** Its MAT files are missing, or their
processing hasn't finished: files from the first processing step contain
`ROI_N` but no `dat`. Since v14.17 the workbench says which files are affected
and why. While nothing is loaded it leaves the saved flags file and the
project's file list untouched. Finish the processing (for example
`batchProcess.py <folder> --only-stats --backgrounds all ...`), wait until
every file is done, then open the project again.


- **A box skipped some particles.** In a box that mixes unreviewed and
  flagged particles, only unreviewed ones are taken (status bar: *skipped N
  accepted*). Box just the flagged ones to take them, or set *Boxes pick* to
  *Everything*.
- **Nothing happens when I drag on the camera plot.** Check the *Box* mode.
  It stays on what you last chose (e.g. *− Remove* or *✕ Reject* from the
  focus plot); click **▭ New**.
- **The focus/size plots say "No stage X/Z positions yet".** Import the
  Stage CSV. If none match, check the series (beads/analogue), time basis
  and clock conventions.
- **"No best-focus X/Z reference".** Relative axes (*d*, X − Xbest, …) need a
  draft or saved Xbest/Zbest. Select particles, or activate a set with a
  reference.
- **Plots look empty after filtering.** Check the **Filters** tab, the
  **Show** toggles, and *Only selection / current batch*.
- **Large files.** Each MAT file can be ~200 MB. The workbench holds one file
  in memory at a time and caches thumbnails. Hovering on a particle from
  another file loads that file in the background, which takes a moment.
- **Zoom got stuck.** Press **⤢ Fit view**.
- **Lost a selection.** **Ctrl+Z** restores it.

---

## 17. Code layout

| File | Purpose |
|---|---|
| `gui.py` | The PySide6 workbench window (`cpi-calibrate`). |
| `data.py` | Reading MAT files (`ROI_N`, `dat`), the particle catalog, the flags CSV, stage-position logs. |
| `selection_state.py` | Box/selection operations, pick scopes and review batches. |
| `labels.py` | The shared label library and applying true sizes to particles. |
| `combine.py` | Combine several projects' particle tables (`cpi-combine`). |
| `sizing.py` | Size/focus calibration fit, Bayesian size & distance retrieval, depth of field, pixel size from bead tracking, leave-one-batch-out validation (`cpi-fit`, `cpi-retrieve`). |
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

---

## Edge Lab: compare algorithms across multiple batches without saving any results

1. In **Review → Flag history**, click a batch, or **Ctrl/Shift-click several**,
   then **Test edges →**. The batch selection is the source for the **next** trial;
   it does not remove any trials already on the graph.
2. Choose a relative or fixed transmittance threshold, radial/steepest-gradient/
   raw-contour outline, threshold fraction, median passes and Gaussian smoothing.
   **Add trial to plot** snapshots all selected batch IDs and these settings and runs
   the real detector on a worker thread. You can change settings and **Add trial**
   again: every successful trial remains an independent, colour-coded curve. ROIs
   shared between batches are analysed once per trial.
3. The **Trials** list lets you select a dataset to inspect, untick it to hide its
   curve, or **Remove selected**. Show/hide all is also available. The graph
   overlays original measurements in grey and experimental points and binned
   median traces in separate trial colours. Change X axis, size and focus metrics
   without rerunning the detector.
4. **Hover directly over a point** in either graph to preview that exact trial's
   ROI with its cyan boundary. Click the curve to make that trial active.
   **Right-click** to see the ROI and test boundary in the context menu, inspect
   the larger Particle viewer (stored boundary red, selected trial cyan), or hide
   or remove that trial.
5. **Production CLI flags** updates as you change the controls. Press **Copy flags**
   for the currently displayed input settings or **Copy selected flags** for the
   original options of a stored trial. Example for a relative-radial 0.25 trial:

   ```bash
   --edge-method transmittance --edge-mode relative --outline radial --edge-fraction 0.25 --denoise 1 --smooth-sigma 0
   ```

   These are arguments to `python python/proc/batchProcess.py /path/to/data`;
   they are not executed by the workbench. Add `--only-stats` **only for a COPY**
   of a processed MAT directory: production processing overwrites MAT statistics.
6. **CPIview focus** is selectable per trial. Turn on **CPIview focus** to compute
   the independent CPIview 2 implementation (12 difference-image median passes,
   16 boundary-normal gradient profiles; focus is peak/FWHM-based). Select
   **CPIview 2 focus** or **CPIview edge FWHM** as the lower graph's metric.
   The **ⓘ** button explains the contrast with the ordinary raw-difference
   edge gradient and normalised transmittance-gradient metrics. CPIview is slower;
   it is only available for trials calculated with that checkbox ticked, which
   adds `--cpiview` to the copied command-line arguments.
7. On the **Gallery** tab, tick **Stored boundary · red** and/or
   **Active trial boundary · cyan**. The caption menu shows focus/length,
   signed object-plane distance, stage X/Z, stored/tested diameter, or a
   combination; full details are available in each thumbnail's tooltip. The
   gallery keeps thumbnail/status selection controls separate from the read-only
   experimental overlays.

**Safety and reproducibility:** Edge Lab reads stored ROIs and backgrounds from
processed MAT files, never writes them, and never alters the project's flags,
labels, batches or calibration. Trial results are held only in memory and are
cleared on project switch or close. The worker reads each MAT file once per trial
and processes experiments sequentially to limit resource use. Gallery boundaries
are fetched in MAT-file order to avoid excessive file reloads.

**Important parity caveat:** the temporary preview uses the backgrounds already
in the MAT files and disables automatic replacement with the external background
bank. Full production `batchProcess.py` may substitute a different background.
Processing CLI flags reproduce the displayed edge-processing settings but do not
force identical background selection or a per-ROI pixel scale in a mixed dataset.
Consequently some production outcomes may differ. Use an independent COPY for
any production reprocessing comparison.


### Edge Lab: absolute transmittance (October 2026 update)

Select **Absolute transmittance · radial outline** and set **Absolute T** (for example,
`0.80` to measure the 20% attenuation contour). This is **not** a fractional
shadow-depth threshold. The existing background-normalisation uses the median
image/background ratio at the ROI border to correct multiplicative laser-power
changes, so unobstructed pixels have median transmittance near one. This
assumes the border is clear, the frames are unsaturated, and the laser spatial
profile has not changed appreciably.

Detection is independent of the fixed sizing threshold: the algorithm first
selects a central connected shadow with at least `min_dark_pixels` pixels below
`1 - min_contrast` (defaults: 10 pixels and 15% attenuation), then traces the
requested absolute isophote. The old requirement of ten pixels below T=0.45
for fixed mode has been removed. A missing measured isophote or insufficient
contrast is reported as a failure; no arbitrary replacement threshold is applied.
The detector uses the same logic in production and Edge Lab; all Edge Lab
results remain temporary and do not alter stored MAT files.

In Edge Lab, **right-click a trial particle → Transmittance profiles / threshold…**
to inspect horizontal and vertical centre profiles, the chosen isophote, and
the observed ROI border median. These are diagnostic slices, not the complete
set of rays used by the radial algorithm.


## Batch NetCDF export (raw ROIs and backgrounds)

In **Review → Flag history**, click a saved batch (Ctrl/Shift-click to combine
several), then choose **Export selected batch(es) to NetCDF…**. Choose a `.nc`
filename. A progress dialog lets you cancel without overwriting your previous
output. Export is read-only: it **never** runs edge detection, changes pixel
intensities, reassigns backgrounds, edits particle labels, or commits anything
to the project.

The exporter accepts both MATLAB structured-array and cell-array-of-structs
layouts of `ROI_N.IMAGE` and `BG`, including the format generated by
`proc/postProcess.py`. You do **not** need to convert or reprocess MAT files
for export.

Each exported particle has its **original 2D ROI** and its **corresponding
stored 2D background**. Because each ROI can be a different size, the file
uses *contiguous ragged arrays*:

- `roi_pixels` and `background_pixels` contain raw intensities in pixel order.
- `pixel_start[i]`, `pixel_count[i]`, `roi_nx[i]`, `roi_ny[i]` describe how to
  reconstruct image `i` using the original MATLAB array orientation [camera X,
  camera Y], without rotation, cropping, resizing or division.
- `background_present[i]` is 0 when the stored MAT background is missing or
  has the wrong shape; the corresponding background intensities are `-1`.
- `source_file`, `roi_index`, `particle_id`, `time_matlab`, `label`, and
  `batch_memberships_json` preserve provenance, including shared particles
  when several batches are exported together (no duplicates).
- `label_length_um` and `label_width_um` come from the **current label
  library** (NaN if unlabelled). `measured_length_um` and `measured_width_um`
  are the processed measurements and remain separate from the true sizes.
- `stage_x_mm`, `stage_z_mm`, `x_best_mm`, `z_best_mm`,
  `distance_from_object_plane_mm` (signed), `in_plane_offset_mm` and `zd`
  use saved best-focus references, **never the unsaved editor preview**.
  Missing or conflicting references are recorded as NaN with `reference_source`
  explaining why. The applied plane-normal angle is saved in the file.

Example of reading a selected particle in Python (with `netCDF4` installed):

```python
import numpy as np
from netCDF4 import Dataset

with Dataset("batch_rois.nc") as nc:
    i = 0
    start = int(nc["pixel_start"][i])
    n = int(nc["pixel_count"][i])
    shape = (int(nc["roi_nx"][i]), int(nc["roi_ny"][i]))
    roi = np.asarray(nc["roi_pixels"][start:start+n]).reshape(shape)
    bg = np.asarray(nc["background_pixels"][start:start+n]).reshape(shape)
    length_true_um = float(nc["label_length_um"][i])
    stage_x_mm = float(nc["stage_x_mm"][i])
    d_mm = float(nc["distance_from_object_plane_mm"][i])
```

Exports use compressed NetCDF4 when `netCDF4` is installed
(`python -m pip install netCDF4`, or install the repository's `.[output]`
extra). Otherwise, small exports use uncompressed NetCDF3 with the same
variable layout; for large batches you will be asked to install `netCDF4`.
Both preserve original pixel values as 16-bit signed integers, adequate for
normal 8-bit CPI camera intensities.

### Edge Lab: stored reference and signed-shadow fitted circle (October 2026)

Select one or more batches under **Review → Flag history**, then choose **Test edges →**.
The workbench automatically adds a **Stored** reference with the measurements
already in the particle catalogue. Stored reference plots do **not** recompute
any particle boundaries. Its on-hover and right-click overlays use the
previously saved `dat.foc[*].boundaries` directly and use the curve's colour.
Select **＋ Stored (instant)** to add another snapshot of the currently selected
batch(es). All curves (including Stored) can be hidden, selected or removed.

To evaluate the experimental size proxy, choose **Signed shadow integral · fitted
circle (experimental)** in the Boundary menu, choose **Extra radius [px]**
(initially **8**), and click **＋ Add trial to plot**. It evaluates each original
ROI and matching stored background without writing either. At each ROI:

1. Retrieve the saved particle outline from `dat.foc`; fit a robust circle to
   its points. This is only a seed for the integration region, not a new edge
   measurement or an estimate of physical diameter.
2. Increase the fitted radius by the selected extra pixels. If the circle extends
   outside the ROI, treat missing pixels as clear **normalised transmittance T=1**
   (zero signed deficit), retaining the fraction of the circle observed for quality
   control instead of rejecting the entire particle.
3. Calculate `T = ROI / BG` with the existing border-median multiplicative
   illumination normalisation and selected median/Gaussian filters.
4. Sum the **signed** deficit `A = pixel_um² × sum_circle(1 − T)` using 4×4
   subpixel aperture coverage. Light above the corrected baseline contributes
   negatively; no clipping of the bright diffraction rings is performed.
5. Report **uncalibrated shadow-equivalent diameter**
   `D_shadow = 2 × sqrt(A / pi)` and plot `D_shadow / labelled_true_length`
   when Size is set to **Diameter / labelled true size**. The bottom Focus panel
   shows the pre-existing saved focus metric for a shadow trial because the
   integral itself does not calculate focus.

The default X axis is signed distance from the object plane. Each trial has its
own colour. Its displayed circle, hover preview, right-click context-menu
preview, particle inspection and gallery overlay use that **same colour**.
Unlike edge-trial outlines, the shadow circle is the **integration aperture**,
**not a detected particle edge**. The stored boundary is shown in grey/slate
when another experimental trial is active. You can compare several padding
values and traditional edge methods simultaneously.

The new method is **research-only**: it has *no* production CLI flags, and it
is not yet a calibrated physical true-diameter retrieval. In particular, its
accuracy is affected by background residuals, camera pedestal, neighbouring
particles and the stored contour used to initialise the aperture. Different
bead materials may need different calibrations. Invalid nonpositive integrals
are excluded from the plot. Truncated fixed-radius circles are retained but
may underestimate size if the missing region contains actual shadow. All lab results remain
in memory only and are discarded on closing or clearing the lab.


## Fitting recipes: signed-shadow circle + normalised focus (October 2026)

From Calibration → **Fit size & focus model**, choose the recipe at the top:
"Original size and focus" (**default**), "Contour size + normalised focus", or
"Circle shadow size + normalised focus" (**experimental; opt in**). Select the bead labels and focus threshold, then
**Fit & validate**. The figure displays per-size ratio against signed position,
measured and fitted focus, DOF versus size, the size-error histograms and depth
retrieval. Report tab includes parameters, size-specific DOF and held-out errors.

Circle size is saved separately as `dat.shadow_deq` only when processing with
`--shadow-circle-padding N` (0–30 px). MAT files also store `shadow_area_px`,
`shadow_radius_px` and local background-corrected illumination
`local_illum_bg`. Feret length, conventional equivalent-area contour size,
roundness, boundaries and original focus are unchanged. Processing does not
know bead true size; the affine shadow size calibration is fitted later.

For **temporary on-demand** calculation without overwriting MAT files, open a
single project in the fitting dialog, choose your aperture extension and click
**Compute circle sizes for this project (temporary)**. The calculation is done
asynchronously from saved MAT images/backgrounds and contours and only updates
the dialog's in-memory rows. Multi-project fits require durable processing of
source projects beforehand.

Normalised focus is the *original gradient focus* divided by the local
illumination; it is not `focus_T` and must not be divided again. Fit
`B+A(L)*exp[-(|d-d0|/w±(L))^p]`, with `A(L)=A0(L/Lref)^a` and
`w±(L)=w±0(L/Lref)^k`; a,k≥0 when fitted. Different widths on each side,
fitted or Gaussian `p`, and the two size dependencies are optional.
The DOF threshold is in **focus/local illumination** units, not the original
raw CPI focus units. Default 0.12 is a suggested initial plotting value only.

`cpi-fit --fit-recipe circle` implements the same model for saved particle
tables, with `--fixed-gaussian`, `--symmetric-focus`, `--no-peak-size`,
`--no-width-size`, and `--foc-crit`. Old models and original fitting remain
readable and usable. The retrieval code picks the same saved shadow/contour
column and focus definition that were used to fit the model.


### Production processing defaults from 23 October 2026

The new default in `python/proc/batchProcess.py` is relative radial transmittance
edge detection at fraction **0.55**, plus independently stored signed-shadow
equivalent diameter from a fitted circle enlarged by **8 pixels**, and
`dat.local_illum_bg`. Existing contour morphology and raw gradient focus
are retained. The fitting dialog now starts on circle size + original gradient
focus / local illumination, including for older projects without circle data
(use its read-only temporary calculation button). `--edge-method legacy`,
`--shadow-circle-padding N` and `--no-shadow-circle` override the defaults.

A +8-pixel margin does **not** imply the signed-integral curve has reached a
plateau. Masking all objects and diffraction rings to estimate the correction
from unobstructed pixels is a separate prospective illumination algorithm,
not implemented or activated by these defaults.

#### Experimental adaptive circle (2026-10-23)

Edge Lab > Method: Signed shadow integral; Circle radius: **Automatic plateau**.
The adjustable value is then **maximum additional radius**, not a fixed +px.
The trial searches for a stable cumulative signed deficit and displays the
selected circle in the trial colour. Right-click a particle > **Cumulative
shadow integral vs circle radius…** to inspect the radial integral and the
observed ROI coverage. Trials are read-only. Fitting still uses the existing
saved circle values unless you recompute them using the existing (fixed-radius)
fitting control; automatic-radius fitting should only use measurements actually
saved with the experimental processing flag.

The fixed +8 px option remains the production default. It now permits circles
extending outside the ROI, with the unseen portion defined to have T=1 and
therefore zero signed deficit; coverage is recorded. Current 2002 bead ROIs
can be too small for trustworthy automatic convergence. The adaptive method
will not assert a plateau if any required annulus has less than 85% coverage.
No extrapolation of unseen shadow or fringes is performed.
On reprocessing the processor now saves any substituted background in sparse
`BG_USED`. Edge Lab loads these corrected backgrounds automatically; original
`BG` is preserved for audit. Files processed with earlier releases do not
contain this history and require reprocessing to gain it.

### Microscopy / SEM references on particle labels

Open **Labels**, click a label, and choose **Microscope / SEM reference images…**
(or right-click its label). The reference-image editor is independent of a
calibration batch and supports multiple imported optical / SEM images for each
shared bead or ice-analogue label.

* **Import images…** supports PNG, JPEG, TIFF (including 16-bit images), BMP and
  WebP. The original file is copied unchanged to a `microscopy/images/` folder
  beside the shared label library (`CPI_cals/ReferenceLibrary/labels.json`, or
  the explicitly selected location). Annotations are saved separately in
  `microscopy/index.json`. Keep this directory with your label library when
  backing up or moving to another computer. Deleting an image from the list
  unlinks its annotation but does not delete the copied source.
* **Find object boundaries** proposes automatic contours, drawn in light blue.
  Click the wanted outline; if it misses the target, choose **Draw circular
  boundary** and drag, or **Draw straight-line polygon** and click successive
  points, then **Finish / join polygon** (or double-click). The **Edit** tool
  drags polygon vertices and moves or resizes the circular boundary.
* **Fit enclosing circle around boundary** creates the *minimum enclosing
  circle*, which contains the complete selected/drawn outline. The object
  silhouette (green) and enclosing circle (orange dashed) remain separate.
* **Draw scale reference**: drag over a scale bar in the microscope image and
  enter its real length in micrometres. The calibration is stored per-image;
  until it is set, measurements are in **pixels**, not invented µm values.
* The panel reports **maximum dimension** (maximum Feret dimension),
  **silhouette area**, **enclosing-circle area** and **silhouette area / circle
  area** (all derived from the chosen image and scale). The annotation index
  retains the measurements, vertices, fitted circle and scale; edits are
  saved as you go.
* **Save annotated image as PNG…** exports a fresh full-resolution PNG with
  the label at the top and the boundary, enclosing circle, scale and measured
  quantities drawn onto it. The original microscopy image is never overwritten.
  The editable annotation remains in `microscopy/index.json`.

This is *reference-microscopy annotation*, not the processing of raw CPI ROIs.
Measurements do not overwrite the CPI bead label's nominal true length, the
calibration MAT data or the existing calibration fit. The enclosing-circle diameter is always at least as large as the measured
maximum Feret dimension of the chosen boundary.


### Shared reference library and visible outlines (October 2026)

Projects anywhere within the same `CPI_cals/` tree automatically share
`CPI_cals/ReferenceLibrary/labels.json` and
`CPI_cals/ReferenceLibrary/microscopy/` (including both `CPICalibration/`
and `ICEAnalogues/CPI/`). The **Labels** page shows the active location and
has controls to select another reference folder or restore automatic discovery.
The chosen location is remembered using Qt settings; the label content and
microscopy images are never stored in Qt settings. When moving a legacy library
to an empty new location the GUI requests permission and copies the labels,
original microscopy images and annotations without deleting the old copy.
The environment variable `CPI_CALIBRATION_LABELS` still pins an explicit path.

Particle contours use vivid, curve-matched colours and a contrasting dark halo
in the particle viewer, gallery, Edge Lab hover and microscope editor (including
annotated PNG exports). Edge Lab shadow integration is a **size-only** trial:
raw `focus` and `focus_T` remain those of the original processed particles for
**all** trials, including particles without a valid shadow integral. CPIview
focus is an optional separate experimental metric. MAT files are unchanged.

## Shared graticule microscope images

The **Labels → Shared microscope / graticule images…** button opens a multi-object editor. Each image has one shared X/Y graticule calibration and multiple independent outlines, each optionally linked to an existing label. You may export an annotated PNG or object-level CSV, or explicitly confirm copying a selected object’s measurements into the reference label. Linked annotations alone never overwrite labels.json. See `UPDATE_NOTES_2026-10-24_shared_graticule.md`.
