# Ice Crystal Classifier

A web page for looking through CPI particle images and classifying them by hand.
People can browse without an account. Signing in saves their classifications
under their name.

```
 MAT files on the server ──► Admin page: choose a folder, fill in the flight details, press Import
 (python/proc output)                │  runs exporter/cpi_export.py in the background (~5 s, ~190 MB per MAT file)
                                     ▼
                         web/data/  small PNG per particle + one table of particle properties
                                     │  served as plain files
                                     ▼
                         browser: filtering, sorting, sampling and display all happen here
                                     │  labels only
                                     ▼
                         web/api/ (PHP) ──► MySQL / MariaDB ──► download as CSV / JSON
```

The MAT files stay where you put them and are only read. Importing a flight
reads each MAT file **once** and writes what the browser needs: one small PNG per
particle, a compact table of its properties (sizes, shape, focus, time,
position, temperature) and the outlines. After that, viewing and classifying
only fetches those small files, so the server does almost no work. Browsers
can't read MAT files, and a 385 MB file per page view would be far too slow
anyway. Classifications go into MySQL as people make them.

## What's here

| Path | What it is |
|---|---|
| `exporter/cpi_export.py` | The importer. The Admin page runs it for you; you can also run it from the command line. Needs numpy and scipy (and netCDF4 for netCDF nav files). |
| `exporter/flight.example.json`, `ground.example.json` | Flight / ground-session descriptions to copy. |
| `web/` | The website. Put this folder (or its contents) in the web root. |
| `web/config/scheme.json` | The classes, flags and keyboard keys. Edit this to change the scheme. |
| `web/api/` | PHP API: sign-in, saving labels, stats, CSV export (`index.php`), and importing flights (`admin.php`). Creates its own tables. |
| `web/api/schema/mysql.sql` | Table definitions, if you'd rather create them yourself. |
| `deploy/DEPLOY.md`, `deploy/apache-cpi-classifier.conf` | Step-by-step server set-up (Ubuntu, Apache, MariaDB), and the Apache settings. Runs on PHP 7.4 and 8.x. |

## 1. Set up the server (Apache + PHP 7.4 or 8 + MySQL/MariaDB + Python 3.8+)

**Follow `deploy/DEPLOY.md`**, written for an Ubuntu 20.04 server with Apache,
MariaDB and an SSH tunnel until the firewall is opened. It covers packages,
database, folders and permissions, the importer's Python, config.php, Apache, back-ups
and HTTPS. `deploy/apache-cpi-classifier.conf` holds the Apache settings: the site
is served at `/cpi/` and the data from `/var/www/cpi/data` at `/cpi/data/`.

Once it's up, open the site and press *Set up the administrator account*. The
username must be one of `admins` in config.php. Or, from the server's terminal:

```bash
php web/api/tools/user.php add pconnolly "Paul Connolly" --admin
```

This prints a temporary password, and you choose your own at first sign-in. Then
the top of **Admin → Flights and imports** checks the set-up and lists anything
to fix.

PHP (7.4 or 8.x) needs `pdo_mysql`, `mbstring` and `shell_exec` (`gd` is optional, for shrinking project photos). Some hosts disable `shell_exec`
in php.ini; the Admin page tells you if so. There's no Node, no build step and no
PHP framework.

## 2. Add flights

1. Copy a flight's MAT files into a folder per project and flight:

   ```
   /mnt/CPI-data/PICASSO/C123/        11071159.mat  11071230.mat  core_faam_c123.nc
   /mnt/CPI-data/PICASSO/C124/        ...
   /mnt/CPI-data/JFJ-2026/2026-01-20/ ...   (ground site: one folder per day or session)
   ```

   A flight can have many MAT files. Put its navigation or met file (CSV, or
   netCDF such as a FAAM core file) in the same folder.
2. On the **Admin** page, folders are listed by project. Press **Import…**. The
   form fills in the project and flight id from the folder names: `C123` in
   project `PICASSO`. If that id is already taken it becomes `PICASSO-C123`.
   Check the form:
   * **What it is:** flight id, name, project, aircraft or ground site, platform, date and notes.
   * **Instrument:** CPI version, serial and pixel size. Leave pixel size blank to
     use the value the processing recorded.
   * **MAT files:** which files to include.
   * **Position and temperature:** choose the navigation file. The page reads its
     columns and suggests time, latitude, longitude, altitude and temperature.
     Check the units (K or °C, m or ft). Ground sites can give a fixed position.
   * **Which particles:** measured particles only, a size range, and whether to
     leave out ones cut off at the edge.
   * **Images:** colour map (grey recommended) and orientation.
3. Press **Import**. A progress bar shows each MAT file being read and its
   images being made. When it finishes, the flight is on the classify page.
   Imports run one at a time; others wait in a queue.

Afterwards, *Change settings* edits a flight and imports it again. *Shown*
switches a flight on or off for people classifying. *Remove* deletes its imported
images. In both cases everyone's classifications stay in the database, and they
reattach if you import the flight again under the same id (labels refer to flight
id, MAT file name and index). The flight id can't be changed after the first
import for this reason.

The form saves its settings as `work_dir/flights/<id>.json`, in the flight.json
format below, so you can also run the same import by hand.

### Importing your CPI pipeline's PNG export

If you've already run the pipeline's PNG export (`batchProcess.py --export-png`
or `exportParticles.py`), the classifier uses that output directly. A project
folder like

```
/mnt/CPI-raw/ACTIVE/
    01201447.mat  01201447.roi  ...          MAT files (statistics, outlines)
    particle_catalogue.csv                    written by exportParticles
    flights.csv
    2005-11-16_AE04/ACTIVE_11161351_roi0000002_p01.png ...
    2005-11-09/ACTIVE_11091358_roi0000001_p01.png ...
```

appears on the Admin page under *Exports from your CPI pipeline*. Press
**Import project…**, tick the flights to include and press **Import**:

* **Flights:** each flight in the catalogue becomes its own entry, with id
  `<project>-<flight>` (e.g. `ACTIVE-AE04`). Particles with no flight are grouped
  by date (`ACTIVE-2005-11-09`). *Only flights* leaves those out.
* **Images:** your PNGs are used as they are, background-corrected by the
  pipeline. Hard links put them into the web data folder without using extra
  space when both are on the same disk; otherwise they're copied.
* **Statistics:** sizes, shape, focus, time and outlines are read from each MAT
  file's `dat`. Only `dat` is loaded, so this is quick. Outlines and all the
  information overlays then work as for any other flight.
* **Ids:** each particle's pipeline id (`ACTIVE_11161351_roi0000002_p01`) is shown
  in the details panel and included in downloaded CSVs, so labels can be joined
  back to your ML pipeline.
* **Updates:** re-run the pipeline export (e.g. with new thresholds), then press
  *Import again*.

Set `raw_root` in `config.php` to the folder that holds these projects, e.g.
`/mnt/CPI-raw`. The web server needs read access to the MAT files, the
catalogue and the PNGs:

```bash
sudo chgrp -R www-data /mnt/CPI-raw/ACTIVE && sudo chmod -R g+rX /mnt/CPI-raw/ACTIVE
```

The pipeline writes Blues_r colour PNGs by default. Those keep their colours on
the classify page, and the live colour-map picker is switched off for them. With
the optional patch `png-colormap-option.patch` applied to your cpi3v code, add
`--png-colormap gray` to write 8-bit grey PNGs instead. They are smaller, and
people can then choose any colour map live.

### From the command line (alternative)

```bash
python exporter/cpi_export.py /var/www/cpi/work/flights/C123.json --out /var/www/cpi/cpi3v/web/data
python exporter/cpi_export.py my_flight.json --out web/data --limit 200   # quick test
```

You can also run this on your laptop and copy `web/data` to the server. Every
run rebuilds `data/manifest.json`, the list of flights the page shows. Images
that already exist are skipped (`--force` rewrites them).

### flight.json (reference)

| Key | Meaning |
|---|---|
| `id` | Short unique id, letters/digits/`._-` (e.g. `C123`, `JFJ-2026-01-20`). Forms part of every particle id, so don't change it after people start classifying. |
| `label`, `campaign`, `platform`, `date`, `notes` | Shown on the Flights page. `campaign` is the project; flights are grouped by it. |
| `kind` | `aircraft`, `ground` or `lab`. |
| `instrument` | `{"name": "CPI", "version": "3V-CPI", "serial": "…"}`, free text. |
| `pixel_um` | Pixel size. Leave `null` to take it from `dat.proc_options`. |
| `mat_files` | List of files or glob patterns, absolute or relative to flight.json. One flight can have many MAT files. |
| `transpose` | `true` (default) shows the CPI x direction horizontally. The MAT arrays store x down the rows. |
| `select` | Which ROIs to export. `valid_only` (default true) skips ROIs with no measured particle. You can also set `min_size_um`, `max_size_um`, `size_field`, `exclude_edge_touch` and `image_types`. |
| `nav` | Optional CSV or netCDF time series interpolated onto each particle: `fields` maps `lat`, `lon`, `alt`, `temp` to column / variable names. `time_units` is `iso`, `unix`, `datenum` or `seconds since <date>` (netCDF reads it from the file). Also `temp_units` (`K` or `C`), `alt_units` (`m` or `ft`) and `max_gap_s`, beyond which a particle gets no value rather than an interpolated one. FAAM-style `(Time, sps)` arrays are handled. |
| `site` | For ground sites: fixed `lat`, `lon`, `alt` used when there's no nav value. |

### ROI files that span more than one flight

One MAT/ROI file can hold particles from several flights: the import puts each
particle in the flight given for it in `particle_catalogue.csv`, so a shared file
is simply split between flights. Particles your pipeline could only place in
"either flight" (e.g. `C077+C078`, when the flight list gives no times to tell
them apart) become one combined flight, labelled `C077+C078` (id `…-C077-C078`).
Give the flights start and end times in your flight list and they are split by
time instead.

### Colour maps

You can apply a colour map in two places:

* **Live on the web page (recommended).** Export in grey (the default). Each
  person then picks a colour map from **Display** in the toolbar, or presses M to
  step through the maps. The page recolours the grey images with an SVG filter,
  so it is instant and doesn't change the files. Brightness, contrast, outline
  colour and a caption under each particle are in the same panel.
* **Baked into the PNGs by the exporter.** Use `"colormap": "map2"` in
  flight.json, or `--colormap map2` (add `--force` to redo images that already
  exist). The PNGs are written as 8-bit palette images, so they are no bigger
  than grey ones. Everyone then sees that map; the page still applies
  brightness and contrast, but disables its own colour map for that flight.

The available maps are:

* Grey, Negative, Bone, Ice
* **CPI MAP2** and **CPI MAP**: the two tables from `python/proc/cmap.mat` that
  `exportImagesDriver` uses. MAP2 runs from black through blue to white and
  shows saturated pixels in yellow.
* Blues and Reds, as `exportImages` uses
* Viridis, Cividis, Magma, Inferno, Plasma, Turbo, Cubehelix

The exporter also accepts:

* any matplotlib map name, if matplotlib is installed
* `"file.mat:VAR"` for any 256×3 table in a MAT file
* a `.json` or `.csv` file of RGB rows

Add `_r` to any of these to reverse it.

```bash
python exporter/cpi_export.py --list-colormaps
python exporter/cpi_export.py flight.json --colormap-preview maps.html   # compare them on your own particles
```

The preview page shows ten of the flight's particles, smallest to largest, in
every map at the same scale. `"stretch": "per-image"` stretches each ROI's
contrast (1st–99th percentile) before the colour map is applied. Leave it
`null` to keep grey levels comparable between particles.

The maps are defined once in `exporter/colormaps.json`, and
`web/config/colormaps.json` is an identical copy. Add a map to both and it
appears in the exporter and on the page.

### What gets exported

Per particle, from `dat` (and `ROI_N`):

* **Sizes (µm):** `len`, `wid`, `deq`, `len_feret`, `wid_feret`, `len_ellipse`, `wid_ellipse`, `shadow_deq`, `area`, `perimeter`
* **Shape:** `round`, `area_ratio` (area_ratio_circle), `aspect` (len/wid), `orientation`, `interior_bright` (fraction)
* **Quality:** `focus`, `focus_T`, `T_min`, `contrast`, `edge_touch`, `cutoff_pct`
* **Time and position:** `t` (from `dat.Time`), plus `lat`, `lon`, `alt` and `temp` when available
* **Instrument:** `imageType`, `x0`, `y0` (ROI position in the frame), `w`, `h`
* **Boundary:** `foc.boundaries`, loaded only when someone turns on outlines

To add another `dat` field, add one line to `FIELDS` at the top of
`cpi_export.py`. It then appears automatically in the details panel and in the
"Add a filter on…" list.

**Particle id:** `<flight id>:<MAT file stem>:<index>`, where index is the 0-based
position in the MAT arrays. So `C123:11071159:10` is `dat.len[10]` in Python, or
`dat.len(11)` in MATLAB. The CSV export also splits the id into separate columns.

Output layout:

```
data/manifest.json
data/flights/<id>/meta.json        metadata + full proc_options
data/flights/<id>/catalog.json     all particle properties, one array per property (~180 bytes/particle, gzips ~5x)
data/flights/<id>/img/<stem>/<index>.png
data/flights/<id>/bnd/<stem>.json  outlines
```

## Try it on your laptop

```bash
cd web
cp api/config.sample.php api/config.php
# in config.php: 'dsn' => 'sqlite:/tmp/cpi-labels.sqlite', and set raw_root / work_dir to local folders
php -S 127.0.0.1:8000
# open http://127.0.0.1:8000, create the admin account, then use Admin to import a folder
```

## Who can see what

The `.htaccess` files compress the JSON, cache the images, and stop the SQL
files being served. Anyone who can reach the site can see the imported images
and data; sign-in protects only the classifications. Only admins can see the
Admin page or call its API, and every path it is given is checked to lie inside
`raw_root`. To restrict viewing too, put the site behind your university's SSO
or Apache basic auth.

## Using the page

* **Classify:** pick flights, then narrow down with histogram filters. Drag on a
  histogram to set a range and double-click it to clear. Each histogram counts
  the particles that pass the *other* filters, so you can see what you're cutting.
  * The size measure can be switched (max length, D_eq, Feret, shadow circle…).
    Aspect ratio, area ratio, focus, time and (when nav data exist) temperature
    and altitude have filters by default. Any other property can be added.
  * The "Show" menu chooses which particles: not yet classified by me, all, not
    classified by anyone, where people disagree, or my own (optionally only one
    class, for checking consistency).
  * Orders: random (reproducible until you press Shuffle); random with sizes
    mixed (round-robin over size bins, good for an unbiased training set);
    chronological; by flight; largest or smallest first; and fewest
    classifications first, which spreads the work between people.
  * **Grid** shows many particles at once: select several and press one key to
    classify them all. **One at a time** shows a large image, and the next
    particle appears as soon as you classify.
  * **True scale** draws every particle at the same µm per screen pixel, with a
    scale bar, so sizes really compare. **Fit** fills each tile.
  * Outline (O) overlays the detected boundary. Contrast (B) stretches the image.
  * Going to another page (Guide, Progress, Flights…) and back, or reloading the
    browser tab, returns you to the same particles, scrolled to where you were.
    Changing the flights, filters, order or Shuffle starts a fresh set.
  * **Drag and drop:** drag a particle, or a whole selection, onto a class button
    to put it in that class, or onto *Remove label*. In the one-at-a-time view
    you can drag the large image. Each class button shows how many particles
    you've put there; click the number to see them all and check them.
  * **Fixing mistakes:**
    * Press another class key, or drop the particle on another class, to change it.
    * Press Delete to remove your label.
    * Use the *Change* button in the details panel to change one particle from anywhere.
    * *History* lists everything you've done this session; click any particle there to change it.
  * **Undo / redo:** undo with Z (or Ctrl/⌘ Z); redo with Y, Shift Z (or
    Ctrl/⌘ Shift Z). Every action can be undone, including group labels, drops,
    flag changes and removals. A group action is undone in one step.
* **Display** (toolbar): colour map, brightness and contrast, outline on or off
  and its colour, and what to **show with each particle**:
  * size (L × W, or the chosen size measure)
  * aspect ratio and area ratio
  * temperature, altitude and time
  * flight and project
  * a scale bar on every tile, useful in Fit mode where each tile has its own scale

  In the one-at-a-time view the same information appears in a box on the image.
  The settings are remembered per browser.
* **Flights are grouped by project** in the sidebar and on the Flights page.
  Tick a project to include all of its flights.
* **Downloading images** (the download button in the toolbar, *Download* in the
  selection strip, or *Download* next to a single particle):
  * **Which particles:** the selected ones, all shown with the current filters,
    everything you've classified, or the group's majority classifications.
  * **Scale:** *Same scale for every particle* puts each image in a fixed frame,
    e.g. 400 µm square at 1.25 µm per pixel, so tightly cropped ROIs don't look
    bigger than they are and every image is comparable. Particles larger than
    the frame are cut and marked in the CSV. *As recorded* keeps each ROI's own
    crop, magnified by a whole number.
  * **Colours:** as on screen (colour map, brightness, contrast) or the original
    grey levels.
  * **Drawn on the image:** outline, scale bar and class label, plus a strip
    with any of size, aspect ratio, area ratio, temperature, altitude, time,
    flight and project. Turn them all off for clean images, e.g. for training a
    classifier.
  * **The ZIP:** a folder per class (or one folder), `particles.csv` with every
    image's id, flight, project, MAT file and index, class, votes, sizes, shape,
    focus, time, position and temperature, and a README recording the scale and
    colour settings.

  Images are drawn in the browser, so downloading doesn't load the server.
* **?** opens a list of keyboard shortcuts. The first visit shows a short card
  explaining the three ways to classify. A strip at the bottom of the grid shows
  how many particles are selected, and the palette shows how many you've
  classified this session and your rate per minute.
* **Flights:** metadata and processing options for each dataset, with a
  sparkline of particles through the flight (and altitude, when there's
  navigation data).
* **Progress:** counts per class, coverage per flight, contributors, agreement
  between people, and CSV/JSON download. Admins can download everyone's labels;
  other people download their own.
* **Guide:** the scheme, keyboard shortcuts, and example particles for each
  class: the ones people agree on most, or your own. You can resize them, show
  them at the same scale, filter by project and instrument, click one to change
  your classification, or drag it onto another class (see *Guide page* below).

Labels save automatically in small batches. If the connection drops, unsent
labels are kept in the browser and sent later. Every change also goes to an
append-only `label_log` table, so you can see when someone changed their mind.

## People and accounts

**Admin → People:**

* **Add a person:** give a name; the username is suggested. They get a
  temporary password like `kdw7-m3pq-tx9h`, shown once with *Copy password* and
  *Copy message* (a ready-to-send note with the site address, username and
  password).
* **First sign-in:** they must choose their own password (at least 10
  characters) before they can classify. Everything they classify is saved under
  their name.
* **New password:** gives someone a fresh temporary password, e.g. if they forget theirs.
* **Make admin / Remove admin:** controls who can add people, edit classes and
  import flights.
* **Switch off:** stops someone signing in. Their classifications are kept.
* **Expertise:** a drop-down per person (see *Practice and demonstration*).
* **from file / No login:** people whose classifications came from a session
  file. *Give a login* gives them a username and temporary password.

Anyone can change their own password from the menu under their name.
Self-registration is off by default. Set `allow_registration` to `true` in
config.php to allow it, optionally with a `registration_code`.

From the server's terminal:

```bash
php web/api/tools/user.php add jsmith "Jo Smith"        # prints a temporary password
php web/api/tools/user.php reset jsmith                 # new temporary password
php web/api/tools/user.php list
```

## Practice and demonstration

* **Practice mode:** anyone who isn't signed in can still classify, by keys,
  clicks or drag and drop. Their classifications stay in their own browser and
  are never written to the database. The bottom bar says *Practice mode*. If
  they sign in later, they're asked whether to keep those classifications under
  their name.
* **Saving practice work to a file:** *Save to a file* (bottom bar, Progress
  page, and a reminder every 25 unsaved classifications) asks for their name,
  level of expertise, affiliation and notes once, then downloads a session file
  `cpi-session-<name>-<date>.csv`. Nothing reaches the server. The file starts
  with `#` lines holding those details, the time span, the site and the class
  list, then one row per particle: pid, flight, MAT file, index, pipeline id,
  class, class name, flags, time classified. It opens in Excel or pandas
  (`pd.read_csv(f, comment='#')`).
* **Adding a session file to the database:** on the Progress page, signed in,
  drop the file on *Add classifications from a file*. A preview shows the counts
  per class, anything that will be skipped (unknown classes, unreadable rows)
  and the person's details, which you can correct. Then choose whose work it is:
  * **Mine:** anyone signed in, for their own practice work.
  * **Someone without an account:** administrators only. A contributor with no
    login is created; *Give a login* on Admin → People turns them into a normal
    account later, keeping their classifications.
  * **Someone already listed:** administrators only. A person with the same name
    is picked automatically.

  Where that person already classified a particle, their existing class is kept
  unless you tick the replace box. Each upload is listed under *Sessions added
  from files*. *Remove* takes out what that file added, but not anything changed
  by hand since. The site's own CSV download also reads back in, as one person.
* **Level of expertise:** set by each person under *Your details* (the menu
  under their name), by an administrator on Admin → People, or from a session
  file. The choices are `expertise_levels` in config.php. Expertise, affiliation,
  whether the person has a login, and the session id are columns in the
  *Everyone (CSV)* download, so you can compare novices with experts.
* **A demonstration copy of the site:** set `'demo' => true` in config.php. No
  database is needed and nothing is saved. Everyone gets practice mode, and
  signing in is switched off. A demo copy can use the same imported data as the
  real site: point its `data` folder at the same place (e.g. a symbolic link),
  and give it its own `api/config.php`.

## Project pages

Each project (the `campaign` of its flights) has its own page: open it from
the project cards at the top of **Flights**, *About this project*, or the ⓘ
next to the project in the Classify sidebar.

* **What it shows by itself:** the number of flights and particles, the dates,
  platforms and instruments, how many particles have been classified, and its
  flights with *Classify* buttons.
* **What an administrator can add (*Edit page*):**
  * a title and a one-line summary, which also appear on the Flights page card;
  * a description with simple formatting (blank line = new paragraph,
    `- ` list, `## ` heading, `**bold**`, `*italic*`, `[text](https://…)`),
    with a live preview;
  * links, each with a title and an optional note: the campaign website, papers,
    data archive, flight reports, `mailto:` contacts;
  * pictures (JPEG, PNG, GIF or WebP, up to 25 MB), each with a caption and a
    credit. Drag them, or use the arrows, to change the order. The first picture
    is the cover.
* **Viewing pictures:** click one to see it large. The arrow keys step through
  them.
* **Flight notes:** a short note per flight (e.g. "CPI window iced over after
  14:20"), added under *Flight notes* when editing the project page. It is shown
  on the project page, the Flights page, in the inspector, and as a marker in
  the Classify sidebar.
* **Where they're kept:** `data/projects/<project>/info.json` plus the
  pictures, next to the flight data. Large photos are shrunk to 2000 pixels
  (with a 560-pixel thumbnail) and turned upright, when PHP has the GD extension
  (`php-gd`); without it they're kept as uploaded.
* **Bigger uploads:** if they fail, raise `upload_max_filesize` and
  `post_max_size` in php.ini (e.g. 32M).

## Classes

**Admin → Classes** edits the classes people sort particles into, and the flags
they can add:

* **What you can change:** name, colour, keyboard key, description (shown as a
  tip and in the Guide) and order, with a live preview of the class buttons.
* **Ids:** each class keeps its id once it exists, so renaming is safe.
* **Removing a class:** it shows how many classifications use it. Those stay in
  the database and downloads under the old id.
* **Saving:** the edited list is kept in `work_dir/scheme.json` and used by
  everyone from their next page load. Its version is stored with every new
  classification, so labels made under different class lists can be told
  apart. *Restore the defaults* goes back to `web/config/scheme.json`.

The default list starts with **Droplet** (key D): liquid drops, near-perfect
circles with a bright centre spot. **Quasi-sphere** (key 1) is for round ice and
frozen drops. If you have already saved your own class list on the Admin page,
the new default doesn't replace it: add *Droplet* there, or use *Restore the
defaults*. The same goes for the classes added later (Scroll, the Hollow and
Chain choices, the extra *made of* options): if your Admin → Classes list was
saved before, press *Restore the defaults* to get them (classifications already
made are kept).

Keys used by the page itself can't be given to a class: Z (undo), Y (redo),
V (view), O (outline), B (contrast), M (colour map) and ?. While the *made of*
bar is showing, M means *mixed*.

## Scale, and sampling across sizes

* **True scale:** every particle at the same µm per screen pixel; the slider is the zoom.
* **Largest fills:** also the same scale for every particle, but chosen
  automatically, so the largest particle currently selected fills the space.
  In the grid, the slider sets how big that largest one is; in *One at a time*
  it fills the viewing area. Smaller particles are proportionally smaller. Change
  the filters, and the scale follows. Use a size range to keep one odd giant
  from shrinking everything else.
* **Fit:** each particle fills its own tile, so sizes can't be compared.

**Balanced by size** (the order menu under *Show*) stops a sample being mostly
small particles. Give bin edges in µm (e.g. `50, 100, 200, 400, 800`, or a quick
preset), using the size measure chosen for the filters. Particles are then taken
from each bin in turn, randomly within a bin, so each bin contributes the same
number until it runs out. The panel shows how many particles are in each bin.

## "Made of": a second classification

Some classes ask what the particle is made of:
* **Aggregate:** of bullet rosettes, plates, columns, dendrites, needles, capped columns, bullets, quasi-spheres, mixed habits, or can't tell.
* **Fragment:** of a dendrite, plate, column, capped column, rosette or aggregate, or can't tell.
* **Graupel:** what the rimed crystal was (including capped columns).

Some classes also have a **second choice of their own**, which appears beside
them (in the *made of* bar, and as a flag that only lights up for them):
* **Hollow** (key H): for columns, needles, plates, bullets, capped columns and
  rosettes, e.g. a hollow column or a hollowed (skeletal) plate. It stays a
  column or plate, so counts of columns still include hollow ones.
* **Chain** (key N): for aggregates linked end to end, e.g. *Aggregate of
  quasi-spheres (chain)*. Press N while the *Aggregate of…* bar is up, or later.
* **Scroll / sheath** (key L) is a class of its own.
* **Side planes / polycrystal** (key P): plates growing out at different angles,
  or several crystals grown (not stuck) together; common at about −20 to −40 °C.
  Aggregates, fragments and graupel can also be made of them.
* **Graupel vs Rimed:** Graupel is for crystals so rimed you can't tell what they
  were. If you can still tell, use that habit plus the Rimed flag.
* **Too small to tell:** fewer than about 15 pixels across (≈35 µm on a CPI).
* **Rimed** works on any class, so a rimed capped column is *Capped column* +
  Rimed (R).

These are stored as flags (`hollow`, `chain`) alongside the class and *made of*.
In **Admin → Classes**, any flag can be limited to some classes (*For every
class (limit it to some…)*).

How it works:
* **When you classify:** after choosing one of these classes (by key, click or drag), a bar appears:
  *Aggregate of…*. Press the habit's own key (e.g. `4` for plates, `6` for bullet rosettes),
  `M` for mixed, `X` or Enter for can't tell, or Esc to skip. The page then moves on.
* **Later:** select particles and press **Shift + the habit's key** (e.g. Shift 4),
  or use the buttons in the *Change* window.
* **What's stored:** the class stays the main label (`aggregate`), and what it's made of goes in
  its own column (`made_of`: `plate`). So you can count all aggregates, or just
  aggregates of plates. *Show → I classified as…* also lists e.g. *Aggregate of plates*.
* **Changing which classes ask:** in **Admin → Classes**, use *Also ask what it's made of…*
  under any class.

The default classes now include **Bullet** (key T: one arm broken off a rosette)
and **Fragment** (key F; this replaces the old Fragment flag).

## Seeing classifications that have been made

* **Your own:**
  * **Show → Classified by me (review)**, or **I classified as… → e.g. Aggregate
    of plates**.
  * The number on each class button shows how many you've put there; click it to
    see them.
* **Everyone's:** **Show → Classified by anyone**. Particles you haven't
  classified show what other people said, as a dashed tag with the most common
  class and how many people. A `?` means people disagree. The details panel on
  the right lists every class people chose. **Show → People disagree** finds the
  conflicts.
* **Totals:** the **Progress** page has totals per class, per flight and per
  person. Signed-in people can download their own classifications there, and
  administrators everyone's (CSV with `label`, `made_of`, `flags`, person,
  expertise…).
* **Changing them:** while reviewing, press another class key (or drag to a
  class) to change a class. When the particle(s) you're on are an aggregate,
  fragment or graupel, a dashed *made of* bar appears under the images: click to
  set or change what it's made of, or *Clear*. Select several to change them
  together.
* **Narrowing down:** use the flight list (tick projects or flights), the
  **Instrument** buttons above it (CPI v1.0, v1.5, v2.0, 3V-CPI: ticks every
  flight from that instrument), and the size, shape and focus filters. These
  combine with any *Show* choice.
* **Guide page:** shows example particles for each class.
  * **How to decide** at the top lists the questions in order (artefact? drop?
    can you judge it? aggregate? fragment? graupel? which habit? irregular),
    with the classes for each, so everyone reaches the same answer the same way.
    It also reminds people to classify the 3D shape, not the 2D outline.
  * **Agreed by most** takes them from everyone's classifications (particles most
    people agree on); a tick shows you agree, a dashed outline that you said
    something else. **Mine** shows your own classifications.
  * Filter by **project** and **instrument** (CPI v1.0, v1.5, v2.0, 3V-CPI; the
    version comes from each flight's *Instrument → Version*). *8/16/32/60 per
    class* sets how many; *Other examples* picks a different set.
  * The **slider** sets the picture size. **Same scale** draws every particle at
    the same µm per screen pixel, with a scale bar, so sizes compare.
  * **Click** a picture to change your classification (class and flag keys work,
    ← → step through, Z undoes). **Drag** it onto another class, or
    onto a *made of* chip, to reclassify it. In *Mine* it moves there.

## On a phone

The Classify page works on a phone: the class buttons become a compact strip
(swipe sideways for more), flags and status sit underneath, and the particles get
the rest of the screen. Tap a particle, then tap a class. *One at a time* is
usually easiest on a small screen. While *Aggregate of…* is asking, the other
buttons step aside until you answer or tap Skip. Dragging and the look-alike
piles need a mouse.

## Look-alike piles

The **Piles** row under the class buttons is for sorting before naming. Drag
particles that look alike onto **+ New pile** (or onto an existing pile); each
gets a small letter badge in the grid. Click a pile to rename it ("rounded
blobs"), look through it, take out odd ones, show it on the page, or **classify
the whole pile** as one class (undo with Z). Piles are kept in each person's own
browser, not in the database.

## Classifying without saving to the database

* **Not signed in:** that's practice mode already.
* **Signed in:** choose **Practice mode (save to a file only)** in the menu under
  your name. A *Practice mode* badge shows at the top. Use **Save to a file** in
  the bottom bar (or on the Progress page) to download your work.
* **Afterwards:** *Back to saving* offers to keep those classifications under
  your name. Or the file can be added later by an administrator on the Progress
  page: *Add classifications from a file*.
* **Carrying on later, without an account:** *Save to a file*, close the page,
  and next time (in any browser, on any computer) press **Continue from a file**
  (in the bottom bar when the browser has nothing unsaved, or on the Progress
  page) and pick that file. Its classifications come back into the browser, with
  the name and details saved in it, and the Classify page skips them, so you
  carry on where you left off. Anything already in the browser is kept; where
  both have the same particle, the newer classification wins. Save again before
  you stop, and use the newest file next time.

## Dates, changes of mind, and adding a file to someone's profile

* **Every classification has a date.** Each person has one *current* opinion per
  particle (the `labels` table, with `created_utc` and `updated_utc`), and every
  opinion they have ever given is kept, with its date, in the history
  (`label_log`). Agreement and counts use the current opinion, so nobody counts
  twice. Classifications made before this change already have their dates; nothing
  needs filling in.
* **Adding a file to someone's profile (administrators):** Progress → *Add
  classifications from a file*, then *Someone already listed* (picked
  automatically when the name in the file matches), or *Someone without an
  account*. Every classification in the file goes into that person's history with
  the date it was made. Where they had already classified a particle, choose:
  * **The more recent one counts** (default): whichever was classified later
    becomes their current opinion; the other stays in the history.
  * **The file's one counts**, or **Their existing one counts**.
  Identical classifications are skipped.
* **Full history (CSV)** on the Progress page lists every opinion with its date
  (everyone's for administrators, your own otherwise), so you can see how
  someone's view of a particle changed.

## Display settings

* **Defaults:** the colour map is Blues (Blues_r) for greyscale images. Change it
  under **Display**, where you can also switch the particle **outline** on (or
  press O), and set brightness, contrast and labels.
* **They're remembered:** when you're signed in, these settings are stored with
  your account, so they follow you to another computer, and between the
  SSH-tunnel address and the campus address (browsers keep those apart). Not
  signed in, they're remembered by that browser.
* **Image downloads** start with the outline on or off to match the screen.

## Changing a flight's details after importing

In **Admin → Flights and imports**, use **Change settings** on the flight or
project. Edit the instrument (name, version, e.g. `1.0`, `1.5`, `2.0`, or `3V`
for the 3V-CPI), the platform or the notes, then **Save settings only**. The
imported flights are updated straight away, without importing again.

## Archiving flights

Admin → Flights and imports lists the imported flights by project. Tick flights
or days that aren't useful and press *Archive ticked*, or use *Archive* on one.
Archived flights disappear from the Classify, Flights and Progress pages for
everyone. Their images and classifications are kept, and they're listed under
*Archived* with *Restore*. *Remove* deletes the imported images instead (the MAT
files and classifications are kept).

## Filtering

The classify page's sidebar narrows down what you see:

* **Default histogram filters:** size (choose the measure: maximum length,
  equivalent diameter, Feret, shadow circle, …), width, aspect ratio, area
  ratio, focus and time. Temperature and altitude are added when the flight has
  navigation data.
* **Add a filter on…:** any other property.
* **Show:** not yet classified by me, all, not classified by anyone, where
  people disagree, or one of my classes. This is how you review a class and
  move mistakes.

Each histogram shows how many particles pass the other filters. Hover over it
to see the counts.

## Keeping it in a git repository

The folder can live inside another repository (e.g. a top-level `cpi-classifier/`
folder in your processing code). The included `.gitignore` keeps out
`web/api/config.php` and the generated `web/data/`. On the server, clone or
pull, and point the web server (or a symbolic link) at `cpi-classifier/web`. The
data folder and config.php stay where they are.

## Files to keep private

`web/api/config.php` holds the database password. Don't commit it.
