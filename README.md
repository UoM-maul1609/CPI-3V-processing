# CPI-3V processing

Tools for processing CPI-3V ROI files, calculating particle properties and concentration time series, reviewing calibration particles, exporting data, and running experimental machine-learning workflows.

The Python processing code is the current reference implementation. The repository also retains MATLAB and legacy ML code for reproducibility and comparison.

## Recommended Python setup

Use a dedicated virtual environment for this repository. This prevents CPI dependencies such as NumPy/SciPy, OpenCV, Qt, Matplotlib, NetCDF, and TensorFlow from changing packages used by unrelated projects.

### Python version

Python **3.12.14** is the recommended version for the current repository. Python **3.12.0 should not be used** because of an incompatibility encountered between its import machinery and the PySide6/`six` stack used by the calibration GUI.

If you use `pyenv`, the committed `.python-version` file selects 3.12.14 automatically once that Python version is installed:

```bash
pyenv install 3.12.14
python --version
```

You do not need `pyenv`; any suitable Python installation can create the virtual environment.

### Create the environment

From the repository root:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Check that the expected interpreter is active:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The executable should point inside this repository's `.venv` directory.

### Install the project

For core processing only:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For core processing **and the calibration GUI**:

```bash
python -m pip install -e ".[gui]"
```

For NetCDF/HDF5 output support as well:

```bash
python -m pip install -e ".[gui,output]"
```

For the experimental machine-learning dependencies:

```bash
python -m pip install -e ".[ml]"
```

The existing `requirements-*.txt` files are retained for users who prefer requirements-file installs, for example:

```bash
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
```

### Deactivate the environment

When finished:

```bash
deactivate
```

To use the repository again later:

```bash
cd /path/to/cpi3v
source .venv/bin/activate
```

There is no need to reinstall packages each time.

## Calibration GUI

After installing the GUI dependencies, launch the calibration program from anywhere in the activated environment with:

```bash
cpi-calibrate
```

or:

```bash
python -m calibration
```

You can optionally provide the processed-data directory immediately:

```bash
cpi-calibrate /path/to/processed/calibration
```

Running `python python/calibration/gui.py` is retained for compatibility, but the installed command or module launch is preferred.

The GUI reviews one particle at a time, supports filtering by particle/time/image metrics, records keep/reject decisions, and stores object-plane positions in a CSV sidecar without modifying the processed MAT files. See `python/calibration/README.md` for details.

## Main repository areas

- `python/proc/` — main CPI ROI processing and time-series calculation.
- `python/calibration/` — interactive calibration-particle review GUI.
- `python/post/` — post-processing/calibration support routines.
- `python/output/` — NetCDF/output conversion.
- `python/ml/` — legacy and modernized experimental ML workflows.
- `matlab/` — retained MATLAB implementation and calibration code.
- `bash/` — batch/helper scripts.

## Git and the virtual environment

The local `.venv/` directory is intentionally ignored by Git. Commit the dependency definitions (`pyproject.toml`, `requirements-*.txt`, and `.python-version`), not the installed environment itself.

## Processing

The processing workflow extracts data blocks from CPI-3V `.roi` files, associates backgrounds, derives particle properties and boundaries, exports particle imagery, and calculates concentration time series segregated by particle properties.

See the README files under `python/` and `matlab/` for historical workflow notes. New development should preferentially target the Python implementation.

## Use and citation

The code was developed by Dr Paul Connolly, University of Manchester, and is available for collaborative use. Please cite the original source of the code when using it in scientific work.
