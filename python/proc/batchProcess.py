#!/usr/bin/env python3
"""Top-level CPI processing driver.

Large scientific stages run in disposable subprocesses.  The coordinator stays
small, and per-file workers are used where that is compatible with the data flow.
"""

from __future__ import annotations

from os import environ

# Prevent native libraries from multiplying the Python-level worker count.
# Set these before any child imports NumPy/SciPy/OpenCV.
environ.setdefault("OMP_NUM_THREADS", "1")
environ.setdefault("OPENBLAS_NUM_THREADS", "1")
environ.setdefault("MKL_NUM_THREADS", "1")
environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import os
import subprocess
import sys

from io_utils import normalise_directory

# Processing configuration ---------------------------------------------------
foc_crit = 5
min_len = 1
dt = 10
ds = 10
vel = 100
find_particle_edges = True
process_sweep1_if_exist = True

# Preserve the *effective* settings from the supplied archive.  The original
# file defined these twice, with the later block silently overriding the first.
process_roi_driver = True
process_image_stats = True
export_images = True
output_timeseries = True

# Edge detection (proc/edgeDetect.py).  "legacy" reproduces the original
# processing exactly; "transmittance" thresholds image/background at
# edge_level, traces the edge sub-pixel, uses the maximum Feret length and
# stores variable-length boundaries.  The options are saved in every output
# file (dat.proc_options) so runs can be compared.
# Recommended (best size collapse on the 21/3/02 15/40/100 um beads, README 3.6):
#   method "transmittance", edge_mode "relative", edge_fraction 0.55
#   i.e.  --edge-method transmittance --edge-mode relative --edge-fraction 0.55
# (0.55: sizes grow away from the object plane on both sides; 0.65 is as consistent but
#  100 um beads shrink slightly on the negative side -- README 3.6)
edge_options = {
    "method": "legacy",
    "edge_mode": "relative",    # "relative": edge at a fraction of each particle's own shadow depth;
                                # "fixed": edge at a fixed transmittance (loses faint small particles)
    "edge_level": 0.5,          # fixed mode: transmittance at the edge
    "edge_fraction": 0.55,      # relative mode: fraction of the shadow depth
    "denoise": 1,
    "boundary_tol_px": 0.25,
}

num_cores = 4
cpiv1 = False
outputfile = "timeseries.mat"

classifierFile = "/models/mccikpc2/DCMEX/CPI-analysis/cnn/model_t5_epochs_100_dense64_3a_freeze_final"
classifier = False
minClassSize = 50.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process CPI ROI files")
    parser.add_argument("path", help="Directory containing .roi files")
    parser.add_argument(
        "--cores", type=int, default=num_cores, help="Concurrent image-stat workers"
    )
    parser.add_argument("--edge-method", choices=("legacy", "transmittance"),
                        help="edge detection method (default: edge_options in this file)")
    parser.add_argument("--edge-level", type=float, help="transmittance at the edge (fixed mode)")
    parser.add_argument("--edge-mode", choices=("fixed", "relative"),
                        help="fixed transmittance, or a fraction of each particle's own shadow depth")
    parser.add_argument("--edge-fraction", type=float, help="fraction of the shadow depth (relative mode)")
    parser.add_argument("--outline", choices=("radial", "edge", "contour"),
                        help="radial: smoothed threshold outline for roughly round particles (default, most "
                             "consistent sizes); edge: outline at the steepest brightness rise, as drawn by hand; "
                             "contour: raw threshold outline")
    parser.add_argument("--pixel-um", type=float,
                        help="pixel size in µm used to convert lengths/areas (default 2.3, the nominal value). "
                             "Recorded in every file (dat.proc_options) and read by the workbench fit and the "
                             "concentration time series, so it is never applied twice. Measure it with a graticule "
                             "at the object plane.")
    parser.add_argument("--cpiview", action="store_true",
                        help="also compute CPIview's own particle measures and focus (port of SPEC CPIview 2: "
                             "segmentation, length/width, cut-off %%, water flag, 16-sample focus with its spread "
                             "and edge FWHM), stored per particle as dat.cpv_*, for comparison in the workbench "
                             "(transmittance method only; slower)")
    parser.add_argument("--max-cutoff", type=float,
                        help="leave particles with more than this %% of their outline on the ROI edge "
                             "(dat.cutoff_pct) out of the concentration time series, e.g. 6 as CPIview's spectra; "
                             "their statistics stay in the files")
    parser.add_argument("--only-stats", action="store_true",
                        help="only redo particle statistics on already-converted MAT files "
                             "(run it on a copy of a processed directory to compare edge methods)")
    parser.add_argument("--cpiv1", action="store_true", help="CPI 1.0 data (image type 33857)")
    parser.add_argument("--no-images", action="store_true",
                        help="skip exporting particle images (the slow PNG/montage step)")
    parser.add_argument("--no-timeseries", action="store_true", help="skip the concentration time series")
    parser.add_argument("--backgrounds", choices=("all", "choose", "saved"),
                        help="which full-frame backgrounds to use: all (default); choose = open a window to tick "
                             "them (offers the previous choice if there is one) and save the choice in the folder "
                             "(background_selection.json); saved = reuse that saved choice without the window. "
                             "With --only-stats, giving this re-assigns backgrounds to the ROIs first "
                             "(left out, the backgrounds already in the MAT files are kept)")
    return parser.parse_args()


def _run_script(script, *args):
    command = [sys.executable, os.path.join(_SCRIPT_DIR, script), *map(str, args)]
    subprocess.run(command, check=True)


def _background_selection(path1: str, mode: str) -> None:
    """Put the background selection for this run in force (CPI_BG_SELECTION, read by the workers)."""
    from backgroundSelection import ENV_VAR, read_selection, selection_path
    os.environ.pop(ENV_VAR, None)
    if mode == "all":
        print("Backgrounds: using all")
        return
    if mode == "choose":
        result = subprocess.run([sys.executable, os.path.join(_SCRIPT_DIR, "backgroundSelector.py"), path1])
        if result.returncode != 0:
            raise SystemExit("Background selection cancelled -- nothing more was processed.")
    path = selection_path(path1)
    if not os.path.exists(path):
        raise SystemExit(f"No saved background selection in {path1}: run with --backgrounds choose first.")
    record = read_selection(path)
    print(f"Backgrounds: using {record.get('n_used', '?')} of {record.get('n_frames', '?')} "
          f"(selection saved {record.get('saved', '?')})")
    os.environ[ENV_VAR] = path


def _assign_backgrounds(path1: str, mat_files, use_v1: bool) -> None:
    """Second sweep only: assign (the selected) backgrounds to every ROI of these MAT files."""
    print("=========================assigning backgrounds================================")
    for name in mat_files:
        _run_script("ROIDataDriver.py", "sweep2", path1, name, use_v1)


def run_jobs(path1: str, cores: int = num_cores, edge=None, only_stats: bool = False,
             cpi_v1: bool = None, images: bool = None, timeseries: bool = None,
             backgrounds: str = None, max_cutoff: float = None) -> None:
    edge = dict(edge_options, **(edge or {}))
    use_v1 = cpiv1 if cpi_v1 is None else cpi_v1
    path1 = normalise_directory(path1)
    if not os.path.isdir(path1):
        raise NotADirectoryError(path1)

    if only_stats:
        # Already-converted MAT files (with ROI_N and BG), e.g. a copy of a processed run.
        filename1 = sorted(f for f in os.listdir(path1) if f.endswith(".mat")
                           and f not in ("timeseries.mat", "full_backgrounds.mat", outputfile))
        if not filename1:
            hint = (" — this folder has raw .roi files only: run without --only-stats to process them from scratch,"
                    " or copy the processed .mat files (incl. full_backgrounds.mat) here first"
                    if any(f.endswith(".roi") for f in os.listdir(path1)) else "")
            raise FileNotFoundError(f"No processed .mat files found in {path1}{hint}")
    else:
        filename1 = sorted(f for f in os.listdir(path1) if f.endswith(".roi"))
        if not filename1:
            raise FileNotFoundError(f"No .roi files found in {path1}")

    if only_stats:
        if backgrounds:
            _background_selection(path1, backgrounds)
            _assign_backgrounds(path1, filename1, use_v1)
    elif process_roi_driver:
        # ROIDataDriver itself uses a fresh child for each file.  Running its
        # coordinator in a child also keeps SciPy out of this top-level process.
        mode = backgrounds or "all"
        if mode == "all":
            _background_selection(path1, "all")
            _run_script("ROIDataDriver.py", "driver", path1, dt, process_sweep1_if_exist, use_v1)
        else:
            # collect the backgrounds, choose, then assign them
            _run_script("ROIDataDriver.py", "driver", path1, dt, process_sweep1_if_exist, use_v1, "1")
            _background_selection(path1, mode)
            _run_script("ROIDataDriver.py", "driver", path1, dt, process_sweep1_if_exist, use_v1, "2")

    if process_image_stats:
        # This driver is intentionally lightweight; each file is still handled
        # by a separate subprocess, up to --cores concurrently.
        from imageStatsDriver import imageStatsDriver

        imageStatsDriver(
            path1, filename1, find_particle_edges, use_v1, max(1, int(cores)), edge_options=edge
        )
        if only_stats:
            return

    if (export_images if images is None else images):
        from exportImagesDriver import exportImagesDriver

        exportImagesDriver(
            path1,
            filename1,
            foc_crit,
            min_len,
            use_v1,
            classifier,
            classifierFile,
            minClassSize,
        )

    if (output_timeseries if timeseries is None else timeseries):
        if not classifier:
            # Keep the large per-file MAT reads out of the coordinator.
            _run_script(
                "calcTimeseriesDriver.py",
                "worker",
                path1,
                foc_crit,
                dt,
                ds,
                vel,
                outputfile,
                use_v1,
                *([max_cutoff] if max_cutoff is not None else []),
            )
        else:
            # The classifier path remains a separate legacy path because it
            # carries model state across all files.
            from calcTimeseriesClassifierDriver import calcTimeseriesClassifierDriver

            calcTimeseriesClassifierDriver(
                path1,
                filename1,
                foc_crit,
                dt,
                ds,
                vel,
                "timeseries_class.mat",
                use_v1,
                classifierFile,
                minClassSize,
            )


if __name__ == "__main__":
    args = _parse_args()
    edge = {}
    if args.edge_method:
        edge["method"] = args.edge_method
    if args.edge_level is not None:
        edge["edge_level"] = args.edge_level
    if args.edge_mode:
        edge["edge_mode"] = args.edge_mode
    if args.edge_fraction is not None:
        edge["edge_fraction"] = args.edge_fraction
    if args.outline:
        edge["outline"] = args.outline
    if args.cpiview:
        edge["cpiview"] = True
    if args.pixel_um is not None:
        if not 0.5 <= args.pixel_um <= 20:
            raise SystemExit(f"--pixel-um {args.pixel_um}: expected a pixel size in µm (about 2-3 for the CPI)")
        edge["pixel_um"] = args.pixel_um
    run_jobs(args.path, args.cores, edge=edge, only_stats=args.only_stats, cpi_v1=True if args.cpiv1 else None,
             images=False if args.no_images else None, timeseries=False if args.no_timeseries else None,
             backgrounds=args.backgrounds, max_cutoff=args.max_cutoff)
