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
import math
import json
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
    "method": "transmittance",
    "edge_mode": "relative",    # "relative": edge at a fraction of each particle's own shadow depth;
                                # "fixed": edge at a fixed transmittance (loses faint small particles)
    "edge_level": 0.5,          # fixed mode: transmittance at the edge
    "edge_fraction": 0.55,      # relative mode: fraction of the shadow depth
    "outline": "radial",      # smooth radial boundary; shape / morphology remains separate
    "denoise": 1,
    "shadow_circle_padding": 8.0,  # additional shadow-equivalent diameter by default
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
                        help="edge method (default: transmittance; use legacy to reproduce older processing)")
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
    parser.add_argument("--denoise", type=int, choices=range(4), metavar="{0,1,2,3}",
                        help="3x3 median-filter passes for transmittance processing (default: 1)")
    parser.add_argument("--smooth-sigma", type=float,
                        help="Gaussian smoothing sigma in pixels (default: 0)")
    parser.add_argument("--shadow-circle-padding", type=float, default=None,
                        help="also store signed-shadow equivalent diameter in dat.shadow_deq (µm), "
                             "from a circle fitted to the ordinary contour, extended by this many pixels "
                             "(0–30, default 8 px); leaves normal shape and focus intact")
    parser.add_argument("--shadow-circle-auto", action="store_true",
                        help="experimental: select integration radius from a sustained signed-integral plateau; "
                             "fail/flag if no trustworthy plateau or ROI is too cropped")
    parser.add_argument("--shadow-circle-max-padding", type=float, default=None,
                        help="maximum additional radius searched for plateau (6–60 px; default 30)")
    parser.add_argument("--no-shadow-circle", action="store_true",
                        help="disable additional signed-shadow circle sizing (on by default, +8 px)")
    parser.add_argument("--cpiview", action="store_true",
                        help="also compute CPIview's own particle measures and focus (port of SPEC CPIview 2: "
                             "segmentation, length/width, cut-off %%, water flag, 16-sample focus with its spread "
                             "and edge FWHM), stored per particle as dat.cpv_*, for comparison in the workbench "
                             "(transmittance method only; slower)")
    parser.add_argument("--max-cutoff", type=float,
                        help="leave particles with more than this %% of their outline on the ROI edge "
                             "(dat.cutoff_pct) out of the concentration time series, e.g. 6 as CPIview's spectra; "
                             "their statistics stay in the files")
    parser.add_argument("--export-only", action="store_true",
                        help="export ROI PNGs/catalogue from existing MAT files; skip all CPI processing")
    parser.add_argument("--export-png", action="store_true", help="export individual full-ROI PNG files")
    parser.add_argument("--png-dir", help="output root for the PNG image archive")
    parser.add_argument("--project", help="campaign name, otherwise input folder name")
    parser.add_argument("--flight-csv", help="CSV with flight/date associations")
    parser.add_argument("--min-size-um", type=float, default=0., help="minimum measured length for PNG export (µm)")
    parser.add_argument("--min-focus", type=float, help="minimum existing CPI particle focus value for PNG export")
    parser.add_argument("--grayscale", "--greyscale", action="store_true",
                        help="use grayscale instead of colour maps for background-normalised particle PNGs and montages")
    parser.add_argument("--background-level", type=float, default=180.,
                        help="display intensity at background transmittance 1 (default: 180; 0 < value <= 255)")
    parser.add_argument("--overwrite-png", action="store_true", help="regenerate PNGs that already exist")
    parser.add_argument("--only-stats", action="store_true",
                        help="only redo particle statistics on already-converted MAT files "
                             "(run it on a copy of a processed directory to compare edge methods)")
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
             backgrounds: str = None, max_cutoff: float = None, grayscale: bool = False, background_level: float = 180.) -> None:
    edge = dict(edge_options, **(edge or {}))
    if classifier:
        raise NotImplementedError("Auto instrument profiles are not yet supported by the classifier timeseries path")
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
        roi_files = sorted(f for f in os.listdir(path1) if f.endswith(".roi"))
        if not roi_files:
            raise FileNotFoundError(f"No .roi files found in {path1}")
        filename1 = roi_files

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

    # CPI_SKIP_FAILED_V1: only send successfully converted files downstream.
    if not only_stats and process_roi_driver:
        report_file = os.path.join(path1, "cpi_processing_report.json")
        if not os.path.exists(report_file):
            raise RuntimeError("Conversion report missing; refusing to use potentially stale MAT files")
        with open(report_file) as f:
            report = json.load(f)
        filename1 = [f for f in filename1 if f in report.get("sweep2_ok", [])]
        print(f"Continuing with {len(filename1)} converted files; "
              f"{len(report.get('failures', []))} conversion failures. "
              f"See {report_file}", flush=True)
        if not filename1:
            raise RuntimeError("No ROI files converted successfully; stopping further processing")

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
            grayscale=grayscale,
            background_level=background_level,
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
    if not math.isfinite(args.background_level) or not 0 < args.background_level <= 255:
        raise SystemExit("--background-level must be > 0 and <= 255")
    if args.export_only and not args.export_png:
        raise SystemExit("--export-only requires --export-png")
    if args.export_png and not args.png_dir:
        raise SystemExit("--export-png requires --png-dir")
    if args.export_only:
        from exportParticles import export
        export(args.path, args.png_dir, args.project, args.flight_csv,
               args.min_size_um, args.min_focus, overwrite=args.overwrite_png, grayscale=args.grayscale,
               background_level=args.background_level, cores=args.cores)
        sys.exit(0)
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
    if args.no_shadow_circle and args.shadow_circle_padding is not None:
        raise SystemExit("Use either --no-shadow-circle or --shadow-circle-padding, not both")
    if args.no_shadow_circle:
        edge["shadow_circle_padding"] = None
    if args.shadow_circle_padding is not None:
        if not 0 <= args.shadow_circle_padding <= 30:
            raise SystemExit("--shadow-circle-padding must be between 0 and 30 pixels")
        edge["shadow_circle_padding"] = args.shadow_circle_padding
    if args.shadow_circle_auto:
        if args.no_shadow_circle:
            raise SystemExit("Cannot combine --shadow-circle-auto with --no-shadow-circle")
        edge["shadow_circle_auto"] = True
    if args.shadow_circle_max_padding is not None:
        if not 6 <= args.shadow_circle_max_padding <= 60:
            raise SystemExit("--shadow-circle-max-padding must be 6–60 px")
        edge["shadow_circle_max_padding"] = args.shadow_circle_max_padding
    if args.cpiview:
        edge["cpiview"] = True
    if args.denoise is not None:
        edge["denoise"] = args.denoise
    if args.smooth_sigma is not None:
        if not 0 <= args.smooth_sigma <= 3:
            raise SystemExit("--smooth-sigma must be between 0 and 3 pixels")
        edge["smooth_sigma"] = args.smooth_sigma
    if args.pixel_um is not None:
        if not 0.5 <= args.pixel_um <= 20:
            raise SystemExit(f"--pixel-um {args.pixel_um}: expected a pixel size in µm (about 2-3 for the CPI)")
        edge["pixel_um"] = args.pixel_um
    run_jobs(args.path, args.cores, edge=edge, only_stats=args.only_stats, cpi_v1=None,
             images=False if args.no_images else None, timeseries=False if args.no_timeseries else None,
             backgrounds=args.backgrounds, max_cutoff=args.max_cutoff, grayscale=args.grayscale,
             background_level=args.background_level)

    if args.export_png and not args.only_stats:
        from exportParticles import export
        export(args.path, args.png_dir, args.project, args.flight_csv,
               args.min_size_um, args.min_focus, overwrite=args.overwrite_png, grayscale=args.grayscale,
               background_level=args.background_level, cores=args.cores)
