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
#   method "transmittance", edge_mode "relative", edge_fraction 0.65
#   i.e.  --edge-method transmittance --edge-mode relative --edge-fraction 0.65
edge_options = {
    "method": "legacy",
    "edge_mode": "relative",    # "relative": edge at a fraction of each particle's own shadow depth;
                                # "fixed": edge at a fixed transmittance (loses faint small particles)
    "edge_level": 0.5,          # fixed mode: transmittance at the edge
    "edge_fraction": 0.65,      # relative mode: fraction of the shadow depth
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
    parser.add_argument("--only-stats", action="store_true",
                        help="only redo particle statistics on already-converted MAT files "
                             "(run it on a copy of a processed directory to compare edge methods)")
    parser.add_argument("--cpiv1", action="store_true", help="CPI 1.0 data (image type 33857)")
    return parser.parse_args()


def _run_script(script, *args):
    command = [sys.executable, os.path.join(_SCRIPT_DIR, script), *map(str, args)]
    subprocess.run(command, check=True)


def run_jobs(path1: str, cores: int = num_cores, edge=None, only_stats: bool = False,
             cpi_v1: bool = None) -> None:
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
            raise FileNotFoundError(f"No processed .mat files found in {path1}")
    else:
        filename1 = sorted(f for f in os.listdir(path1) if f.endswith(".roi"))
        if not filename1:
            raise FileNotFoundError(f"No .roi files found in {path1}")

    if process_roi_driver and not only_stats:
        # ROIDataDriver itself uses a fresh child for each file.  Running its
        # coordinator in a child also keeps SciPy out of this top-level process.
        _run_script(
            "ROIDataDriver.py",
            "driver",
            path1,
            dt,
            process_sweep1_if_exist,
            use_v1,
        )

    if process_image_stats:
        # This driver is intentionally lightweight; each file is still handled
        # by a separate subprocess, up to --cores concurrently.
        from imageStatsDriver import imageStatsDriver

        imageStatsDriver(
            path1, filename1, find_particle_edges, use_v1, max(1, int(cores)), edge_options=edge
        )
        if only_stats:
            return

    if export_images:
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

    if output_timeseries:
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
    run_jobs(args.path, args.cores, edge=edge, only_stats=args.only_stats, cpi_v1=True if args.cpiv1 else None)
