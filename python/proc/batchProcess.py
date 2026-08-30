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
process_roi_driver = False
process_image_stats = True
export_images = True
output_timeseries = True

num_cores = 4
cpiv1 = True
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
    return parser.parse_args()


def _run_script(script, *args):
    command = [sys.executable, os.path.join(_SCRIPT_DIR, script), *map(str, args)]
    subprocess.run(command, check=True)


def run_jobs(path1: str, cores: int = num_cores) -> None:
    path1 = normalise_directory(path1)
    if not os.path.isdir(path1):
        raise NotADirectoryError(path1)

    filename1 = sorted(f for f in os.listdir(path1) if f.endswith(".roi"))
    if not filename1:
        raise FileNotFoundError(f"No .roi files found in {path1}")

    if process_roi_driver:
        # ROIDataDriver itself uses a fresh child for each file.  Running its
        # coordinator in a child also keeps SciPy out of this top-level process.
        _run_script(
            "ROIDataDriver.py",
            "driver",
            path1,
            dt,
            process_sweep1_if_exist,
            cpiv1,
        )

    if process_image_stats:
        # This driver is intentionally lightweight; each file is still handled
        # by a separate subprocess, up to --cores concurrently.
        from imageStatsDriver import imageStatsDriver

        imageStatsDriver(
            path1, filename1, find_particle_edges, cpiv1, max(1, int(cores))
        )

    if export_images:
        from exportImagesDriver import exportImagesDriver

        exportImagesDriver(
            path1,
            filename1,
            foc_crit,
            min_len,
            cpiv1,
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
                cpiv1,
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
                cpiv1,
                classifierFile,
                minClassSize,
            )


if __name__ == "__main__":
    args = _parse_args()
    run_jobs(args.path, args.cores)
