#!/usr/bin/env python3
"""Run image export in a disposable subprocess."""

from __future__ import annotations

import os
import subprocess
import sys

from io_utils import normalise_directory, parse_bool


def exportImagesDriver(
    path1,
    filename1,
    foc_crit,
    min_len,
    cpiv1,
    classifier,
    classifierFile,
    minClassSize,
):
    # The child discovers the same sorted .roi list itself.  This keeps
    # Matplotlib (and optional Keras) out of the long-lived coordinator.
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "worker",
        normalise_directory(path1),
        str(foc_crit),
        str(min_len),
        str(bool(cpiv1)),
        str(bool(classifier)),
        classifierFile,
        str(minClassSize),
    ]
    subprocess.run(command, check=True)


def _worker(path1, foc_crit, min_len, cpiv1, classifier, classifierFile, minClassSize):
    import scipy.io as sio

    from exportImages import exportImages

    path1 = normalise_directory(path1)
    filenames = sorted(f for f in os.listdir(path1) if f.endswith(".roi"))
    cmap_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmap.mat")
    MAP2 = sio.loadmat(cmap_file, variable_names=["MAP2"])["MAP2"]

    print("====================exporting images =============================")
    print("exporting all images...")
    exportImages(
        path1,
        filenames,
        foc_crit,
        min_len,
        MAP2,
        cpiv1,
        classifier,
        classifierFile,
        minClassSize,
    )
    print("done")


if __name__ == "__main__":
    if len(sys.argv) != 9 or sys.argv[1] != "worker":
        raise SystemExit(
            "usage: exportImagesDriver.py worker PATH FOC MINLEN CPIV1 CLASSIFIER MODEL MINCLASSSIZE"
        )
    _worker(
        sys.argv[2],
        float(sys.argv[3]),
        float(sys.argv[4]),
        parse_bool(sys.argv[5]),
        parse_bool(sys.argv[6]),
        sys.argv[7],
        float(sys.argv[8]),
    )
