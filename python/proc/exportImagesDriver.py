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
    grayscale=False,
    background_level=180,
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
        str(bool(grayscale)),
        str(float(background_level)),
    ]
    subprocess.run(command, check=True)


def _worker(path1, foc_crit, min_len, cpiv1, classifier, classifierFile, minClassSize, grayscale=False, background_level=180):
    import scipy.io as sio

    from exportImages import exportImages

    path1 = normalise_directory(path1)
    filenames = []
    for f in sorted(f for f in os.listdir(path1) if f.endswith(".roi")):
        # only files that were converted and have particle statistics (a ROI file that
        # failed conversion has no MAT; one with no particles may have no 'dat')
        mat = os.path.join(path1, f[:-4] + ".mat")
        try:
            ok = "dat" in {v[0] for v in sio.whosmat(mat)}
        except (OSError, ValueError):
            ok = False
        if ok:
            filenames.append(f)
        else:
            print(f"WARNING: skipping {f}: no processed MAT file with particle statistics", flush=True)
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
        grayscale=grayscale,
        background_level=background_level,
    )
    print("done")


if __name__ == "__main__":
    if len(sys.argv) not in (9, 10, 11) or sys.argv[1] != "worker":
        raise SystemExit(
            "usage: exportImagesDriver.py worker PATH FOC MINLEN CPIV1 CLASSIFIER MODEL MINCLASSSIZE [GRAYSCALE] [BACKGROUND_LEVEL]"
        )
    _worker(
        sys.argv[2],
        float(sys.argv[3]),
        float(sys.argv[4]),
        parse_bool(sys.argv[5]),
        parse_bool(sys.argv[6]),
        sys.argv[7],
        float(sys.argv[8]),
        parse_bool(sys.argv[9]) if len(sys.argv) >= 10 else False,
        float(sys.argv[10]) if len(sys.argv) == 11 else 180.,
    )
