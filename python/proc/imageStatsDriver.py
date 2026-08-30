#!/usr/bin/env python3
"""Parallel per-file particle-statistics driver.

Each ROI file is processed by a fresh Python subprocess.  This is intentional:
when the child exits, all NumPy/SciPy/OpenCV allocations are returned to the OS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass


from io_utils import atomic_savemat, normalise_directory, parse_bool


@dataclass
class _Job:
    process: subprocess.Popen
    filename: str
    file_index: int
    retries: int = 0


def _worker_command(path1, filename, find_particle_edges, cpiv1, position):
    return [
        sys.executable,
        os.path.abspath(__file__),
        "worker",
        path1,
        filename,
        str(bool(find_particle_edges)),
        str(bool(cpiv1)),
        str(position),
    ]


def imageStatsDriver(
    path1,
    filename1,
    find_particle_edges,
    cpiv1,
    num_cores,
    max_retries=1,
    poll_interval=0.2,
):
    """Process files concurrently, with one fresh subprocess per file."""
    print("====================particle properties===========================")
    path1 = normalise_directory(path1)
    filenames = list(filename1)
    if not filenames:
        return

    nworkers = min(max(1, int(num_cores)), len(filenames))
    active: dict[int, _Job] = {}
    failures: list[tuple[str, int]] = []
    next_file = 0

    try:
        while next_file < len(filenames) or active:
            # Fill free worker slots.
            for slot in range(nworkers):
                if next_file >= len(filenames):
                    break
                if slot in active:
                    continue

                filename = filenames[next_file]
                command = _worker_command(
                    path1, filename, find_particle_edges, cpiv1, slot
                )
                print(" ".join(command))
                active[slot] = _Job(
                    subprocess.Popen(command), filename, next_file, retries=0
                )
                next_file += 1

            made_progress = False
            for slot, job in list(active.items()):
                returncode = job.process.poll()
                if returncode is None:
                    continue

                made_progress = True
                if returncode == 0:
                    del active[slot]
                    continue

                if job.retries < max_retries:
                    retry = job.retries + 1
                    print(
                        f"{job.filename} failed with exit code {returncode}; "
                        f"retrying ({retry}/{max_retries})"
                    )
                    command = _worker_command(
                        path1, job.filename, find_particle_edges, cpiv1, slot
                    )
                    active[slot] = _Job(
                        subprocess.Popen(command),
                        job.filename,
                        job.file_index,
                        retries=retry,
                    )
                else:
                    failures.append((job.filename, returncode))
                    del active[slot]

            if active and not made_progress:
                time.sleep(poll_interval)

    except BaseException:
        # Do not leave orphan workers if the driver is interrupted or fails.
        for job in active.values():
            if job.process.poll() is None:
                job.process.terminate()
        for job in active.values():
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()
        raise

    if failures:
        detail = ", ".join(f"{name} (exit {code})" for name, code in failures)
        raise RuntimeError(f"Particle-stat processing failed: {detail}")


def mult_job(path1, filename1, find_particle_edges, cpiv1, position, lock):
    """Process exactly one MAT file and atomically replace it with stats added."""
    import scipy.io as sio
    from imageStats import imageStats

    path1 = normalise_directory(path1)
    mat_file = os.path.join(path1, filename1.replace(".roi", ".mat"))
    data = sio.loadmat(mat_file, variable_names=["ROI_N", "HOUSE", "IMAGE1", "BG"])
    ROI_N = data["ROI_N"]
    HOUSE = data["HOUSE"]
    IMAGE1 = data["IMAGE1"]
    BG = data["BG"]

    dat = imageStats(
        ROI_N, BG, cpiv1, find_particle_edges, position, filename1, lock
    )

    atomic_savemat(
        mat_file,
        {"ROI_N": ROI_N, "HOUSE": HOUSE, "IMAGE1": IMAGE1, "BG": BG, "dat": dat},
    )
    return position


class NullContextManager:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "worker":
        raise SystemExit(
            "usage: imageStatsDriver.py worker PATH FILE FIND_EDGES CPIV1 POSITION"
        )

    path1 = sys.argv[2]
    filename1 = sys.argv[3]
    find_particle_edges = parse_bool(sys.argv[4])
    cpiv1 = parse_bool(sys.argv[5])
    position = int(sys.argv[6])
    mult_job(
        path1,
        filename1,
        find_particle_edges,
        cpiv1,
        position,
        NullContextManager(),
    )
