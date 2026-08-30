"""Build the ML HDF5 dataset without retaining all particle images in RAM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


def _processed_mat_files(directory: Path):
    return sorted(
        p for p in directory.glob("*.mat")
        if p.name not in {"full_backgrounds.mat", "timeseries.mat"} and not p.name.startswith(".")
    )


def _append_dataset(handle, name, values, shape_tail, dtype, compression="gzip"):
    values = np.asarray(values, dtype=dtype)
    if name not in handle:
        handle.create_dataset(
            name,
            data=values,
            maxshape=(None,) + tuple(shape_tail),
            chunks=True,
            compression=compression,
            shuffle=True,
        )
    else:
        dataset = handle[name]
        old = len(dataset)
        dataset.resize(old + len(values), axis=0)
        dataset[old:] = values


def _process_one(input_mat: Path, background: Path, output_npz: Path, focus: float, min_length: float):
    # Keep this import inside the disposable worker: the legacy algorithm is
    # intentionally preserved for historical-model reproducibility.
    ml_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ml_dir))
    import postProcessImages05

    result = postProcessImages05.postProcessing(
        str(input_mat), str(background), foc_crit=focus, min_len=min_length
    )
    images, lens, times, diams, rounds, l2ws, radii, roi_indices, total = result
    if images:
        payload = {
            "images": np.stack(images).astype(np.uint8),
            "lens": np.asarray(lens, dtype=np.float32),
            "times": np.asarray(times, dtype=np.float64),
            "diams": np.stack(diams).astype(np.float32),
            "rounds": np.asarray(rounds, dtype=np.float32),
            "l2ws": np.asarray(l2ws, dtype=np.float32),
            "radii": np.stack(radii).astype(np.float32),
            "roi_index": np.asarray(roi_indices, dtype=np.int64),
        }
    else:
        payload = {
            "images": np.empty((0, 128, 128), dtype=np.uint8),
            "lens": np.empty(0, dtype=np.float32),
            "times": np.empty(0, dtype=np.float64),
            "diams": np.empty((0, 12), dtype=np.float32),
            "rounds": np.empty(0, dtype=np.float32),
            "l2ws": np.empty(0, dtype=np.float32),
            "radii": np.empty((0, 24), dtype=np.float32),
            "roi_index": np.empty(0, dtype=np.int64),
        }
    np.savez_compressed(output_npz, **payload)


def build_dataset(directories, output, focus=5.0, min_length=50.0):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as handle:
        for root_text in directories:
            root = Path(root_text).resolve()
            background = root / "full_backgrounds.mat"
            if not background.exists():
                raise FileNotFoundError("Missing {}".format(background))
            for input_mat in _processed_mat_files(root):
                fd, temp_name = tempfile.mkstemp(prefix="cpi_ml_", suffix=".npz")
                os.close(fd)
                temporary = Path(temp_name)
                try:
                    command = [
                        sys.executable, os.path.abspath(__file__), "worker",
                        str(input_mat), str(background), str(temporary), str(focus), str(min_length),
                    ]
                    subprocess.run(command, check=True)
                    with np.load(temporary) as data:
                        n = len(data["lens"])
                        if not n:
                            continue
                        _append_dataset(handle, "images", data["images"], (128, 128), np.uint8)
                        _append_dataset(handle, "lens", data["lens"], (), np.float32)
                        _append_dataset(handle, "times", data["times"], (), np.float64)
                        _append_dataset(handle, "diams", data["diams"], (data["diams"].shape[1],), np.float32)
                        _append_dataset(handle, "rounds", data["rounds"], (), np.float32)
                        _append_dataset(handle, "l2ws", data["l2ws"], (), np.float32)
                        _append_dataset(handle, "radii", data["radii"], (data["radii"].shape[1],), np.float32)
                        _append_dataset(handle, "roi_index", data["roi_index"], (), np.int64)
                        source = np.asarray([str(input_mat)] * n, dtype=object)
                        if "source_file" not in handle:
                            handle.create_dataset("source_file", data=source, maxshape=(None,), dtype=string_dtype, chunks=True)
                        else:
                            ds = handle["source_file"]
                            old = len(ds)
                            ds.resize(old + n, axis=0)
                            ds[old:] = source
                    handle.flush()
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build CPI ML dataset with per-file worker processes")
    parser.add_argument("output")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--focus", type=float, default=5.0)
    parser.add_argument("--min-length", type=float, default=50.0)
    args = parser.parse_args(argv)
    build_dataset(args.directories, args.output, args.focus, args.min_length)


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "worker":
        _process_one(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    else:
        main()
