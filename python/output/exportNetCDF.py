#!/usr/bin/env python3
"""Export CPI time-series MAT files to CF-style NetCDF.

The old script embedded experiment paths, commit ids and metadata and emitted an
invalid ``%MS`` time-unit string.  This version separates data conversion from
campaign metadata and works with current scipy MAT files; ``hdf5storage`` is an
optional fallback for historical v7.3 files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import scipy.io as sio


def matlab2datetime(value):
    value = float(np.asarray(value).reshape(-1)[0])
    return datetime.fromordinal(int(value)) + timedelta(days=value % 1.0) - timedelta(days=366)


def _load_timeser(filename):
    try:
        loaded = sio.loadmat(filename, variable_names=["timeser"], simplify_cells=True)
        return loaded["timeser"]
    except NotImplementedError:
        try:
            import hdf5storage
        except ImportError as exc:
            raise RuntimeError(
                "This is a MATLAB v7.3/HDF5 file. Install hdf5storage to read it."
            ) from exc
        return hdf5storage.loadmat(filename, variable_names=["timeser"])["timeser"]


def _field(timeser, name):
    if isinstance(timeser, dict):
        return np.asarray(timeser[name])
    value = timeser[name]
    # Remove only MATLAB struct/object wrapper dimensions, not real data axes.
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return np.asarray(value).squeeze()


def _parse_bins(text):
    if text is None or text.strip() == "":
        return []
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def export_netcdf(
    input_file,
    output_file,
    drop_bins=(),
    ice_bins=(),
    title=None,
    institution="University of Manchester",
    source="CPI-3V processed particle imagery",
    history=None,
    comment=None,
):
    from netCDF4 import Dataset

    timeser = _load_timeser(str(input_file))
    matlab_time = np.asarray(_field(timeser, "Time"), dtype=float).reshape(-1)
    size1_data = np.asarray(_field(timeser, "size1"), dtype=float).reshape(-1)
    size2_data = np.asarray(_field(timeser, "size2"), dtype=float).reshape(-1)
    conc_data = np.asarray(_field(timeser, "conc"), dtype=float).reshape(-1)
    conc2_data = np.asarray(_field(timeser, "conc2"), dtype=float)
    conc2ar = np.asarray(_field(timeser, "conc2ar"), dtype=float)

    if conc2_data.shape != (len(matlab_time), len(size1_data)):
        conc2_data = np.reshape(conc2_data, (len(matlab_time), len(size1_data)))
    if conc2ar.shape[0:2] != (len(matlab_time), len(size1_data)):
        conc2ar = np.reshape(conc2ar, (len(matlab_time), len(size1_data), -1))

    drop_bins = np.asarray(list(drop_bins), dtype=int)
    ice_bins = np.asarray(list(ice_bins), dtype=int)
    for label, bins in (("drop", drop_bins), ("ice", ice_bins)):
        if bins.size and (bins.min() < 0 or bins.max() >= conc2ar.shape[2]):
            raise IndexError(
                "{} class-axis index outside [0, {}]: {}".format(
                    label, conc2ar.shape[2] - 1, bins.tolist()
                )
            )

    origin = matlab2datetime(matlab_time[0])
    seconds = (matlab_time - matlab_time[0]) * 86400.0
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(output_file, mode="w", format="NETCDF4_CLASSIC") as ncfile:
        ncfile.title = title or "CPI-3V concentration and size-distribution data"
        ncfile.institution = institution
        ncfile.source = source
        ncfile.history = history or "Converted from CPI processing MAT output"
        if comment:
            ncfile.comment = comment
        ncfile.references = "https://github.com/UoM-maul1609/CPI-3V-processing"

        ncfile.createDimension("time", len(matlab_time))
        ncfile.createDimension("size", len(size1_data))

        time = ncfile.createVariable("time", np.float64, ("time",))
        time.units = "seconds since " + origin.strftime("%Y-%m-%d %H:%M:%S")
        time.calendar = "proleptic_gregorian"
        time.long_name = "time"
        time[:] = seconds

        size1 = ncfile.createVariable("size1", np.float64, ("size",))
        size1.units = "micrometres"
        size1.long_name = "lower particle-size bin edge"
        size1[:] = size1_data

        size2 = ncfile.createVariable("size2", np.float64, ("size",))
        size2.units = "micrometres"
        size2.long_name = "upper particle-size bin edge"
        size2[:] = size2_data

        conc = ncfile.createVariable("conc", np.float64, ("time",), zlib=True)
        conc.units = "m-3"
        conc.long_name = "total particle number concentration"
        conc[:] = conc_data

        conc2 = ncfile.createVariable("conc2", np.float64, ("time", "size"), zlib=True)
        conc2.units = "m-3"
        conc2.long_name = "particle number concentration by size bin"
        conc2[:] = conc2_data

        if drop_bins.size:
            drop2 = np.nansum(conc2ar[:, :, drop_bins], axis=2)
            variable = ncfile.createVariable("conc2Drops", np.float64, ("time", "size"), zlib=True)
            variable.units = "m-3"
            variable.long_name = "drop number concentration by size bin"
            variable[:] = drop2
            variable = ncfile.createVariable("concDrops", np.float64, ("time",), zlib=True)
            variable.units = "m-3"
            variable.long_name = "drop number concentration"
            variable[:] = np.nansum(drop2, axis=1)

        if ice_bins.size:
            ice2 = np.nansum(conc2ar[:, :, ice_bins], axis=2)
            variable = ncfile.createVariable("conc2Ice", np.float64, ("time", "size"), zlib=True)
            variable.units = "m-3"
            variable.long_name = "ice-crystal number concentration by size bin"
            variable[:] = ice2
            variable = ncfile.createVariable("concIce", np.float64, ("time",), zlib=True)
            variable.units = "m-3"
            variable.standard_name = "number_concentration_of_ice_crystals_in_air"
            variable.long_name = "ice-crystal number concentration"
            variable[:] = np.nansum(ice2, axis=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export CPI MAT time series to NetCDF")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--drop-bins", default="", help="comma-separated conc2ar axis indices")
    parser.add_argument("--ice-bins", default="", help="comma-separated conc2ar axis indices")
    parser.add_argument("--title")
    parser.add_argument("--institution", default="University of Manchester")
    parser.add_argument("--source", default="CPI-3V processed particle imagery")
    parser.add_argument("--history")
    parser.add_argument("--comment")
    args = parser.parse_args(argv)
    export_netcdf(
        args.input,
        args.output,
        _parse_bins(args.drop_bins),
        _parse_bins(args.ice_bins),
        args.title,
        args.institution,
        args.source,
        args.history,
        args.comment,
    )


# Backwards-compatible function name used by historical scripts.
def exportNetCDF(inputTSFile, outputFile, dropBins=(), iceBins=(), description=""):
    title = "CPI-3V concentration and size-distribution data"
    if description:
        title += " from " + description.strip()
    return export_netcdf(inputTSFile, outputFile, dropBins, iceBins, title=title)


if __name__ == "__main__":
    main()
