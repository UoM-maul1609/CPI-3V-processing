# MATLAB / Python parity

The maintained scientific reference is the Python pipeline under `python/proc`.
The MATLAB code is retained for inspection, reproduction of older work, and
cross-checking. Safe structural fixes have been ported, but the two particle
segmentation implementations should **not** be assumed numerically identical.

## Ported from the Python path

- ROI-to-image association no longer scans every ROI for every image marker.
- Time-series windows use explicit bin edges, including the final half-window.
- `nimages` and `deadtimes` accumulate when input files overlap an output bin.
- CPI-v1 packet/dead-time handling is available in `calcTimeseriesDriver`.
- Particle `imageType` is copied into `dat`, avoiding a later load of all ROI
  images just to perform packet selection.
- Time-series scaling avoids full 3-D `repmat` arrays.
- MATLAB statistics/time-series output replacement uses a temporary file before
  `movefile`, reducing the chance of a killed job corrupting the previous MAT.

## Deliberately not auto-ported

`matlab/proc/imageStats.m` predates the current Python segmentation algorithm.
The current Python implementation uses Otsu thresholding, repeated 3x3 median
filtering, scikit-image contour/region measurements, an eccentricity guard,
CPI-v1 strip repair, and a Python/scikit-image orientation convention. The old
MATLAB implementation uses a fixed threshold, a 5x5 median filter, MATLAB
`bwboundaries`/`regionprops`, and different coordinate/orientation conventions.

Blindly making the MATLAB code *look* like the Python code risks producing
subtly different lengths, centroids, orientations, boundaries, and therefore ML
inputs. Port this routine only with a real calibration/flight test corpus and an
old-vs-new numerical regression test. Until then, use Python-generated `dat`
for quantitative work.

## Raw reader

The Python raw reader is also the reference for CPI-v1. The MATLAB raw-reader
path remains useful for older non-v1 files but has not been rewritten to claim
full CPI-v1 parity.
