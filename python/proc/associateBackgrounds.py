import numpy as np
from scipy.interpolate import interp1d


def associateBackgrounds(ROI_N, FULL_BG):
    """Associate each ROI with the nearest-in-time full-frame background."""
    times = FULL_BG["Time"][0, 0][:, 0]
    if len(times) == 0:
        raise ValueError(
            "No background images were found. Cannot associate ROI backgrounds."
        )

    background_indices = np.arange(len(times))
    nearest = None
    if len(times) > 1:
        nearest = interp1d(
            times,
            background_indices,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )

    nroi = len(ROI_N["IMAGE"][0, 0][0, :])
    BG = [dict() for _ in range(nroi)]
    for i in range(nroi):
        jj = 0 if nearest is None else int(nearest(ROI_N["Time"][0, 0][i, 0]))

        xl = int(ROI_N["StartX"][0, 0][0, i])
        xu = int(ROI_N["EndX"][0, 0][0, i]) + 1
        yl = int(ROI_N["StartY"][0, 0][0, i])
        yu = int(ROI_N["EndY"][0, 0][0, i]) + 1
        BG[i]["BG"] = FULL_BG[0, 0]["IMAGE"][0, jj][0, 0]["BG"][xl:xu, yl:yu]

    return BG
