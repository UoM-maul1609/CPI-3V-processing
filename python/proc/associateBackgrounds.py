import numpy as np
from scipy.interpolate import interp1d

try:
    from backgroundSelection import active_mask
except ImportError:  # pragma: no cover - imported as part of a package
    from .backgroundSelection import active_mask


def associateBackgrounds(ROI_N, FULL_BG, use=None):
    """Associate each ROI with the nearest-in-time full-frame background.

    ``use``: optional use-flag per background; by default the selection in force
    (``background_selection.json`` via CPI_BG_SELECTION, see backgroundSelection.py),
    or all backgrounds if there is none.
    """
    times = np.asarray(FULL_BG["Time"][0, 0], float).ravel()
    if len(times) == 0:
        raise ValueError(
            "No background images were found. Cannot associate ROI backgrounds."
        )
    if use is None:
        use = active_mask(times)
    if use is None:
        use = np.ones(len(times), bool)
    use = np.asarray(use, bool)
    if not use.any():
        raise ValueError("No backgrounds selected. Cannot associate ROI backgrounds.")
    if not use.all():
        print(f"using {int(use.sum())} of {len(use)} backgrounds (background selection)")

    background_indices = np.flatnonzero(use)
    nearest = None
    if len(background_indices) > 1:
        nearest = interp1d(
            times[background_indices],
            background_indices,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )

    nroi = len(ROI_N["IMAGE"][0, 0][0, :])
    BG = [dict() for _ in range(nroi)]
    for i in range(nroi):
        jj = int(background_indices[0]) if nearest is None else int(nearest(ROI_N["Time"][0, 0][i, 0]))

        xl = int(ROI_N["StartX"][0, 0][0, i])
        xu = int(ROI_N["EndX"][0, 0][0, i]) + 1
        yl = int(ROI_N["StartY"][0, 0][0, i])
        yu = int(ROI_N["EndY"][0, 0][0, i]) + 1
        BG[i]["BG"] = FULL_BG[0, 0]["IMAGE"][0, jj][0, 0]["BG"][xl:xu, yl:yu]

    return BG
