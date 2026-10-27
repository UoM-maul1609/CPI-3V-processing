"""Experimental coherent propagation of a single background-normalised CPI ROI.

This is NOT quantitative holographic phase retrieval. Only measured amplitude is
known; an initially flat phase is assumed. Imaging optics, slide and pupil are
not inverted, so signed distance is an effective *object-space* parameter.
Nothing in this module modifies accepted particles or stored images.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes, label


def normalized_intensity(image, background):
    a = np.asarray(image, float)
    b = np.asarray(background, float)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError('ROI and background must be matching 2-D arrays')
    t = a / np.maximum(b, 1.0)
    border = np.concatenate((t[0], t[-1], t[1:-1, 0], t[1:-1, -1]))
    scale = float(np.median(border)) if len(border) else 1.0
    if np.isfinite(scale) and scale > 0.2:
        t /= scale
    return np.clip(np.nan_to_num(t, nan=1., posinf=1., neginf=0.), 0., None)


def propagate_flat_phase(transmission, distance_mm, pixel_um=2.46, wavelength_nm=810., pad_factor=2):
    """Angular-spectrum propagation of sqrt(T), constant-phase input.

    Pixels are in camera-index order (first and second axes), with equal
    object-space sampling. Positive/negative propagation is an effective model,
    NOT a recovered particle-plane intensity or physical CPI optical inversion.
    """
    T = np.asarray(transmission, dtype=float)
    if T.ndim != 2 or min(T.shape) < 4 or not np.isfinite(T).all():
        raise ValueError('Transmission must be a finite 2-D ROI')
    if pixel_um <= 0 or wavelength_nm <= 0 or pad_factor < 1:
        raise ValueError('Positive pixel size, wavelength and padding required')
    wave_um = wavelength_nm * 1e-3
    z_um = distance_mm * 1000.
    # Embed the contrast perturbation in a unity-valued plane wave, to avoid
    # hard aperture edges in the FFT. Preserves unmodified signed transmission.
    shape = (int(T.shape[0]*pad_factor), int(T.shape[1]*pad_factor))
    field = np.ones(shape, dtype=complex)
    start = tuple((m-n)//2 for m,n in zip(shape,T.shape))
    slices = tuple(slice(a, a+n) for a,n in zip(start,T.shape))
    field[slices] = np.sqrt(np.maximum(T,0.))
    if z_um == 0:
        return T.copy()
    fy = np.fft.fftfreq(shape[0], d=pixel_um)
    fx = np.fft.fftfreq(shape[1], d=pixel_um)
    f2 = fy[:,None]**2 + fx[None,:]**2
    # Remove global carrier phase; retain Fresnel diffraction phase. For CPI
    # sampling the propagating band dominates; evanescent spectrum discarded.
    transfer = np.exp(1j * z_um * 2*np.pi/wave_um *
                      (np.sqrt(np.maximum(1-wave_um**2*f2,0.))-1.))
    transfer = np.where(wave_um**2*f2 <= 1, transfer, 0.)
    result = np.fft.ifft2(np.fft.fft2(field)*transfer)
    return np.abs(result[slices])**2


def edge_sharpness(intensity):
    """Exploratory gradient energy, not a validated best-focus estimator."""
    t = gaussian_filter(np.asarray(intensity,float), 0.5)
    grad = np.gradient(t)
    return float(np.mean(sum(g*g for g in grad)))


def scan(transmission, distances_mm, pixel_um=2.46, wavelength_nm=810.):
    scores = []
    for z in distances_mm:
        scores.append(edge_sharpness(propagate_flat_phase(
            transmission,float(z),pixel_um,wavelength_nm)))
    return np.asarray(scores)


def _propagate_field(field, distance_mm, pixel_um, wavelength_nm):
    """FFT Fresnel/angular-spectrum propagation of a complex field on the object-space grid.

    A constant incident reference field remains constant. The field must be
    padded with a unity reference by the caller to limit wrap-around.
    """
    wave_um = float(wavelength_nm) * 1e-3
    z_um = float(distance_mm) * 1000.0
    if z_um == 0:
        return np.asarray(field, complex).copy()
    shape = field.shape
    f0 = np.fft.fftfreq(shape[0], d=pixel_um)[:, None]
    f1 = np.fft.fftfreq(shape[1], d=pixel_um)[None, :]
    arg = 1.0 - wave_um**2 * (f0**2 + f1**2)
    transfer = np.exp(1j * (2*np.pi/wave_um) * z_um *
                      (np.sqrt(np.maximum(arg, 0.0))-1.0))
    transfer[arg < 0] = 0
    return np.fft.ifft2(np.fft.fft2(field)*transfer)


def estimate_support(transmission, margin_px=8):
    """A deliberately loose *hypothesised* object support, not a measured boundary."""
    t = np.asarray(transmission, float)
    smoothed = gaussian_filter(np.abs(t - 1.0), sigma=1.5)
    # Threshold relative to variations inside ROI; exclude weak background noise.
    q = float(np.quantile(smoothed, 0.95))
    mask = smoothed > max(0.025, 0.12 * q)
    labels, count = label(mask)
    if count:
        areas = np.bincount(labels.ravel())[1:]
        # We do not assume that a crystal must be a single dark component.
        keep = areas >= max(3, 0.04 * areas.max())
        mask = np.isin(labels, np.flatnonzero(keep) + 1)
    mask = binary_fill_holes(mask)
    if margin_px:
        mask = binary_dilation(mask, iterations=int(margin_px))
    return mask


def retrieve_single_plane(transmission, distance_mm, pixel_um=2.46,
                          wavelength_nm=810., iterations=25, support_margin_px=8,
                          pad_factor=2):
    """Experimental constrained single-intensity-plane inline phase retrieval.

    Reconstructs a possible object-plane intensity by alternating measured
    camera amplitude with a weak prior: transmission equals the reference
    outside an automatically estimated, deliberately loose object support.
    Inside support complex phase is permitted but amplitude may not exceed 1.
    This is an assumption, not a unique solution; nonabsorbing refractors and
    multiple particles can violate the constraints. No absolute phase recovery
    or calibration claims are made.
    """
    T = np.asarray(transmission, float)
    if T.ndim != 2 or min(T.shape) < 4 or not np.isfinite(T).all():
        raise ValueError('Transmission must be a finite 2-D ROI')
    if not 0 < pixel_um or not 0 < wavelength_nm or not 1 <= pad_factor or iterations < 1:
        raise ValueError('Invalid phase retrieval settings')
    if float(distance_mm) == 0:
        return T.copy(), np.zeros_like(T), 0.
    shape = tuple(int(np.ceil(n * pad_factor)) for n in T.shape)
    starts = tuple((n - m)//2 for n,m in zip(shape,T.shape))
    crop = tuple(slice(j,j+n) for j,n in zip(starts,T.shape))
    measured = np.ones(shape, float)
    measured[crop] = np.sqrt(np.maximum(T, 0))
    supp = np.zeros(shape, bool)
    supp[crop] = estimate_support(T, support_margin_px)
    # Initial camera field has zero phase. Iterate between camera and candidate
    # object planes; each iteration imposes the measured intensity at camera.
    camera_field = measured.astype(complex)
    for _ in range(int(iterations)):
        obj = _propagate_field(camera_field, distance_mm, pixel_um, wavelength_nm)
        obj[~supp] = 1.0 + 0.0j
        amps = np.abs(obj[supp])
        too_large = amps > 1.0
        constrained = obj[supp]
        constrained[too_large] /= amps[too_large]
        obj[supp] = constrained
        camera_field = _propagate_field(obj, -distance_mm, pixel_um, wavelength_nm)
        # Hold camera measurement exactly; reference phase is not observed.
        camera_field *= measured / np.maximum(np.abs(camera_field), 1.e-12)
    obj = _propagate_field(camera_field, distance_mm, pixel_um, wavelength_nm)
    out = obj[crop]
    # Residual measures violation of the assumed support and amplitude priors;
    # it is NOT evidence that the recovered image is physically correct.
    residual = float(np.mean(np.abs(obj[~supp] - 1.0)**2)) if (~supp).any() else 0.
    return np.abs(out)**2, np.angle(out), residual


def particle_sharpness(intensity, support=None):
    """Gradient sharpness near a hypothesised particle, excluding most ROI background."""
    t = gaussian_filter(np.asarray(intensity, float), 0.5)
    grad = np.gradient(t)
    energy = sum(g*g for g in grad)
    if support is None:
        support = estimate_support(intensity, margin_px=4)
    return float(np.mean(energy[support])) if np.any(support) else float(np.mean(energy))


def transparent_sphere_forward(shape, diameter_um, distance_mm, pixel_um=2.46,
                               wavelength_nm=810., refractive_index=1.50,
                               numerical_aperture=0.12, centre=None, pad_factor=3):
    """Coherent thin-phase spherical bead, propagated to a CPI-like pupil-limited image.

    Returns reference-normalised *intensity*, not a recovered hologram. z=0 is
    nominal best image plane, and +/- z represent effective object-space defocus.
    The scalar, paraxial thin-object model neglects glass-slide coupling, thick
    sphere refraction, polarization, aberrations, and the real CPI pupil shape.
    """
    ny, nx = (int(shape[0]), int(shape[1]))
    if min(ny, nx) < 4 or not (0 < diameter_um and 0 < pixel_um and 0 < wavelength_nm):
        raise ValueError('Positive diameter, pixel size and wavelength required')
    if not (1 < refractive_index < 3 and 0 < numerical_aperture < 1):
        raise ValueError('Refractive index must exceed 1; NA must lie in (0,1)')
    large = (ny * int(pad_factor), nx * int(pad_factor))
    off = ((large[0] - ny)//2, (large[1] - nx)//2)
    crop = (slice(off[0], off[0]+ny), slice(off[1], off[1]+nx))
    if centre is None:
        centre = ((ny-1)/2, (nx-1)/2)
    cy, cx = float(centre[0]) + off[0], float(centre[1]) + off[1]
    yy = (np.arange(large[0], dtype=float)-cy)*pixel_um
    xx = (np.arange(large[1], dtype=float)-cx)*pixel_um
    r2 = yy[:, None]**2 + xx[None, :]**2
    radius = diameter_um / 2.
    thickness = 2. * np.sqrt(np.maximum(radius*radius-r2, 0.))
    wave_um = wavelength_nm * 1e-3
    phase = (2*np.pi/wave_um)*(refractive_index-1.)*thickness
    field = np.exp(1j * phase)
    fy = np.fft.fftfreq(large[0], pixel_um)[:, None]
    fx = np.fft.fftfreq(large[1], pixel_um)[None, :]
    f2 = fy*fy + fx*fx
    z_um = distance_mm * 1000.
    fresnel = np.exp(-1j*np.pi*wave_um*z_um*f2)
    # Soft-edge approximation to the unknown actual CPI coherent pupil.
    fc = numerical_aperture / wave_um
    f = np.sqrt(f2)
    transition = max(fc * .07, 1.e-8)
    pupil = 1. / (1. + np.exp(np.clip((f-fc)/transition, -60, 60)))
    # Ensure the DC reference is transmitted at exactly unit amplitude.
    pupil /= float(pupil[0, 0])
    result = np.fft.ifft2(np.fft.fft2(field) * fresnel * pupil)
    return np.abs(result[crop])**2


def bead_pattern_mismatch(measured, simulated, centre=None, diameter_um=40.,
                          pixel_um=2.46):
    """Compare contrast patterns using a background-masked weighted RMS.

    A fitted multiplicative contrast scale accommodates unknown optical contrast
    without pretending that its value is physically calibrated. No offsets are fit.
    """
    obs = np.asarray(measured, float) - 1.
    pred = np.asarray(simulated, float) - 1.
    if centre is None:
        centre = ((obs.shape[0]-1)/2, (obs.shape[1]-1)/2)
    rr, cc = np.indices(obs.shape)
    radius_px = diameter_um / (2*pixel_um)
    distance = np.hypot(rr-centre[0], cc-centre[1])
    mask = distance <= max(9., 2.5 * radius_px)
    if not mask.any():
        return float('nan'), 1.
    a = pred[mask]
    b = obs[mask]
    denom = float(np.dot(a, a))
    gain = max(0., float(np.dot(a,b)) / denom) if denom > 1.e-10 else 0.
    rms = float(np.sqrt(np.mean((gain*a-b)**2)))
    return rms, gain
