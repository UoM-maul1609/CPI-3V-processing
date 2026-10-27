"""Read-only single-ROI exploratory diffraction-reconstruction dialog."""
from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QDoubleSpinBox, QSlider, QPushButton, QCheckBox, QComboBox)
try:
    from .diffraction_reconstruction import normalized_intensity, propagate_flat_phase, scan, retrieve_single_plane, particle_sharpness, estimate_support, transparent_sphere_forward, bead_pattern_mismatch
except ImportError:
    from diffraction_reconstruction import normalized_intensity, propagate_flat_phase, scan, retrieve_single_plane, particle_sharpness, estimate_support, transparent_sphere_forward, bead_pattern_mismatch


class DiffractionDialog(QDialog):
    def __init__(self, image=None, background=None, pixel_um=2.46, parent=None, records=None, loader=None, initial_id=None, position_getter=None):
        super().__init__(parent)
        self.setWindowTitle('CPI diffraction experiments — transparent bead forward model')
        self.resize(1030, 760)
        self.records = list(records or [])
        self.loader = loader
        self.position_getter = position_getter
        self.position_detail = ""
        self.T = None
        if image is not None and background is not None:
            self.T = normalized_intensity(image, background)
        self.pixel = float(pixel_um) if pixel_um and np.isfinite(pixel_um) else 2.46
        self.measured_um = None
        root = QVBoxLayout(self)
        explanation = QLabel('Experimental only: propagates sqrt(background-normalised intensity) with ZERO assumed phase. '
                             'Not a recovered hologram or a calibrated CPI optical model. ' 
                             'Positive/negative propagation is relative to the recorded image, not the saved particle position.')
        explanation.setWordWrap(True)
        root.addWidget(explanation)
        if self.records:
            selection_row = QHBoxLayout()
            selection_row.addWidget(QLabel('Particle in Edge Lab selection'))
            self.particle_select = QComboBox()
            for rec in self.records:
                label = (f'{getattr(rec, "particle_id", "unknown")} · '
                         f'{getattr(rec, "label", "")}')
                self.particle_select.addItem(label, getattr(rec, 'particle_id', None))
            selection_row.addWidget(self.particle_select, 1)
            selection_row.addWidget(QLabel(f'{len(self.records)} selected'))
            root.addLayout(selection_row)
            self.particle_select.currentIndexChanged.connect(self._load_selected)
        ctrls = QHBoxLayout()
        ctrls.addWidget(QLabel('Applied propagation [mm]'))
        self.z = QDoubleSpinBox()
        self.z.setRange(-1,1); self.z.setDecimals(3); self.z.setSingleStep(.025); self.z.setValue(0)
        ctrls.addWidget(self.z)
        ctrls.addWidget(QLabel('Wavelength [nm]'))
        self.wave = QDoubleSpinBox()
        self.wave.setRange(300,1600); self.wave.setDecimals(1); self.wave.setValue(810)
        self.wave.setToolTip('Set to the CPI illumination wavelength for your instrument; 810 nm is documented for the older CPI; verify for this instrument.')
        ctrls.addWidget(self.wave)
        ctrls.addWidget(QLabel('Calibrated pixel [µm @ 5×]'))
        self.px = QDoubleSpinBox()
        self.px.setRange(.05,50); self.px.setDecimals(3); self.px.setValue(self.pixel)
        ctrls.addWidget(self.px)
        self.px.setToolTip('Object-space calibrated pixel size at the nominal 5×. Effective sampling = calibrated pixel × 5 / magnification.')
        ctrls.addWidget(QLabel('Magnification [×]'))
        self.mag = QDoubleSpinBox()
        self.mag.setRange(0.5, 30.0)
        self.mag.setDecimals(2)
        self.mag.setSingleStep(0.25)
        self.mag.setValue(5.0)
        self.mag.setToolTip('Nominal CPI magnification 5×. Changes the effective object-space pixel scale, not the propagation distance directly.')
        ctrls.addWidget(self.mag)
        ctrls.addStretch()
        root.addLayout(ctrls)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Reconstruction'))
        self.model = QComboBox()
        self.model.addItem('Flat-phase propagation (baseline)', 'flat')
        self.model.addItem('Constrained phase retrieval (experimental)', 'phase')
        self.model.addItem('Transparent spherical bead — forward model', 'bead')
        mode_row.addWidget(self.model)
        mode_row.addWidget(QLabel('Iterations'))
        self.iterations = QDoubleSpinBox()
        self.iterations.setRange(5, 80); self.iterations.setDecimals(0); self.iterations.setValue(25)
        mode_row.addWidget(self.iterations)
        mode_row.addWidget(QLabel('Support margin [px]'))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(2, 35); self.margin.setDecimals(0); self.margin.setValue(8)
        mode_row.addWidget(self.margin)
        mode_row.addStretch()
        root.addLayout(mode_row)
        bead_row = QHBoxLayout()
        bead_row.addWidget(QLabel('Bead diameter [µm]'))
        self.bead_d = QDoubleSpinBox(); self.bead_d.setRange(3, 250); self.bead_d.setDecimals(1)
        self.bead_d.setValue(40); self.bead_d.setSingleStep(1)
        bead_row.addWidget(self.bead_d)
        bead_row.addWidget(QLabel('Bead refractive index'))
        self.bead_n = QDoubleSpinBox(); self.bead_n.setRange(1.01, 2.50)
        self.bead_n.setDecimals(3); self.bead_n.setSingleStep(0.01); self.bead_n.setValue(1.50)
        bead_row.addWidget(self.bead_n)
        bead_row.addWidget(QLabel('Objective NA (approx.)'))
        self.bead_na = QDoubleSpinBox(); self.bead_na.setRange(0.01, 0.70)
        self.bead_na.setDecimals(3); self.bead_na.setSingleStep(0.01); self.bead_na.setValue(0.12)
        bead_row.addWidget(self.bead_na)
        bead_row.addStretch()
        root.addLayout(bead_row)
        self.bead_help = QLabel('Bead mode simulates a transparent sphere through an approximate coherent pupil. '
                                'This is a forward prediction, not a reconstruction of the measured image. '
                                'Diameter and refractive index are provisional; slide and real CPI aberrations are omitted.')
        self.bead_help.setWordWrap(True); root.addWidget(self.bead_help)
        self.position_label = QLabel('Measured particle position: unknown | Applied propagation: 0.000 mm')
        self.position_label.setWordWrap(True)
        root.addWidget(self.position_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-100,100); self.slider.setValue(0)
        root.addWidget(self.slider)
        self.fig = Figure(figsize=(10,6),constrained_layout=True)
        self.ax_raw, self.ax_rec, self.ax_curve = self.fig.subplots(1,3, gridspec_kw={'width_ratios':[1,1,1.2]})
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas,1)
        buttons = QHBoxLayout()
        self.scan_button = QPushButton('Scan defocus: mismatch for bead / sharpness otherwise')
        buttons.addWidget(self.scan_button)
        self.note = QLabel('0 mm shows the recorded image; sharpness maxima do not establish the physical particle plane.')
        self.note.setWordWrap(True)
        buttons.addWidget(self.note,1)
        root.addLayout(buttons)
        self.distances = None; self.scores = None
        self.z.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.wave.valueChanged.connect(self._reset_scan)
        self.px.valueChanged.connect(self._reset_scan)
        self.mag.valueChanged.connect(self._reset_scan)
        self.model.currentIndexChanged.connect(self._reset_scan)
        self.iterations.valueChanged.connect(self._reset_scan)
        self.margin.valueChanged.connect(self._reset_scan)
        self.bead_d.valueChanged.connect(self._reset_scan)
        self.bead_n.valueChanged.connect(self._reset_scan)
        self.bead_na.valueChanged.connect(self._reset_scan)
        self.scan_button.clicked.connect(self._scan)
        if self.records:
            target = next((i for i, rec in enumerate(self.records)
                           if getattr(rec, 'particle_id', None) == initial_id), 0)
            self.particle_select.setCurrentIndex(target)
            self._load_selected(target)
        elif self.T is not None:
            self._draw()

    def _load_selected(self, index):
        if index < 0 or index >= len(self.records) or self.loader is None:
            return
        try:
            rec = self.records[index]
            img, bg = self.loader(rec)
            self.T = normalized_intensity(img, bg)
            p = float(getattr(rec, 'pixel_um', 2.46))
            if np.isfinite(p) and p > 0:
                self.px.setValue(p)
            self.scores = None
            self.distances = None
            self.measured_um = None
            self.position_detail = ""
            if self.position_getter is not None:
                try:
                    value, detail = self.position_getter(rec)
                    value = float(value)
                    if np.isfinite(value):
                        self.measured_um = value * 1000.0  # signed-plane distance is in mm
                    self.position_detail = str(detail)
                except Exception as exc:
                    self.position_detail = f"Position unavailable: {exc}"
            self.z.setValue(0.0)
            self.note.setText('Signed position uses the same object-plane reference as the calibration plot. '
                              'Applied propagation remains relative to the recorded image; 0 mm = original image.')
            self._draw()
        except Exception as exc:
            self.T = None
            self.note.setText(f'Cannot load particle: {exc}')
            for ax in (self.ax_raw, self.ax_rec, self.ax_curve):
                ax.clear()
            self.canvas.draw_idle()

    def _spin_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round(float(value)*100))
        self.slider.blockSignals(False)
        self._draw()

    def _slider_changed(self, value):
        self.z.setValue(value/100.)

    def _reset_scan(self, *_):
        self.scores = None
        self._draw()

    def _effective_pixel_um(self):
        # Preserve the calibrated object-space sampling at the reference 5×.
        return self.px.value() * 5.0 / self.mag.value()

    def _scan(self):
        if self.T is None: return
        self.distances = np.linspace(-1,1,81 if self.model.currentData() != 'bead' else 41)
        self.scan_button.setEnabled(False)
        try:
            # Scanning a constrained inverse at 81 positions is costly; the
            # baseline score is shown for comparison, explicitly labelled.
            if self.model.currentData() == 'bead':
                centre = self._bead_centre()
                self.scores = np.asarray([bead_pattern_mismatch(
                    self.T, transparent_sphere_forward(self.T.shape, self.bead_d.value(), z,
                    self._effective_pixel_um(), self.wave.value(), self.bead_n.value(),
                    self.bead_na.value(), centre=centre), centre,
                    self.bead_d.value(), self._effective_pixel_um())[0] for z in self.distances])
                finite = np.isfinite(self.scores)
                if finite.any():
                    self.z.setValue(float(self.distances[np.where(finite)[0][np.argmin(self.scores[finite])]]))
            else:
                support = estimate_support(self.T, margin_px=4)
                self.scores = np.array([particle_sharpness(propagate_flat_phase(
                    self.T, z, self._effective_pixel_um(), self.wave.value()), support)
                    for z in self.distances])
            self._draw()
        finally:
            self.scan_button.setEnabled(True)

    def _bead_centre(self):
        # Keep the centre fixed across the scan to avoid fitting position to noise.
        # Infer it from smoothed image contrast, not the brightest saturated pixel.
        from scipy.ndimage import gaussian_filter
        delta = gaussian_filter(np.abs(self.T - 1.), 1.5)
        from scipy.ndimage import label
        threshold = max(.03, .25 * float(np.max(delta)))
        labels, count = label(delta > threshold)
        if count:
            weights = np.bincount(labels.ravel(), weights=delta.ravel())[1:]
            chosen = int(np.argmax(weights)) + 1
            weights2 = np.where(labels == chosen, delta, 0.)
        else:
            weights2 = delta
        yx = np.indices(delta.shape)
        total = float(weights2.sum())
        if total <= 0:
            return (self.T.shape[0]-1)/2, (self.T.shape[1]-1)/2
        return (float((yx[0]*weights2).sum()/total),
                float((yx[1]*weights2).sum()/total))

    def _draw(self):
        if self.T is None: return
        z = self.z.value()
        pos = (f'{self.measured_um:+.1f} µm ({self.measured_um / 1000:+.4f} mm) from object plane' if self.measured_um is not None
               else 'not available')
        self.position_label.setText(f'Measured particle position: {pos}   |   '
                                    f'Applied model propagation: {z:+.3f} mm from recorded image. '
                                    f'Effective sampling: {self._effective_pixel_um():.3f} µm at {self.mag.value():g}×. '
                                    'These are different coordinates. ' + self.position_detail)
        if self.model.currentData() == 'bead':
            centre = self._bead_centre()
            rec = transparent_sphere_forward(self.T.shape, self.bead_d.value(), z,
                self._effective_pixel_um(), self.wave.value(), self.bead_n.value(),
                self.bead_na.value(), centre=centre)
            residual, gain = bead_pattern_mismatch(self.T, rec, centre, self.bead_d.value(),
                                                     self._effective_pixel_um())
            method_name = f'Simulated bead, z={z:+.3f} mm'
            self.note.setText(f'Forward-model pattern RMS: {residual:.3f}; fitted contrast gain: {gain:.2f}. '
                'Positive/negative z are effective object-space defocus; sign and magnitude are not calibrated. '
                'Model omits slide, thick-lens effects and aberrations.')
        elif self.model.currentData() == 'phase' and abs(z) > 1.e-12:
            rec, retrieved_phase, residual = retrieve_single_plane(
                self.T, z, self._effective_pixel_um(), self.wave.value(),
                iterations=int(self.iterations.value()),
                support_margin_px=int(self.margin.value()))
            method_name = f'Constrained retrieval {z:+.3f} mm'
            self.note.setText('Experimental single-image constrained inversion; object support and absorption assumptions '
                              'can be invalid for transparent ice. No unique phase or physical plane is established. '
                              f'Prior residual: {residual:.3g}')
        else:
            rec = propagate_flat_phase(self.T,z,self._effective_pixel_um(),self.wave.value())
            method_name = f'Flat-phase propagation {z:+.3f} mm'
            self.note.setText('Flat-phase propagation is not genuine holographic refocusing. '
                              'Curve uses particle-local gradient energy of the baseline model, not phase-retrieval scores.')
        # Common display stretch so shifts in intensity remain interpretable.
        vmin,vmax = np.quantile(self.T,[.01,.99])
        vmin = min(float(vmin),.5); vmax = max(float(vmax),1.5)
        for ax in (self.ax_raw,self.ax_rec,self.ax_curve): ax.clear()
        self.ax_raw.imshow(self.T,cmap='gray',origin='upper',vmin=vmin,vmax=vmax,interpolation='nearest')
        self.ax_raw.set_title('Measured T (unaltered)')
        self.ax_rec.imshow(rec,cmap='gray',origin='upper',vmin=vmin,vmax=vmax,interpolation='nearest')
        self.ax_rec.set_title(method_name)
        if self.scores is not None:
            self.ax_curve.plot(self.distances,self.scores,color='#24629b')
        self.ax_curve.axvline(z,color='#cf7044',linestyle='--')
        self.ax_curve.set(xlabel='Effective propagation / defocus [mm]',
            ylabel=('Pattern mismatch (lower is better)' if self.model.currentData() == 'bead' else 'Gradient energy'),
            title=('Bead model vs measured ROI' if self.model.currentData() == 'bead' else 'Flat-phase sharpness (particle-local)'))
        for ax in (self.ax_raw,self.ax_rec):
            ax.set_xlabel('CPI pixel coordinate'); ax.set_ylabel('CPI pixel coordinate')
        self.canvas.draw_idle()
