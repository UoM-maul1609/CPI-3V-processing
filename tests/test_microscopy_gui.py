"""Optional GUI regression: runs offscreen when PySide6 is available."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import cv2
import pytest

pytest.importorskip('PySide6')


def test_reference_label_gui_roundtrip(tmp_path):
    from PySide6.QtWidgets import QApplication
    from calibration.labels import LabelLibrary, ParticleLabel
    from calibration.microscopy_gui import MicroscopyDialog
    app = QApplication.instance() or QApplication([])
    lib = LabelLibrary(tmp_path/'labels.json')
    label = ParticleLabel('Test bead', length_um=25)
    lib.upsert(label)
    lib.save()
    path = tmp_path/'optical.png'
    cv2.imwrite(str(path),np.full((300,400), 150, np.uint8))
    dialog = MicroscopyDialog(lib,label)
    try:
        e = dialog.store.import_image(path,label.name)
        dialog.refresh_list(e['id'])
        assert dialog.active is not None
        dialog.active.update(mode='polygon',boundary=[[100,100],[150,100],[150,150],[100,150]])
        dialog.fit_circle()
        dialog.active['scale_line']=[[20,20],[120,20]]
        dialog.active['scale_um']=10
        dialog.persist()
        outfile=tmp_path/'annotated.png'
        dialog.canvas.export_png(outfile)
        assert outfile.exists()
        assert cv2.imread(str(outfile)).shape[:2] == (300,400)
        assert dialog.store.list_for(label.name)[0]['measurements']['max_dimension_um'] > 0
    finally:
        dialog.close()
