"""Focus plots may not be changed by a shadow integration experiment."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from calibration.edge_preview import trial_focus_value


def test_shadow_reuses_saved_focus_even_when_integral_is_invalid():
    record = SimpleNamespace(focus=37.1, focus_T=.291, cpv_focus=7.2, cpv_fwhm_px=2.3)
    shadow = {'overlay_kind': 'integration circle', 'focus': float('nan'), 'focus_T': float('nan')}
    edge = {'focus': 3.2, 'focus_T': .08}
    for metric in ('focus', 'focus_T'):
        expected = getattr(record, metric)
        assert trial_focus_value(record, None, metric) == pytest.approx(expected)
        assert trial_focus_value(record, shadow, metric) == pytest.approx(expected)
        assert trial_focus_value(record, edge, metric) == pytest.approx(expected)
    assert trial_focus_value(record, shadow, 'cpv_focus') == 7.2
    assert trial_focus_value(record, {'cpv_focus': 6.3}, 'cpv_focus') == 6.3
