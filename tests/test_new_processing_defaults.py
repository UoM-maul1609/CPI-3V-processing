"""Production recipe defaults without changing legacy explicit options."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python' / 'proc'))
import batchProcess
from edgeDetect import EdgeOptions

def test_production_defaults_contour_focus_circle():
    d = batchProcess.edge_options
    assert d['method'] == 'transmittance'
    assert d['edge_mode'] == 'relative'
    assert d['edge_fraction'] == .55
    assert d['outline'] == 'radial'
    assert d['shadow_circle_padding'] == 8.0
    # Old records retain their explicitly recorded options.
    assert EdgeOptions(method='legacy').method == 'legacy'

def test_cli_exposes_disable_and_override(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['batchProcess.py', '/tmp', '--no-shadow-circle'])
    args = batchProcess._parse_args()
    assert args.no_shadow_circle
    monkeypatch.setattr(sys, 'argv', ['batchProcess.py', '/tmp', '--shadow-circle-padding', '6'])
    args = batchProcess._parse_args()
    assert args.shadow_circle_padding == 6
    assert not args.no_shadow_circle
