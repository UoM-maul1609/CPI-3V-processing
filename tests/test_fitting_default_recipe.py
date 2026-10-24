"""Startup fitting recipe must not silently select the experimental circle method.

Source/AST check is deliberate: this test runs without PySide6, which is optional
in the core processor environment.
"""
import ast
from pathlib import Path


def test_dialog_starts_with_established_fit():
    root = Path(__file__).resolve().parents[1]
    gui = root / 'python' / 'calibration' / 'gui.py'
    tree = ast.parse(gui.read_text())
    dialog = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'SizingDialog')
    init = next(f for f in dialog.body if isinstance(f, ast.FunctionDef) and f.name == '__init__')
    choices = []
    for node in ast.walk(init):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (node.func.attr == 'addItem' and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'preset' and len(node.args) >= 2):
            choices.append((node.lineno, ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1])))
    recipes = [c[2] for c in sorted(choices)]
    assert recipes == ['original', 'contour', 'circle']
    assert '(experimental)' in sorted(choices)[-1][1]
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == '_preset_changed' for node in ast.walk(init))
