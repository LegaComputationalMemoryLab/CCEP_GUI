import py_compile
from pathlib import Path


def test_app_py_compiles():
    app_path = Path(__file__).resolve().parents[1] / "src" / "ieeg_ccep_analyzer" / "app.py"
    py_compile.compile(str(app_path), doraise=True)
