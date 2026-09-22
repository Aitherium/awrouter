"""Import the package FROM THE TREE, the awmine/tests idiom.

Without this the suite needs `pip install -e` before it can even be collected, and
the hermetic CI gate installs nothing -- so the whole directory errored at conftest
import and ran nowhere (CGT001, measured 2026-09-22).
"""
import sys as _sys
from pathlib import Path as _Path

_PKG_ROOT = _Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))
