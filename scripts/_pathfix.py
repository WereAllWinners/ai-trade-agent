"""
_pathfix.py — sys.path bootstrap for scripts that live in subdirectories.

Import this at the top of any script inside agents/, analysis/, data/,
training/, or tools/.  It puts scripts/ root and all subdirectories onto
sys.path so every module can be imported with a plain 'import <name>'
regardless of which subdirectory the caller lives in.

Usage (add after the stdlib imports, before any project imports):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import _pathfix  # noqa: F401
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_SUBDIRS = ('agents', 'analysis', 'data', 'training', 'tools')

for _p in [_SCRIPTS] + [_SCRIPTS / d for d in _SUBDIRS]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
