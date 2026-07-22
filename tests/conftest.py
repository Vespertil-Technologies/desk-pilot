"""
Put the repo root on sys.path so the suite runs from a fresh clone.

The modules are top-level (see py-modules in pyproject.toml), so without this
`pytest` only works once the package has been installed, and it breaks again if
an editable install is left pointing at an old checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
