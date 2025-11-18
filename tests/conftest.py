"""Pytest hooks for LinearNexus."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))
