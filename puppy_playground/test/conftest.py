"""Run against the repo's own puppy_motion source: with the repo mounted at the working directory,
its top-level puppy_motion/ folder would otherwise shadow the installed package as a bare
namespace package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "puppy_motion"))
