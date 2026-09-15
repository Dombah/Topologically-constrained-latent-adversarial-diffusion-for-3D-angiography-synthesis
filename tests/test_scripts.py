"""Every entry script imports cleanly and parses its arguments (catches broken imports after edits).

The real end-to-end check is each training script's --smoke flag; see Instructions.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_runs(script):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True,
                            text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
