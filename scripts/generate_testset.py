#!/usr/bin/env python3
"""Thin CLI wrapper for governed RAGAS synthetic benchmark generation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from eval.testset_generation import main


if __name__ == "__main__":
    raise SystemExit(main())
