#!/usr/bin/env python3
"""Compatibility wrapper for evaluation 1 normalized PPL."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "evalution1_normalized_ppl" / "eval_normalized_ppl.py"
runpy.run_path(str(TARGET), run_name="__main__")
