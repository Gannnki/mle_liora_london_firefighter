"""Compatibility shim for historical Streamlit scripts and pickles.

The implementation lives in src.FeatureEngineering, shared with training.
"""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)

from src.FeatureEngineering import FeatureEncoder, FeatureScaler  # noqa: E402,F401
