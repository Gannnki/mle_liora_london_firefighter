"""Launch Streamlit with a stable working directory on Windows/Linux/macOS."""
import os
from pathlib import Path
import subprocess
import sys

app_dir = Path(__file__).resolve().parents[1] / "src/display_streamlit"
if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", *sys.argv[1:]],
        cwd=app_dir,
        env={**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"},
    ))
