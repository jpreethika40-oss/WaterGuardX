"""
WaterGuardX — Streamlit Community Cloud Root Entrypoint.
Enables seamless 1-click deployment on Streamlit Cloud without manual path configuration.
"""
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
APP_SCRIPT = APP_DIR / "app.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Execute the main application
runpy.run_path(str(APP_SCRIPT), run_name="__main__")
