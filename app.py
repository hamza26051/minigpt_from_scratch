from pathlib import Path
import sys

project_dir = Path(__file__).resolve().parent / "MiniGPT-From-Scratch"
sys.path.insert(0, str(project_dir))

from src.api import app
