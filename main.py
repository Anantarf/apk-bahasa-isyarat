"""
Main entry point for SIBI Sign Language Recognition Application.
Run: python main.py
"""
import sys
import subprocess
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-fallback to local .venv python if current environment lacks dependencies
try:
    import customtkinter
except ImportError:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
        result = subprocess.run([str(venv_python), __file__] + sys.argv[1:])
        sys.exit(result.returncode)

from src.isyarat_gui import main

if __name__ == "__main__":
    main()
