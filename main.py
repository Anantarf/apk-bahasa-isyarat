"""
Main entry point for SIBI Sign Language Recognition Application.
Run: python main.py
"""
import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.isyarat_gui import main

if __name__ == "__main__":
    main()
