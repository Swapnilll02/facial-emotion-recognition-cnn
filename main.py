"""
Root launcher for Facial Emotion Recognition using CNN.
Delegates to Facial-Emotion-Recognition-using-CNN/main.py.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "Facial-Emotion-Recognition-using-CNN"
sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    from main import main
    main()
