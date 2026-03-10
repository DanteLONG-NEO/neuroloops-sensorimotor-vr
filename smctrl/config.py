from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

def phase_data_dir(phase: str):
    return DATA_DIR / phase

def phase_output_dir(phase: str):
    out = OUTPUT_DIR / phase
    out.mkdir(parents=True, exist_ok=True)
    return out
