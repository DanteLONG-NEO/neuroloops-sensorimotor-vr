from pathlib import Path
from smctrl.config import DATA_DIR, OUTPUT_DIR

class Phase:
    def __init__(self, name: str):
        self.name = name
        self.data_dir = DATA_DIR / name
        self.output_dir = OUTPUT_DIR / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modality_freq_dict={
            "DART":90,
            "HMD":90,
            "FINGER":120,
            "EYE": 200,
            "EMG": 250,
            "EEG": 250
        }
        self.all_session_order  = None
        self.all_total_trials   = None
        self.all_acc_trials     = None
        self.board_radius = 0.256
        self.cond_order = [1, 3, 5, 2, 4, 6]