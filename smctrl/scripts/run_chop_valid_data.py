from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime

# rearange result data

def run_corr(phase, sub_nums:list, modality:dict):
    required_keys = ["IsHand", "IsFinger", "IsEye", "IsEMG", "IsEEG", "IsDart"]
    assert all(k in modality for k in required_keys), \
        f"Missing keys: {[k for k in required_keys if k not in modality]}"
    
    base_dir = Path(phase.data_dir)

    for sub in sub_nums:
        print(f"sub{sub}:")
        file_order = 0

        def extract_datetime(path):
            ts = path.stem.replace("corr_result_", "")
            return datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")

        results = sorted(
            base_dir.glob(f"sub{sub}_*/corr_result_*.csv"),
            key=extract_datetime
        )

        for result in results:
            # two must have file: markers and results
            df_result = pd.read_csv(result)
            marker_files = list(result.parent.glob("corr_marker_*.csv"))
            df_marker = pd.read_csv(marker_files[0])

            # optional modalities
            dart_files = list(result.parent.glob("corr_dart_*.csv"))
            hmd_files = list(result.parent.glob("corr_hmd_*.csv"))
            emg_files = list(result.parent.glob("emg_data_*.csv"))
            eeg_files = list(result.parent.glob("eeg_data_*.csv"))
            eye_files = list(result.parent.glob("eye_*.csv"))
            finger_files = list(result.parent.glob("hand_position_*.csv"))

            if modality["IsDart"] and dart_files:
                dart_file = dart_files[0]
                df_dart = pd.read_csv(dart_file)
            else:
                df_dart = pd.DataFrame()
            
            if modality["IsHand"] and hmd_files:
                hmd_file = hmd_files[0]
                df_hmd = pd.read_csv(hmd_file)
            else:
                df_hmd = pd.DataFrame()

            if modality["IsFinger"] and finger_files:
                finger_file = finger_files[0]
                df_finger = pd.read_csv(finger_file)
            else:
                df_finger = pd.DataFrame()
            
            if modality["IsEye"] and eye_files:
                eye_file = eye_files[0]
                df_eye = pd.read_csv(eye_file)
            else:
                df_eye = pd.DataFrame()
            
            if modality["IsEEG"] and eeg_files:
                eeg_file = eeg_files[0]
                df_eeg = pd.read_csv(eeg_file)
            else:
                df_eeg = pd.DataFrame()
            
            if modality["IsEMG"] and emg_files:
                emg_file = emg_files[0]
                df_emg = pd.read_csv(emg_file)
            else:
                df_emg = pd.DataFrame()          

            for cond, group in df_result.groupby("Condition"):
                print(f"processing: {cond}")
                # Bounds for dart/HMD/marker
                start_hit = group.iloc[0]["TotalHitTimes"]
                end_hit   = group.iloc[-1]["TotalHitTimes"]

                if df_marker is not None and not df_marker.empty:
                    conds_start_time = df_marker.loc[df_marker["TotalHitTimes"] == start_hit, "LocalTime"].iloc[0]
                    conds_end_time = df_marker.loc[df_marker["TotalHitTimes"] == end_hit, "LocalTime"].iloc[-1]

                    # UnixTime bounds
                    conds_start_unix = df_marker.loc[df_marker["TotalHitTimes"] == start_hit, "UnixTime(ms)"].iloc[0]
                    conds_end_unix = df_marker.loc[df_marker["TotalHitTimes"] == end_hit, "UnixTime(ms)"].iloc[-1]

                    marker_cond = df_marker[
                        (df_marker["TotalHitTimes"] >= start_hit) &
                        (df_marker["TotalHitTimes"] <= end_hit)
                    ].copy()
                    marker_cond["Condition"] = cond

                if df_dart is not None and not df_dart.empty:
                    dart_cond = df_dart[
                        (df_dart["LocalTime"] >= conds_start_time) & 
                        (df_dart["LocalTime"] <= conds_end_time)
                    ].copy()
                    dart_cond["Condition"] = cond

                if df_hmd is not None and not df_hmd.empty:
                    hmd_cond = df_hmd[
                        (df_hmd["LocalTime"] >= conds_start_time) & 
                        (df_hmd["LocalTime"] <= conds_end_time)
                    ].copy()
                    hmd_cond["Condition"] = cond

                # Eye slicing by 'Capture Unix timestamp' (ns) converted to ms to match df_result
                if not df_eye.empty and "Capture Unix timestamp" in df_eye.columns:
                    eye_unix_ns = pd.to_numeric(df_eye["Capture Unix timestamp"], errors="coerce")
                    eye_time_ms = (eye_unix_ns // 1_000_000).astype("Int64")
                    mask = ((eye_time_ms >= conds_start_unix) & (eye_time_ms <= conds_end_unix)).fillna(False)
                    eye_cond = df_eye.loc[mask].copy()
                    eye_cond["Condition"] = cond
                else:
                    eye_cond = pd.DataFrame()

                # Hand position slicing by 'unix_time' (ns) converted to ms to match df_result
                if not df_finger.empty and "unix_time" in df_finger.columns:
                    hand_unix_ns = pd.to_numeric(df_finger["unix_time"], errors="coerce")
                    hand_time_ms = (hand_unix_ns // 1_000_000).astype("Int64")
                    mask = ((hand_time_ms >= conds_start_unix) & (hand_time_ms <= conds_end_unix)).fillna(False)
                    finger_cond = df_finger.loc[mask].copy()
                    finger_cond["Condition"] = cond
                else:
                    finger_cond = pd.DataFrame()

                # EMG slicing
                if not df_emg.empty and "Timestamp(ms)" in df_emg.columns:
                    emg_unix_ms = pd.to_numeric(df_emg["Timestamp(ms)"], errors="coerce")
                    mask = ((emg_unix_ms >= conds_start_unix) & (emg_unix_ms <= conds_end_unix)).fillna(False)
                    emg_cond = df_emg.loc[mask].copy()
                    emg_cond["Condition"] = cond
                else:
                    emg_cond = pd.DataFrame()

                # EEG slicing
                if not df_eeg.empty and "Timestamp(ms)" in df_eeg.columns:
                    eeg_unix_ms = pd.to_numeric(df_eeg["Timestamp(ms)"], errors="coerce")
                    mask = ((eeg_unix_ms >= conds_start_unix) & (eeg_unix_ms <= conds_end_unix)).fillna(False)
                    eeg_cond = df_eeg.loc[mask].copy()
                    eeg_cond["Condition"] = cond
                else:
                    eeg_cond = pd.DataFrame()

                # Outputs for dart/HMD/marker
                out_dir = result.parent.parent / f"1_chopped/sub{sub}_"
                out_dir.mkdir(parents=True, exist_ok=True)
                dart_cond.to_csv(out_dir / f"dart_order{file_order}_cond{file_order}.csv", index=False)
                hmd_cond.to_csv(out_dir / f"hmd_order{file_order}_cond{file_order}.csv", index=False)
                marker_cond.to_csv(out_dir / f"marker_order{file_order}_cond{file_order}.csv", index=False)
                group.to_csv(out_dir / f"result_order{file_order}_cond{file_order}.csv", index=False)

                if not eye_cond.empty:
                    eye_cond.to_csv(out_dir / f"eye_order{file_order}_cond{file_order}.csv", index=False)
                if not emg_cond.empty:
                    emg_cond.to_csv(out_dir / f"emg_order{file_order}_cond{file_order}.csv", index=False)
                if not eeg_cond.empty:
                    eeg_cond.to_csv(out_dir / f"eeg_order{file_order}_cond{file_order}.csv", index=False)
                if not finger_cond.empty:
                    finger_cond.to_csv(out_dir / f"finger_order{file_order}_cond{file_order}.csv", index=False)

                file_order+=1