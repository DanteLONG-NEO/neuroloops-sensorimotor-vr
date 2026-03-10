from pathlib import Path
import pandas as pd
import numpy as np
import os
from smctrl import HELPER, phase, condition_mapping, kinematics
import struct

# rearange result data

def run_corr(phase, sub_nums:list, condition_mapper:condition_mapping.CONDITION_MAPPING, modality:dict):
    required_keys = ["IsHand", "IsFinger", "IsEMG", "IsEEG", "IsDart"]
    assert all(k in modality for k in required_keys), \
        f"Missing keys: {[k for k in required_keys if k not in modality]}"
    
    base_dir = Path(phase.data_dir)
    # rearange result data

    for sub in sub_nums:
        print(f"sub {sub}:")
        csv_files = base_dir.glob(f"sub{sub}_*/result_*.csv")

        df_list = []
        for f in csv_files:
            # marker file must have
            marker_file = list(f.parent.glob("marker_*.csv"))[0]

            # pre-create file holders
            dart_file = None
            hmd_file = None
            emg_file = None
            eeg_file = None
            release_df = pd.DataFrame()

            # extract necessary files
            if modality["IsDart"] and list(f.parent.glob("dart_*.csv")):
                dart_file = list(f.parent.glob("dart_*.csv"))[0]
            if modality["IsHand"] and list(f.parent.glob("hmd_*.csv")):
                hmd_file = list(f.parent.glob("hmd_*.csv"))[0]
            if modality["IsEMG"] and list(f.parent.glob("hmd_*.csv")):
                emg_file = list(f.parent.glob("emg_data_*.bin"))[0]
            if modality["IsEEG"] and list(f.parent.glob("hmd_*.csv")):
                eeg_file = list(f.parent.glob("eeg_data_*.bin"))[0]

            # ====== correct marker destroyed time ======
            df_result = pd.read_csv(f, header=0)
            df_result["source_folder"] = f.parent.name
            df_result["Subject"] = int(f.parent.name.split("_")[0].replace("sub", ""))
            df_result["Condition"] = df_result.apply(condition_mapper.map_condition, axis=1)

            df_marker = pd.read_csv(marker_file)
            df_marker.loc[df_marker["Marker"].isin(["Low Destroyed", "High Destroyed"]), "UnixTime(ms)"] -= 1000
            df_marker.loc[df_marker["Marker"].isin(["Low Destroyed", "High Destroyed"]), "LocalTime"] -= 1.0

            # 1) Found duplicate Hit
            duplicate_hit_mask = (
                (df_marker["Marker"] == "Hit") &
                (df_marker["Marker"].shift(1) == "Hit")
            )
            duplicate_hit_times = df_marker.loc[duplicate_hit_mask, "TotalHitTimes"].to_numpy()

            # 2) Modify the TotalHitTimes collumn of marker file
            df_marker["TotalHitTimes"] -= duplicate_hit_mask.cumsum()

            # 3) Delete duplicate TotalHitTimes collumn in result files
            if duplicate_hit_times.size > 0:
                df_result = df_result[~df_result["TotalHitTimes"].isin(duplicate_hit_times)].reset_index(drop=True)

                hit_array = np.sort(duplicate_hit_times)
                correction = (df_result["TotalHitTimes"].to_numpy()[:, None] > hit_array).sum(axis=1)
                df_result["TotalHitTimes"] -= correction

            # 5) Delete duplicate markers
            df_marker = df_marker.loc[df_marker["Marker"].shift(-1) != df_marker["Marker"]].reset_index(drop=True)

            # ====== correct marker and result hit target times ======
            result_hit_target_mask = (df_result["HitObjectTag"] == "Target")
            df_result["ChangeCount"] = result_hit_target_mask.cumsum()

            # 6) Map changeCount
            cc_map = df_result.drop_duplicates("TotalHitTimes").set_index("TotalHitTimes")["ChangeCount"]
            df_marker["ChangeCount"] = df_marker["TotalHitTimes"].map(cc_map)

            outpath = dart_file.parent / ("corr_" + marker_file.name)
            df_marker.to_csv(outpath, index=False)
            
            outpath = f.parent / ("corr_" + f.name)
            # print(outpath)
            df_result.to_csv(outpath, index=False)
            df_list.append(df_result)

            #  ====== dart file velocity and max pooling ======
            df_dart = pd.read_csv(dart_file)
            destroy_cond = ["Hit", "Low Destroyed", "High Destroyed"]
            pinch_start_time = df_marker.loc[df_marker["Marker"] == "Pinched", ["UnixTime(ms)", "LocalTime"]]
            pinch_end_time   = df_marker.loc[df_marker["Marker"].isin(destroy_cond), ["UnixTime(ms)", "LocalTime"]]
            print(f"pinch start time count:{pinch_start_time.shape[0]}")
            print(f"pinch end time count:{pinch_end_time.shape[0]}")
            if (pinch_start_time.shape != pinch_end_time.shape):
                raise ValueError("pinch start time shape doesn't match with end time")
            
            df_dart_corr = []
            df_dart_release = pd.DataFrame(columns=["TotalHitTimes","RPX","RPY","RPZ"])
            j = 0

            for i in range(pinch_start_time.shape[0]):
                # extract start, end time from marker
                start_time, end_time = pinch_start_time.iloc[i]["UnixTime(ms)"], pinch_end_time.iloc[i]["UnixTime(ms)"]
                # seperate indv dart
                df_indv_dart = df_dart[(df_dart["UnixTime(ms)"] >= start_time) & (df_dart["UnixTime(ms)"] <= end_time)].copy()
                # velocity calculation
                kinematics.calculate_naive_velocity(df_indv_dart, "DartPosition_", "DartNaiveVelocity")
                kinematics.maxpool_past_velocity(df_indv_dart, 5, "DartNaiveVelocity")
                if "Released" in df_indv_dart["Marker"].unique():
                    release_row = df_indv_dart.loc[df_indv_dart["Marker"] == "Released", 
                                ["TotalHitTimes",
                                "DartPosition_X", "DartPosition_Y", "DartPosition_Z"
                                ]].iloc[0]

                    df_dart_release.loc[j, ["TotalHitTimes","RPX", "RPY", "RPZ"]] = release_row.values
                    j += 1

                df_dart_corr.append(df_indv_dart)

            df_dart_corr = pd.concat(df_dart_corr, ignore_index=True)

            df_dart_corr = df_dart_corr.drop(
                columns=["ChangeCount", "CaseCount", "TotalHitTimes"],
                errors="ignore"
            )
            outpath = dart_file.parent / ("corr_" + dart_file.name)
            df_dart_corr.to_csv(outpath, index=False)

            # ====== Emg file bin to csv ========
            if emg_file:
                with open(emg_file, 'rb') as fh:
                    emg_data_parsed = np.array(list(struct.iter_unpack('d'*11, fh.read())))[10:, :]
                    # print(emg_data_parsed.shape)

                emg_record_timestamp_ms = emg_data_parsed[:, 10]

                emg_data = emg_data_parsed[:, 0:8]
                emg_device_timestamp_ms = emg_data_parsed[:, 8] * 1e3
                emg_recv_timestamp_ms = emg_data_parsed[:, 9]
                emg_record_timestamp_ms = emg_data_parsed[:, 10]

                ## Adjusting the timestamp for both the stimuli and the emg system
                # y = ax + b
                y = emg_record_timestamp_ms
                x = np.vstack((emg_device_timestamp_ms, np.ones(len(emg_device_timestamp_ms)))).T
                (a, b) = np.linalg.lstsq(x, y, rcond=None)[0]
                emg_converted_record_timestamp_ms = emg_device_timestamp_ms * a + b

                columns = ["Raw Timestamp(ms)"] + ["Timestamp(ms)"] + [f"Ch{i+1}" for i in range(emg_data.shape[1])]

                df_emg = pd.DataFrame(
                    np.column_stack((emg_record_timestamp_ms, emg_converted_record_timestamp_ms, emg_data)),
                    columns=columns
                )
                outpath = dart_file.parent / (emg_file.name.replace(".bin",".csv"))
                df_emg.to_csv(outpath, index=False)

            # ====== EEG file bin to csv ========
            if eeg_file:
                if os.path.getsize(eeg_file) > 0:
                    with open(eeg_file, 'rb') as fh:
                        raw_bytes = fh.read()
                        if len(raw_bytes) == 0:
                            print(f"[Warning] EEG file {eeg_file} is empty, skipping.")
                        else:
                            eeg_data_parsed = np.array(list(struct.iter_unpack('d'*11, raw_bytes)))[10:, :]

                            eeg_record_timestamp_ms = eeg_data_parsed[:, 10]
                            eeg_data = eeg_data_parsed[:, 0:8]
                            eeg_device_timestamp_ms = eeg_data_parsed[:, 8] * 1e3
                            eeg_recv_timestamp_ms = eeg_data_parsed[:, 9]
                            eeg_record_timestamp_ms = eeg_data_parsed[:, 10]

                            # Adjusting the timestamp: y = a*x + b
                            y = eeg_record_timestamp_ms
                            x = np.vstack((eeg_device_timestamp_ms, np.ones(len(eeg_device_timestamp_ms)))).T
                            (a, b) = np.linalg.lstsq(x, y, rcond=None)[0]
                            eeg_converted_record_timestamp_ms = eeg_device_timestamp_ms * a + b

                            columns = ["Raw Timestamp(ms)", "Timestamp(ms)"] + [f"Ch{i+1}" for i in range(eeg_data.shape[1])]

                            df_eeg = pd.DataFrame(
                                np.column_stack((eeg_record_timestamp_ms, eeg_converted_record_timestamp_ms, eeg_data)),
                                columns=columns
                            )

                            outpath = dart_file.parent / eeg_file.name.replace(".bin", ".csv")
                            df_eeg.to_csv(outpath, index=False)
                else:
                    print(f"[Warning] EEG file {eeg_file} is empty or missing, skipping.")

            #  ====== hmd file clean  ====== 
            if hmd_file:
                df_hmd = pd.read_csv(hmd_file)
                dt = df_hmd["UnixTime(ms)"].diff() / 1000
                for axis in {"X", "Y", "Z"}:
                    pos = df_hmd[f"HandPosition{axis}"]
                    df_hmd[f"HandNaiveVelocity_{axis}"] = pos.diff() / dt
                df_hmd.fillna(0, inplace=True)
                outpath = hmd_file.parent / ("corr_" + hmd_file.name)
                # print(outpath)
                df_hmd.to_csv(outpath, index=False)

        df_results = pd.concat(df_list, ignore_index=True)
        # print(df_results.shape[0])
        
        df_results = kinematics.calculate_corrected_hit_point(df_results)

        for axis in ["X", "Y", "Z"]:
            df_results[f"HitRelativeCorr{axis}"] = (
                df_results[f"HitPosCorr{axis}"] - df_results[f"TargetPos{axis}"]
            )
        
        for axis in ["X", "Y", "Z"]:
            df_results[f"HitRelative{axis}"] = (
                df_results[f"HitPos{axis}"] - df_results[f"TargetPos{axis}"]
            )

        df_results["HitDistance"] = np.sqrt(
            df_results["HitRelativeX"]**2 +
            df_results["HitRelativeY"]**2 +
            df_results["HitRelativeZ"]**2
        )
        df_results["HitDistanceCorr"] = np.sqrt(
            df_results["HitRelativeCorrX"]**2 +
            df_results["HitRelativeCorrY"]**2 +
            df_results["HitRelativeCorrZ"]**2
        )

        print(df_results.shape)

        df_results.to_csv(base_dir /"1_chopped"/ f"{sub}_all_results.csv", index=False)