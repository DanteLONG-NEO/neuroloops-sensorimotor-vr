import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_corrected_hit_point(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["TotalHitTimes"])
    g = df["Gravity"] * 9.81

    scaleFactor = 1.0
    BOARDWIDTH = 0.0153

    RVX = df["RVX"] * scaleFactor
    RVY = df["RVY"] * scaleFactor
    RVZ = df["RVZ"] * scaleFactor

    tz = (df["TargetPosZ"] - BOARDWIDTH/2 - df["RPZ"]) / RVZ
    t_corr = np.abs(tz)

    # compute corr diff
    txhit = (df["HitPosX"] - df["RPX"]) / RVX
    tzhit = (df["HitPosZ"] - df["RPZ"]) / RVZ
    xzdifference = txhit-tzhit

    # plt.figure()
    # plt.hist(xzdifference, bins=10, range=[xzdifference.min(), xzdifference.max()])
    # plt.show()

    print(f"xz time diff: {round(xzdifference.min(),3), round(xzdifference.max(),3)}")


    # txlast = (df["LastPosX"] - df["RPX"]) / RVX.replace(0, np.nan)
    # tzlast = (df["LastPosZ"] - df["RPZ"]) / RVZ.replace(0, np.nan)
    # xzdifference2 = txlast - tzlast

    # xdiff = df["LastPosX"] - df["HitPosX"]
    # ydiff = df["LastPosY"] - df["HitPosY"]
    # zdiff = df["LastPosZ"] - df["HitPosZ"]

    # print(xzdifference2.min(), xzdifference2.max())
    # print([xdiff.min(), xdiff.max()], [ydiff.min(), ydiff.max()],[zdiff.min(), zdiff.max()])

    df["HitPosCorrX"] = pd.to_numeric(df["RPX"] + RVX * t_corr, errors="coerce")
    df["HitPosCorrY"] = pd.to_numeric(df["RPY"] + RVY * t_corr - 0.5 * g * (t_corr**2), errors="coerce")
    df["HitPosCorrZ"] = pd.to_numeric(df["RPZ"] + RVZ * t_corr, errors="coerce")

    return df

def return_unstable_flying_trials(df: pd.DataFrame, sort_cols:tuple):
    df = df.sort_values(sort_cols)

    scaleFactor = 1.0

    RVX = df["RVX"] * scaleFactor
    RVZ = df["RVZ"] * scaleFactor

    # compute corr diff
    txhit = (df["HitPosX"] - df["RPX"]) / RVX
    tzhit = (df["HitPosZ"] - df["RPZ"]) / RVZ
    xzdifference = txhit-tzhit

    return np.abs(xzdifference) <= 0.5

def calculate_naive_velocity(df, incolname, outcolname):
    dt = df["UnixTime(ms)"].diff() / 1000
    for axis in ["X", "Y", "Z"]:
        pos = df[f"{incolname}{axis}"]
        df[f"{outcolname}{axis}"] = pos.diff() / dt
    df.fillna(0, inplace=True)

def maxpool_past_velocity(df: pd.DataFrame, n: int, incolname: str) -> pd.DataFrame:
    vx, vy, vz = df[f"{incolname}X"], df[f"{incolname}Y"], df[f"{incolname}Z"]

    mag = np.sqrt(vx**2 + vy**2 + vz**2)
    # df["VelMag"] = mag

    max_x, max_y, max_z, max_mag = [], [], [], []

    for i in range(len(df)):
        start = max(0, i - n + 1)
        end = i + 1
        window_mag = mag[start:end]
        idx_max = window_mag.idxmax()

        max_x.append(vx.loc[idx_max])
        max_y.append(vy.loc[idx_max])
        max_z.append(vz.loc[idx_max])
        max_mag.append(mag.loc[idx_max])

    df["MaxPoolingVelocity_X"] = max_x
    df["MaxPoolingVelocity_Y"] = max_y
    df["MaxPoolingVelocity_Z"] = max_z
    # df["MaxMag"] = max_mag