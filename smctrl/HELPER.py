import scipy.signal
from scipy import interpolate
import numpy as np
import pandas as pd
from scipy.signal import butter, cheby1, sosfilt, iirnotch, sosfreqz, sosfilt_zi, correlate, filtfilt, sosfiltfilt
from scipy.stats import median_abs_deviation
import re

def calculate_corrected_hit_point(df: pd.DataFrame) -> pd.DataFrame:
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
    # print(f"xz time diff: {round(xzdifference.min(),5), round(xzdifference.max(),5)}")


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
    df["HitPosCorrRelX"] = df["HitPosCorrX"] - df["TargetPosX"]
    df["HitPosCorrRelY"] = df["HitPosCorrY"] - df["TargetPosY"]
    df["HitPosCorrRelZ"] = df["HitPosCorrZ"] - df["TargetPosZ"]
    df["Radial_Error"] = np.sqrt(df["HitPosCorrRelX"]**2 + df["HitPosCorrRelY"]**2)

    return df

def segment_by_gap(df, time_col, max_gap_factor=10.0):
    """
    Segment a dataframe by time gaps in the specified column.
    Returns list of sub-dataframes separated by large time gaps.
    """
    if df.empty or len(df) < 2:
        return []

    t = df[time_col].dropna().to_numpy()
    if len(t) < 2:
        return []

    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return [df]

    dt_med = np.median(dt)
    cut_idx = np.where(dt > max_gap_factor * dt_med)[0]

    starts = np.r_[0, cut_idx + 1]
    ends = np.r_[cut_idx, len(t) - 1]

    segments = [df.iloc[s:e+1] for s, e in zip(starts, ends) if e > s]
    return [seg for seg in segments if not seg.empty]


def resample_df(df, time_col, fs_target, kind="linear",
                min_samples=5, fs_tolerance=0.05):
    """
    Resample a DataFrame whose time column is in milliseconds.

    如果样本太少或采样率跟目标太接近，就不做重采样，
    但仍然统一返回含有 'UnixTimeCorr(ms)' 和 'UnixTimeRaw(ms)' 的 DataFrame。
    """
    df = df.sort_values(by=time_col).drop_duplicates(subset=[time_col])
    t = df[time_col].to_numpy(dtype=float)
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{time_col} must be strictly increasing.")

    # ---------- 小工具：确保时间列存在 ----------
    def _ensure_time_cols(df_out):
        df_out = df_out.copy()
        if "UnixTimeCorr(ms)" not in df_out.columns:
            df_out["UnixTimeCorr(ms)"] = df_out[time_col].to_numpy()
        if "UnixTimeRaw(ms)" not in df_out.columns:
            # 对于没做重采样的情况，raw 就是原 time_col
            df_out["UnixTimeRaw(ms)"] = df_out[time_col].to_numpy()
        return df_out

    # ---------- 0. 样本太少：不 resample，补列后返回 ----------
    if len(df) < min_samples:
        return _ensure_time_cols(df)

    # ---------- 1. 估计原始采样率 ----------
    dt = np.diff(t) / 1000.0  # ms -> s
    fs_est = 1.0 / np.median(dt)
    # print 或 logger 你自己看着办
    print(f"Original fs ≈ {fs_est:.2f} Hz → Target {fs_target:.2f} Hz")

    # 如果采样率已经非常接近目标值，也可以直接 passthrough
    if abs(fs_est - fs_target) / fs_target < fs_tolerance:
        return _ensure_time_cols(df)

    # ---------- 2. 真正构造均匀时间轴 ----------
    t0, t1 = t[0], t[-1]
    step_ms = 1000.0 / fs_target
    n_new = int(np.floor((t1 - t0) / step_ms)) + 1
    t_uniform = t0 + np.arange(n_new) * step_ms

    data = {
        "UnixTimeCorr(ms)": t_uniform,
        "UnixTimeRaw(ms)": np.interp(t_uniform, t, t),
    }

    interp_cols = [
        col for col in df.columns
        if col not in ["LocalTime", "UnixTimeCorr(ms)", time_col]
    ]

    for col in interp_cols:
        f = interpolate.interp1d(
            t,
            df[col].to_numpy(),
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate"
        )
        data[col] = f(t_uniform)

    df_new = pd.DataFrame(data)
    return df_new


def bandpass_filter(
    data,
    lowcut=None,
    highcut=None,
    fs=None,
    filter_type='butter',
    order=4,
    rp=0.1,
):
    """
    Apply flexible Butterworth or Chebyshev I band/low/high-pass filter.
    Automatically detects which type of filter to apply based on lowcut/highcut.
    
    Args:
        data (array-like): Input signal.
        lowcut (float or None): Lower cutoff frequency (Hz). None for no high-pass.
        highcut (float or None): Upper cutoff frequency (Hz). None for no low-pass.
        fs (float): Sampling rate (Hz).
        filter_type (str): 'butter' or 'cheby1'.
        order (int): Filter order.
        rp (float): Ripple (for Chebyshev I).

    Returns:
        np.ndarray: Filtered signal (same shape as input).
    """
    if fs is None:
        raise ValueError("Sampling frequency 'fs' must be provided.")

    nyq = 0.5 * fs

    if lowcut is None and highcut is None:
        return data

    elif lowcut is not None and highcut is None:
        wn = lowcut / nyq
        ftype = 'high'
    
    elif lowcut is None and highcut is not None:
        wn = highcut / nyq
        ftype = 'low'
    
    else:
        wn = [lowcut / nyq, highcut / nyq]
        ftype = 'band'

    # Select filter type
    if filter_type == 'butter':
        sos = butter(order, wn, btype=ftype, output='sos')
    elif filter_type == 'cheby1':
        sos = cheby1(order, rp, wn, btype=ftype, output='sos')
    else:
        raise ValueError("filter_type must be 'butter' or 'cheby1'")

    if len(data) <= 3 * (order * 2 + 1):
        return data
    # Apply zero-phase filter
    y = sosfiltfilt(sos, data)
    return y

def notch_filter(
    data,
    fs,
    notch_freqs=(60.0,), Q=30.0
):
    from scipy.signal import filtfilt
    for f0 in notch_freqs:
        b_notch, a_notch = iirnotch(w0=f0, Q=Q, fs=fs)
        y = filtfilt(b_notch, a_notch, data)

    return y

def interpolate_channel(data, channel, method="nearby"):
    ch_nums = []

    for col in data.columns:
        m = re.match(r"[Cc][Hh]_?(\d+)", col)
        if m:
            ch_nums.append(int(m.group(1)))

    if len(ch_nums) == 0:
        raise ValueError("No channel data to interpolate")

    ch_nums = sorted(ch_nums)
    max_ch = ch_nums[-1]

    if channel == 0:
        data[f"Ch{channel}"] = 2*data[f"Ch{channel + 1}"] - data[f"Ch{channel + 2}"]
        return data

    if channel == max_ch:
        data[f"Ch{channel}"] = 2*data[f"Ch{channel - 1}"] - data[f"Ch{channel - 2}"]
        return data

    # ---------- nearby method ----------
    if method == "nearby":
        data[f"Ch{channel}"] = (
            data[f"Ch{channel - 1}"] + data[f"Ch{channel + 1}"]
        ) / 2.0
        return data

    # ---------- global method ----------
    if method == "global":
        ch_cols = [f"Ch{c}" for c in ch_nums]
        ch_cols_other = [col for col in ch_cols if col != f"Ch{channel}"]
        
        data[f"Ch{channel}"] = data[ch_cols_other].mean(axis=1)
        return data

    raise ValueError(f"Unknown interpolation method: {method}")


def normalize(
    data,
    feat_cols: list,
    method: str = "zscore"
):
    """
    Normalize feature columns using different methods.
    
    Args:
        data (pd.DataFrame): Input dataframe.
        feat_cols (list): List of column names to normalize.
        method (str): 'zscore', 'minmax', 'max', or 'robust_zscore'.
    
    Returns:
        pd.DataFrame: Normalized dataframe (copy).
    """
    data = data.copy()
    
    for feat_col in feat_cols:
        x = data[feat_col].to_numpy(dtype=float)

        if method == "zscore":
            mu, sigma = np.nanmean(x), np.nanstd(x)
            data[feat_col] = (x - mu) / sigma if sigma != 0 else x * 0

        elif method == "robust_zscore":
            med = np.nanmedian(x)
            mad = median_abs_deviation(x, nan_policy='omit')
            data[feat_col] = (x - med) / (1.4826 * mad) if mad != 0 else x * 0

        elif method == "minmax":
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            data[feat_col] = (x - xmin) / (xmax - xmin) if xmax != xmin else x * 0

        elif method == "max":
            xmax = np.nanmax(x)
            data[feat_col] = x / xmax if xmax != 0 else x * 0

        else:
            raise ValueError(f"{method} is not a valid normalization method.")

    return data
    
def bin_features(df, time_col, feat_cols, n_bins=3, agg="max"):
    """
    Bin time-series data into equal time intervals and aggregate feature values.

    Args:
        df (pd.DataFrame): time-series dataframe (must contain time_col).
        time_col (str): column name for time in ms.
        feat_cols (list): feature columns to aggregate.
        n_bins (int): number of bins.
        agg (str): aggregation function ('max', 'mean', 'median', etc.)

    Returns:
        pd.DataFrame: n_bins × len(feat_cols) dataframe.
    """
    # copy and normalize time range
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    t0, t1 = df[time_col].iloc[0], df[time_col].iloc[-1]
    df = df.copy()
    df["t_norm"] = (df[time_col] - t0) / (t1 - t0)  # normalized to [0,1]

    # assign each row to a bin
    df["bin"] = np.floor(df["t_norm"] * n_bins).astype(int)
    df.loc[df["bin"] == n_bins, "bin"] = n_bins - 1  # handle right edge

    # aggregate
    grouped = df.groupby("bin")[feat_cols]
    if agg == "max":
        binned = grouped.max()
    elif agg == "mean":
        binned = grouped.mean()
    elif agg == "median":
        binned = grouped.median()
    else:
        raise ValueError(f"Unsupported agg type: {agg}")

    # ensure bin order and completeness
    binned = binned.reindex(range(n_bins)).reset_index(drop=True)
    return binned

def compute_snr(data_series, baseline_series):
    x = np.asarray(data_series)
    b = np.asarray(baseline_series)

    rms_signal = np.sqrt(np.mean(x**2))
    rms_noise  = np.sqrt(np.mean(b**2))

    snr_db = 20 * np.log10(rms_signal / rms_noise)
    return snr_db

from scipy.stats import t
import numpy as np

def tost_one_sample(x, low, upp, alpha=0.05):
    x = np.array(x)
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x, ddof=1)
    se = sd_x / np.sqrt(n)
    
    # t statistics
    t1 = (mean_x - low) / se     # test H0: mean <= low
    t2 = (mean_x - upp) / se     # test H0: mean >= upp
    
    # one-sided p-values
    p1 = 1 - t.cdf(t1, df=n-1)   # mean > low ?
    p2 = t.cdf(t2, df=n-1)       # mean < upp ?
    
    return t1, t2, p1, p2

def get_feat_columns(csv_path, meta_dict):
    all_cols = pd.read_csv(csv_path, nrows=0).columns
    meta_cols = {
        col
        for cols in meta_dict.values()
        for col in cols
    }
    return [c for c in all_cols if c not in meta_cols]

def build_usecols(csv_path, time_col_dict, meta_dict, feature_dict, modality_key, use_all_feat):
    tcols = time_col_dict.get(modality_key, [])
    if not isinstance(tcols, list):
        tcols = [tcols] if tcols else []
    
    meta_cols = meta_dict.get(modality_key, [])
    if not isinstance(meta_cols, list):
        meta_cols = [meta_cols]

    if use_all_feat:
        all_cols = pd.read_csv(csv_path, nrows=0).columns

        meta_set = set(meta_cols)
        t_set    = set(tcols)

        feature_cols = [
            c for c in all_cols
            if (c not in meta_set) and (c not in t_set)
        ]
    else:
        raw_features = feature_dict.get(modality_key, [])
        if not isinstance(raw_features, list):
            raw_features = [raw_features]

        all_cols = pd.read_csv(csv_path, nrows=0).columns

        expanded_features = []

        for feat in raw_features:
            if feat in all_cols:
                expanded_features.append(feat)
            else:
                matched = [c for c in all_cols if c.startswith(feat)]
                expanded_features.extend(matched)

        feature_cols = list(set(expanded_features))

    feature_dict[modality_key] = feature_cols
    
    merged = []
    for c in tcols + feature_cols:
        if c not in merged:
            merged.append(c)
    return merged

def read_csv_with_dtype(path, usecols, time_cols):

    df = pd.read_csv(
        path,
        usecols=usecols,
        na_values=["", "NA", "NaN", "nan"],
        keep_default_na=True,
        low_memory=False
    )

    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if col not in time_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # print(f"path {path}: {df.head}")

    return df
