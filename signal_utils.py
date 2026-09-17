"""Shared signal-processing helpers.

The routines here are deliberately dependency-light: SciPy filtering plus
NumPy array handling. Every transform applied to the electrocardiogram and to
the inertial channels is defined once in this module so that the donor and
host recordings receive byte-identical treatment, which is a prerequisite for
the graft-controlled corpus described in the manuscript.
"""

import numpy as np
from scipy import signal as sps


def bandpass(x, fs, lo, hi, order=3):
    """Zero-phase Butterworth band-pass."""
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999)
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    return sps.filtfilt(b, a, x, axis=0)


def resample_to(x, fs_in, fs_out):
    """Polyphase resampling with the anti-alias filter SciPy supplies."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    g = np.gcd(int(fs_in), int(fs_out))
    return sps.resample_poly(x, int(fs_out // g), int(fs_in // g), axis=0)


def detect_r_peaks(ecg, fs, refractory_s=0.30, block_s=8.0):
    """Block-adaptive energy-based R-peak localisation.

    A band-limited derivative is squared and smoothed over a 150 ms window.
    Because the recordings alternate between quiet postures and vigorous
    exercise, a single global threshold either misses beats during rest or
    admits motion spikes during running; the threshold is therefore recomputed
    inside consecutive blocks from the block median and median absolute
    deviation of the integrated envelope. Candidates are refined onto the local
    extremum of the band-passed trace and thinned by a refractory interval.
    """
    x = bandpass(ecg, fs, 5.0, min(15.0, 0.45 * fs))
    d = np.diff(x, prepend=x[0])
    env = np.convolve(d ** 2, np.ones(max(int(0.15 * fs), 3)) / max(int(0.15 * fs), 3),
                      mode="same")
    if env.std() < 1e-12:
        return np.array([], dtype=int)

    n_block = max(int(block_s * fs), 4 * fs)
    thr = np.empty_like(env)
    for a in range(0, len(env), n_block):
        b = min(a + n_block, len(env))
        seg = env[a:b]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med)) + 1e-12
        thr[a:b] = med + 1.2 * mad

    cand, _ = sps.find_peaks(env, distance=int(refractory_s * fs))
    cand = cand[env[cand] > thr[cand]]

    refined, half = [], max(int(0.05 * fs), 1)
    for p in cand:
        a, b = max(p - half, 0), min(p + half + 1, len(x))
        refined.append(a + int(np.argmax(np.abs(x[a:b]))))
    refined = np.unique(np.asarray(refined, dtype=int))

    # Enforce the refractory interval after refinement.
    out = []
    for p in refined:
        if not out or p - out[-1] >= int(refractory_s * fs):
            out.append(p)
        elif env[p] > env[out[-1]]:
            out[-1] = p
    return np.asarray(out, dtype=int)


def extract_segments(x, centres, length):
    """Cut fixed-length windows centred on the supplied indices."""
    half = length // 2
    keep, out = [], []
    for i, c in enumerate(centres):
        a, b = c - half, c - half + length
        if a < 0 or b > len(x):
            continue
        out.append(x[a:b])
        keep.append(i)
    if not out:
        return np.zeros((0, length), dtype=np.float32), np.array([], dtype=int)
    return np.asarray(out, dtype=np.float32), np.asarray(keep, dtype=int)


def zscore(x, axis=-1, eps=1e-6):
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def robust_amplitude(x, axis=-1):
    """Median absolute deviation scaled to an approximate standard deviation."""
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis, keepdims=True)


def window_average(values, centres, fs, win_s):
    """Centred moving average of a beat-indexed series over a time window.

    Heart rate is reported clinically as an average over several seconds rather
    than as a single beat-to-beat reciprocal, and the averaged target is what
    the rate head of the model is asked to reproduce.
    """
    centres = np.asarray(centres, dtype=np.float64)
    half = 0.5 * win_s * fs
    out = np.empty(len(values), dtype=np.float64)
    lo = np.searchsorted(centres, centres - half, side="left")
    hi = np.searchsorted(centres, centres + half, side="right")
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    for i in range(len(values)):
        a, b = lo[i], max(hi[i], lo[i] + 1)
        out[i] = (cs[b] - cs[a]) / (b - a)
    return out.astype(np.float32)
