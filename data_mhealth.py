"""MHEALTH ingestion.

Every subject contributes a continuous 50 Hz recording of two-lead
electrocardiography from a chest unit together with tri-axial acceleration at
the chest, the left ankle and the right lower arm, annotated with twelve
scripted activities. Beats are localised on the chest lead, and each beat is
paired with the inertial window that encloses it, so that the rhythm segment
and its motion context describe the same instant of the same recording.
"""

import os
import numpy as np

import config as C
import signal_utils as su

# Column indices (zero-based) of the nine acceleration channels retained.
ACC_COLS = [0, 1, 2, 5, 6, 7, 14, 15, 16]
ECG_COL = 4          # chest lead II
LABEL_COL = 23


def load_subject(sid, mhealth_dir=None):
    mhealth_dir = mhealth_dir or C.MHEALTH_DIR
    path = os.path.join(mhealth_dir, f"mHealth_subject{sid}.log")
    raw = np.loadtxt(path, dtype=np.float32)

    lab = raw[:, LABEL_COL].astype(np.int64)
    ecg = su.bandpass(raw[:, ECG_COL].astype(np.float64), C.FS, *C.ECG_BAND)
    acc = raw[:, ACC_COLS].astype(np.float32)

    peaks = su.detect_r_peaks(ecg, C.FS)
    if len(peaks) < 20:
        return None

    beats, kept = su.extract_segments(ecg, peaks, C.BEAT_LEN)
    peaks = peaks[kept]

    ctx, kept2 = _context_windows(acc, peaks)
    beats, peaks = beats[kept2], peaks[kept2]

    act = lab[peaks]
    valid = act > 0
    beats, peaks, ctx, act = beats[valid], peaks[valid], ctx[valid], act[valid] - 1

    rr = np.diff(peaks, prepend=peaks[0] - int(0.8 * C.FS)) / float(C.FS)
    if len(rr) > 1:
        rr[0] = rr[1]
    hr = su.window_average(60.0 / np.clip(rr, 0.25, 2.5), peaks, C.FS,
                           C.HR_AVG_S).astype(np.float32)

    return {
        "subject": int(sid),
        "beats": beats.astype(np.float32),
        "ctx": ctx.astype(np.float32),
        "activity": act.astype(np.int64),
        "hr": hr,
        "peaks": peaks.astype(np.int64),
        "ecg_raw": ecg.astype(np.float32),
    }


def _context_windows(acc, centres):
    """Decimated inertial windows of CTX_WIN_S seconds around each beat."""
    half = C.CTX_LEN // 2
    out, keep = [], []
    for i, c in enumerate(centres):
        a, b = c - half, c - half + C.CTX_LEN
        if a < 0 or b > len(acc):
            continue
        w = acc[a:b]                                   # (CTX_LEN, 9)
        w = w.reshape(C.CTX_LEN_DS, C.CTX_DS, acc.shape[1]).mean(axis=1)
        out.append(w.T)                                # (9, CTX_LEN_DS)
        keep.append(i)
    if not out:
        return np.zeros((0, C.N_IMU_CH, C.CTX_LEN_DS), np.float32), np.array([], int)
    return np.asarray(out, np.float32), np.asarray(keep, int)


def beat_templates(sub):
    """Per-activity average beat, used to separate morphology from artifact."""
    templates = {}
    for a in np.unique(sub["activity"]):
        m = sub["activity"] == a
        if m.sum() >= 5:
            templates[int(a)] = np.median(sub["beats"][m], axis=0)
    glob = np.median(sub["beats"], axis=0)
    return templates, glob


def motion_energy(ctx):
    """Scalar motion index: mean channel variance of the inertial window."""
    return ctx.var(axis=-1).mean(axis=-1)


if __name__ == "__main__":
    for s in (1, 2):
        d = load_subject(s)
        print(s, d["beats"].shape, d["ctx"].shape,
              "activities", np.bincount(d["activity"], minlength=12).tolist(),
              "HR median %.1f" % np.median(d["hr"]))
