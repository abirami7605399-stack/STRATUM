"""MIT-BIH Arrhythmia Database ingestion.

Each record is read through the WFDB interface, restricted to the modified
limb lead II when present, band-limited, resampled to the common 50 Hz working
rate and cut into beat-centred segments carrying AAMI class labels. Records
with paced beats are excluded following the AAMI recommendation, and the
DS1/DS2 division of the remaining records is used so that no patient appears
on both sides of the evaluation boundary.
"""

import os
import numpy as np
import wfdb

import config as C
import signal_utils as su


def _pick_channel(record):
    """Prefer MLII; fall back to the first available channel."""
    names = [s.strip() for s in record.sig_name]
    for target in ("MLII", "ML II", "II"):
        if target in names:
            return names.index(target)
    return 0


def load_record(rec_id, mitdb_dir=None):
    """Return beat segments, AAMI labels and instantaneous heart rate."""
    mitdb_dir = mitdb_dir or C.MITDB_DIR
    path = os.path.join(mitdb_dir, str(rec_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    ch = _pick_channel(rec)
    sig = np.asarray(rec.p_signal[:, ch], dtype=np.float64)
    fs_in = int(rec.fs)

    sig = su.bandpass(sig, fs_in, *C.ECG_BAND)
    sig = su.resample_to(sig, fs_in, C.FS)
    scale = C.FS / float(fs_in)

    samples, symbols = np.asarray(ann.sample), np.asarray(ann.symbol)
    keep = np.array([s in C.AAMI_MAP for s in symbols])
    samples, symbols = samples[keep], symbols[keep]
    if len(samples) < 8:
        return None

    centres = np.round(samples * scale).astype(int)
    labels = np.array([C.AAMI_CLASSES.index(C.AAMI_MAP[s]) for s in symbols])

    # Instantaneous rate from the preceding RR interval, in beats per minute.
    rr = np.diff(centres, prepend=centres[0] - int(0.8 * C.FS)) / float(C.FS)
    rr[0] = rr[1] if len(rr) > 1 else 0.8
    hr = su.window_average(60.0 / np.clip(rr, 0.25, 2.5), centres, C.FS, C.HR_AVG_S)
    # Prematurity of each beat relative to the prevailing rhythm of its record.
    prem = np.clip(rr / max(np.median(rr), 1e-3), 0.35, 1.8)

    beats, kept = su.extract_segments(sig, centres, C.BEAT_LEN)
    if len(kept) == 0:
        return None

    return {
        "record": int(rec_id),
        "beats": su.zscore(beats).astype(np.float32),
        "labels": labels[kept].astype(np.int64),
        "hr": hr[kept].astype(np.float32),
        "prem": prem[kept].astype(np.float32),
        "age": _header_age(rec),
    }


def _header_age(rec):
    """Patient age from the free-text header comment, or -1 when absent."""
    for line in (rec.comments or []):
        parts = line.split()
        if parts and parts[0].isdigit():
            return int(parts[0])
    return -1


def load_split(rec_ids, mitdb_dir=None, verbose=False):
    out = []
    for r in rec_ids:
        if r in C.PACED:
            continue
        try:
            d = load_record(r, mitdb_dir)
        except Exception as exc:                      # pragma: no cover
            if verbose:
                print(f"  record {r} skipped ({exc})")
            continue
        if d is not None:
            out.append(d)
            if verbose:
                cnt = np.bincount(d["labels"], minlength=C.N_RHYTHM)
                print(f"  {r}: {len(d['labels'])} beats  N/S/V/F = {cnt.tolist()}")
    return out


def donor_pool(records):
    """Class-indexed pool of donor beats with their prematurity ratios.

    The ratio travels with the waveform so that a graft can reproduce not only
    the morphology of an ectopic beat but also the early arrival and the
    compensatory pause that surround it, which is what separates a
    supraventricular ectopic beat from a normal one.
    """
    beats = {c: [] for c in range(C.N_RHYTHM)}
    prem = {c: [] for c in range(C.N_RHYTHM)}
    for d in records:
        for c in range(C.N_RHYTHM):
            m = d["labels"] == c
            if m.any():
                beats[c].append(d["beats"][m])
                prem[c].append(d["prem"][m])
    return {c: (np.concatenate(beats[c]) if beats[c]
                else np.zeros((0, C.BEAT_LEN), np.float32),
                np.concatenate(prem[c]) if prem[c] else np.zeros(0, np.float32))
            for c in range(C.N_RHYTHM)}


if __name__ == "__main__":
    recs = load_split(C.DS1[:3], verbose=True)
    print("loaded", len(recs), "records")
