"""Activity-conditional temperature scaling.

A single temperature assumes that the confidence of a rhythm decision is
mis-scaled by the same factor whatever the wearer is doing. That assumption
fails on ambulatory recordings: confidence degrades far more during vigorous
movement than at rest. One temperature is therefore fitted per motion-intensity
stratum on a held-out federated calibration split, and the stratum of an unseen
beat is taken from the activity the model itself predicts, so no additional
label is required at inference time.
"""

import numpy as np
import config as C

N_STRATA = 3


def _nll(logits, y, t):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits, y, grid=None):
    grid = grid if grid is not None else np.linspace(0.40, 4.0, 91)
    if len(y) < 10:
        return 1.0
    losses = [_nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def fit_conditional(logits, y, strata, global_fallback=True):
    """One temperature per stratum, backing off to the pooled estimate."""
    t_glob = fit_temperature(logits, y)
    temps = np.full(N_STRATA, t_glob, dtype=np.float64)
    for s in range(N_STRATA):
        m = strata == s
        if m.sum() >= 50:
            temps[s] = fit_temperature(logits[m], y[m])
        elif not global_fallback:
            temps[s] = 1.0
    return temps, t_glob


def apply_temperature(logits, temps, strata):
    t = np.asarray(temps, dtype=np.float64)[np.clip(strata, 0, N_STRATA - 1)]
    z = logits / t[:, None]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predicted_strata(logit_a):
    """Motion stratum implied by the activity head."""
    a = logit_a.argmax(axis=1)
    return np.asarray(C.ACTIVITY_INTENSITY)[np.clip(a, 0, C.N_ACTIVITY - 1)]
