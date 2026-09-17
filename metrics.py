"""Evaluation metrics.

Rhythm performance is reported with imbalance-aware measures because the
supraventricular and fusion classes are rare; the rate task is reported in
beats per minute so that the error is directly interpretable; and calibration
is reported separately because a rhythm alarm is acted on by its confidence,
not only by its argument of maximum.
"""

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)

import config as C


def rhythm_metrics(y_true, logits, prefix=""):
    p = _softmax(logits)
    y_pred = p.argmax(axis=1)
    out = {
        f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                            labels=list(range(C.N_RHYTHM)),
                                            zero_division=0)),
        f"{prefix}bacc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}acc": float((y_pred == y_true).mean()),
    }
    for c, name in enumerate(C.AAMI_CLASSES):
        m = y_true == c
        out[f"{prefix}recall_{name}"] = float((y_pred[m] == c).mean()) if m.any() else float("nan")
    try:
        present = sorted(set(y_true.tolist()))
        if len(present) > 1:
            out[f"{prefix}auc"] = float(roc_auc_score(
                y_true, p[:, present] / p[:, present].sum(1, keepdims=True),
                multi_class="ovr", average="macro", labels=present))
        else:
            out[f"{prefix}auc"] = float("nan")
    except Exception:
        out[f"{prefix}auc"] = float("nan")
    return out


def binary_metrics(y_true, logits, prefix=""):
    """Ectopic-beat detection derived from the three-class posterior.

    A rhythm alarm is raised whenever the posterior mass on the two ectopic
    classes exceeds that on the normal class. Sensitivity, specificity and the
    false alarm rate reported this way are the quantities a monitoring service
    actually operates on.
    """
    p = _softmax(logits)
    score = p[:, 1:].sum(axis=1)
    alarm = score > 0.5
    pos = y_true > 0
    sens = float(alarm[pos].mean()) if pos.any() else float("nan")
    spec = float((~alarm[~pos]).mean()) if (~pos).any() else float("nan")
    tp = float((alarm & pos).sum()); fp = float((alarm & ~pos).sum())
    fn = float((~alarm & pos).sum())
    prec = tp / max(tp + fp, 1e-9)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    try:
        auc = float(roc_auc_score(pos.astype(int), score)) if pos.any() and (~pos).any() else float("nan")
    except Exception:
        auc = float("nan")
    return {f"{prefix}sens": sens, f"{prefix}spec": spec, f"{prefix}prec": prec,
            f"{prefix}bf1": float(f1), f"{prefix}bauc": auc,
            f"{prefix}far": 1.0 - spec}


def false_alarm_rate(y_true, logits, intensity, level=2):
    """Share of genuinely normal beats declared ectopic at a given intensity."""
    pred = _softmax(logits).argmax(axis=1)
    m = (y_true == 0) & (intensity == level)
    return float((pred[m] > 0).mean()) if m.any() else float("nan")


def vital_metrics(v_true, v_pred, prefix=""):
    err = v_pred - v_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((v_true - v_true.mean()) ** 2).sum())
    return {
        f"{prefix}mae": float(np.abs(err).mean()),
        f"{prefix}rmse": float(np.sqrt((err ** 2).mean())),
        f"{prefix}r2": float(1.0 - ss_res / max(ss_tot, 1e-9)),
    }


def activity_metrics(a_true, logits, prefix=""):
    pred = logits.argmax(axis=1)
    return {
        f"{prefix}acc": float((pred == a_true).mean()),
        f"{prefix}f1_macro": float(f1_score(a_true, pred, average="macro",
                                            zero_division=0)),
    }


def calibration_metrics(y_true, probs, n_bins=15, prefix=""):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())
    ent = float((-probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1).mean())
    return {f"{prefix}ece": float(ece), f"{prefix}brier": brier,
            f"{prefix}nll": nll, f"{prefix}entropy": ent}


def reliability_curve(y_true, probs, n_bins=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() < 5:
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def confusion(y_true, logits):
    return confusion_matrix(y_true, _softmax(logits).argmax(axis=1),
                            labels=list(range(C.N_RHYTHM)))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


softmax = _softmax
