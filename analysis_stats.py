"""Aggregation and statistical analysis of the recorded runs.

Folds are the experimental unit: every method is trained and evaluated on the
same leave-two-wearers-out partitions with the same seeds, so the comparisons
are paired. A Friedman test first asks whether the methods differ at all across
the paired blocks; only then are pairwise comparisons made against the proposed
framework, with the Holm step-down correction controlling the family-wise error
rate and a rank-biserial correlation reporting the size of each difference.
"""

import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import config as C


def load(tag):
    path = os.path.join(C.RESULT_DIR, f"{tag}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def matrix(rows, metric, key="method"):
    """Return {name: {fold: value}} restricted to folds present for every name."""
    d = defaultdict(dict)
    for r in rows:
        if metric in r and r[metric] is not None:
            v = r[metric]
            if isinstance(v, (int, float)) and np.isfinite(v):
                d[r[key] + ("|" + r["setting"] if r.get("setting") else "")][r["fold"]] = v
    if not d:
        return {}, []
    common = set.intersection(*[set(v.keys()) for v in d.values()])
    common = sorted(common)
    return {k: np.array([v[f] for f in common]) for k, v in d.items()}, common


def summarise(rows, metrics, key="method"):
    out = defaultdict(dict)
    for m in metrics:
        mat, folds = matrix(rows, m, key)
        for name, vals in mat.items():
            out[name][m] = (float(vals.mean()), float(vals.std(ddof=1))
                            if len(vals) > 1 else 0.0, len(vals))
    return out


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for the Wilcoxon statistic."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rn) / r.sum())


def benjamini_hochberg(pvals):
    """Step-up control of the false discovery rate."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * n / (rank + 1))
        adj[idx] = min(running, 1.0)
    return adj


def holm(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def bootstrap_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def compare(rows, metric, reference="STRATUM", higher_is_better=True):
    mat, folds = matrix(rows, metric)
    if reference not in mat:
        return None
    ref = mat[reference]
    names = [n for n in mat if n != reference]
    alt = "greater" if higher_is_better else "less"
    raw_p, stat_w = [], []
    for n in names:
        try:
            w, p = stats.wilcoxon(ref, mat[n], zero_method="wilcox",
                                  alternative=alt)
        except ValueError:
            w, p = float("nan"), 1.0
        raw_p.append(p)
        stat_w.append(w)
    adj_holm = holm(np.array(raw_p)) if raw_p else np.array([])
    adj = benjamini_hochberg(np.array(raw_p)) if raw_p else np.array([])
    out = []
    for i, n in enumerate(names):
        diff, lo, hi = bootstrap_ci(ref, mat[n])
        out.append({
            "method": n, "mean": float(mat[n].mean()),
            "sd": float(mat[n].std(ddof=1)) if len(mat[n]) > 1 else 0.0,
            "delta": diff, "ci_lo": lo, "ci_hi": hi,
            "p_raw": float(raw_p[i]), "p_holm": float(adj_holm[i]),
            "p_fdr": float(adj[i]),
            "effect": rank_biserial(ref, mat[n]),
            "n": len(mat[n]),
        })
    order = sorted(out, key=lambda r: -r["mean"] if higher_is_better else r["mean"])
    return {"reference": reference, "ref_mean": float(ref.mean()),
            "ref_sd": float(ref.std(ddof=1)) if len(ref) > 1 else 0.0,
            "n_folds": len(folds), "rows": order}


def friedman(rows, metric):
    mat, folds = matrix(rows, metric)
    if len(mat) < 3:
        return None
    arrs = [mat[k] for k in sorted(mat)]
    try:
        s, p = stats.friedmanchisquare(*arrs)
    except Exception:
        return None
    ranks = np.array([stats.rankdata(-np.array(col)) for col in zip(*arrs)])
    return {"stat": float(s), "p": float(p), "k": len(arrs), "n": len(folds),
            "mean_ranks": {k: float(ranks[:, i].mean())
                           for i, k in enumerate(sorted(mat))}}


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference of the Nemenyi post-hoc test."""
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
         8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    qa = q.get(k, 3.3)
    return float(qa * np.sqrt(k * (k + 1) / (6.0 * n)))


HIGHER = {"cpl_far_vig": False, "cpl_far_sed": False, "cpl_far_mod": False,
          "cpl_ece": False, "cpl_brier": False, "cpl_nll": False,
          "cpl_hr_mae": False, "cpl_hr_rmse": False}


def report(tag="main", metric="cpl_f1_macro", reference="STRATUM"):
    rows = load(tag)
    cmp = compare(rows, metric, reference, HIGHER.get(metric, True))
    fr = friedman(rows, metric)
    print(f"\n=== {tag} :: {metric} ===")
    if fr:
        print(f"Friedman chi2={fr['stat']:.2f} p={fr['p']:.2e} "
              f"(k={fr['k']}, n={fr['n']}), CD={nemenyi_cd(fr['k'], fr['n']):.2f}")
    if cmp:
        print(f"{reference}: {cmp['ref_mean']:.4f} +/- {cmp['ref_sd']:.4f} "
              f"(n={cmp['n_folds']})")
        for r in cmp["rows"]:
            print(f"  {r['method']:<22s} {r['mean']:.4f}+/-{r['sd']:.4f} "
                  f"d={r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                  f"p={r['p_raw']:.4f} fdr={r['p_fdr']:.4f} "
                  f"holm={r['p_holm']:.4f} r={r['effect']:+.2f}")
    return cmp, fr


if __name__ == "__main__":
    for met in ("cpl_f1_macro", "cpl_far_vig", "adv_f1_macro", "ext_mcc",
                "cpl_hr_mae", "cpl_act_acc", "cpl_ece"):
        report("main", met)
