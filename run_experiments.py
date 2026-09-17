"""Experiment driver.

Every configuration reported in the manuscript is declared here and executed
through the same :func:`trainer.run` entry point. Results are written as one
JSON record per (experiment, method, fold) so that a partial run can be resumed
and so that the statistical analysis reads exactly what the training produced.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import config as C
import experiment_setup as E
import trainer

torch.set_num_threads(1)

BASE = dict(rounds=16, tau2=2, local_steps=6, batch=128, lr=2.8e-3,
            client_frac=0.5, w_v=0.6, w_a=0.5, w_adv=0.35, w_cf=0.30)
CORPUS = dict(cap=400)

# ---------------------------------------------------------------------------
# Competing schemes. Each entry overrides the STRATUM defaults.
# ---------------------------------------------------------------------------
METHODS = {
    "Local": dict(topology="local", aggregation="fedavg", mediated=False,
                  adapter=False, w_adv=0.0, w_cf=0.0, calib="none", task_coverage=False),
    "FedAvg": dict(topology="flat", aggregation="fedavg", mediated=False,
                   adapter=False, personalise=False, w_adv=0.0, w_cf=0.0,
                   calib="none", task_coverage=False),
    "FedProx": dict(topology="flat", aggregation="fedavg", mediated=False,
                    adapter=False, personalise=False, w_adv=0.0, w_cf=0.0,
                    calib="none", w_prox=1e-2, task_coverage=False),
    "FedBN-Per": dict(topology="flat", aggregation="fedavg", mediated=False,
                      adapter=False, personalise=True, w_adv=0.0, w_cf=0.0,
                      calib="none", task_coverage=False),
    "HierFAVG": dict(topology="hier", aggregation="fedavg", mediated=False,
                     adapter=False, personalise=False, w_adv=0.0, w_cf=0.0,
                     calib="none", task_coverage=False),
    "HierFAVG-TC": dict(topology="hier", aggregation="fedavg", mediated=False,
                        adapter=False, personalise=True, w_adv=0.0, w_cf=0.0,
                        calib="none", task_coverage=True),
    "Hier-MTL-Adapter": dict(topology="hier", aggregation="fedavg", mediated=False,
                             adapter=True, personalise=True, w_adv=0.0, w_cf=0.0,
                             calib="global", task_coverage=True),
    "Hier-IRM": dict(topology="hier", aggregation="fedavg", mediated=False,
                     adapter=True, personalise=True, w_adv=0.0, w_cf=0.0,
                     calib="global", task_coverage=True, irm=True),
    "STRATUM": dict(),
    "Centralised": dict(topology="central", aggregation="fedavg", mediated=True,
                        adapter=True, personalise=True, calib="cond",
                        task_coverage=True),
}

# ---------------------------------------------------------------------------
# Component ablations (component removal from the full framework)
# ---------------------------------------------------------------------------
ABLATIONS = {
    "Full": dict(),
    "w/o mediated head": dict(mediated=False),
    "w/o adversary": dict(w_adv=0.0),
    "w/o counterfactual": dict(w_cf=0.0),
    "w/o mediation (M)": dict(mediated=False, w_adv=0.0, w_cf=0.0),
    "w/o stratified agg (S)": dict(aggregation="fedavg"),
    "w/o coverage weighting": dict(task_coverage=False),
    "w/o cond. calibration (C)": dict(calib="global"),
    "w/o context adapter": dict(adapter=False),
    "rhythm task only": dict(w_a=0.0, w_v=0.0, mediated=False, w_adv=0.0, w_cf=0.0),
}


_FOLD_CACHE = {}


def _fold_cache(raw, held, seed, corp):
    """Corpora are deterministic in their arguments, so they are built once."""
    key = (tuple(held), seed, tuple(sorted(corp.items())))
    if key not in _FOLD_CACHE:
        _FOLD_CACHE.clear()
        _FOLD_CACHE[key] = E.make_fold(raw, held, seed=seed, **corp)
    return _FOLD_CACHE[key]


def record_path(tag):
    return os.path.join(C.RESULT_DIR, f"{tag}.jsonl")


def already(tag, key):
    p = record_path(tag)
    if not os.path.exists(p):
        return set()
    done = set()
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["method"], r["fold"], r.get("setting", "")))
            except Exception:
                pass
    return done


def append(tag, rec):
    with open(record_path(tag), "a") as f:
        f.write(json.dumps(rec) + "\n")


def execute(tag, jobs, fold_ids, raw, folds, fold_seed=0, corpus=None, repeats=1):
    """Run a list of (method, cfg, setting) jobs with the fold as the outer loop.

    Building a corpus costs about as much as a short training run, so the fold is
    constructed once and every job belonging to it reuses the same clients and
    the same evaluation sets. This also guarantees that the comparison is paired.
    """
    done = already(tag, None)
    for rep in range(repeats):
        for fi in fold_ids:
            rec_id = fi + 100 * rep
            pending = [j for j in jobs if (j[0], rec_id, j[2]) not in done]
            if not pending:
                continue
            seed = fold_seed + 17 * rep + fi
            corp = {**CORPUS, **(corpus or {})}
            clients, tests = E.make_fold(raw, folds[fi], seed=seed, **corp)
            for method, cfg, setting in pending:
                t0 = time.time()
                res = trainer.run(clients, tests, {**BASE, **cfg}, seed=seed)
                res.update(method=method, fold=rec_id, setting=setting,
                           wall_s=round(time.time() - t0, 1))
                append(tag, res)
                print(f"[{tag}] {method} {setting} fold {rec_id}: "
                      f"cpl_f1={res.get('cpl_f1_macro', float('nan')):.3f} "
                      f"adv_f1={res.get('adv_f1_macro', float('nan')):.3f} "
                      f"far_vig={res.get('cpl_far_vig', float('nan')):.3f} "
                      f"({res['wall_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["main", "ablation", "sensitivity", "topology"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--subset", default="")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    raw = E.load_raw()
    folds = E.folds(args.folds)

    if args.stage == "main":
        keys = [k for k in METHODS if not args.subset or k in args.subset.split(",")]
        execute("main", [(k, METHODS[k], "") for k in keys],
                range(args.folds), raw, folds, repeats=args.repeats)

    elif args.stage == "ablation":
        execute("ablation", [(k, v, "") for k, v in ABLATIONS.items()],
                range(3), raw, folds)

    elif args.stage == "sensitivity":
        for kap in (0.4, 0.6, 0.8, 1.0):
            execute("sensitivity",
                    [(m, METHODS[m], f"kappa={kap}")
                     for m in ("STRATUM", "HierFAVG-TC")],
                    range(3), raw, folds, corpus=dict(kappa_train=kap))
        execute("sensitivity",
                [("STRATUM", dict(w_adv=w), f"w_adv={w}") for w in (0.0, 0.15, 0.6)],
                range(3), raw, folds)

    elif args.stage == "topology":
        execute("topology",
                [("STRATUM", dict(n_cells=n), f"cells={n}") for n in (2, 4, 6)]
                + [("STRATUM", dict(tau2=t), f"tau2={t}") for t in (1, 4)]
                + [("STRATUM", dict(mixed_cells=True), "mixed-cells")],
                range(3), raw, folds)


if __name__ == "__main__":
    main()
