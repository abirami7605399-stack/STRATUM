"""Per-fold sizes of each motion stratum in the coupled evaluation set.

The stratified false alarm rate of a single fold is estimated from a few hundred
beats, so the five folds are combined into one prevalence-weighted estimate. The
denominators are a property of the evaluation set and are identical for every
scheme, which is what makes the weighting legitimate.
"""

import json
import numpy as np

import config as C
import experiment_setup as E
import run_experiments as RX


def main(n_folds=5):
    raw = E.load_raw()
    folds = E.folds(n_folds)
    out = {}
    for fi in range(n_folds):
        _, tests = E.make_fold(raw, folds[fi], seed=fi, **RX.CORPUS)
        d = tests["cpl"]
        m = d["rhythm"] >= 0
        inten = np.asarray(C.ACTIVITY_INTENSITY)[np.clip(d["activity"], 0, 11)]
        rec = {}
        for lvl, tag in ((0, "sed"), (1, "mod"), (2, "vig")):
            sel = m & (inten == lvl)
            rec[f"n_normal_{tag}"] = int(((d["rhythm"] == 0) & sel).sum())
            rec[f"n_{tag}"] = int(sel.sum())
        out[str(fi)] = rec
        print(fi, rec, flush=True)
    with open("results/fold_denoms.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
