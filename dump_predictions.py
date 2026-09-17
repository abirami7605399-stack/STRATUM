"""Save the predictions needed by the calibration figure."""

import numpy as np
import torch

torch.set_num_threads(1)
import calibration as cal
import experiment_setup as E
import run_experiments as RX
import trainer


def main(fold=0):
    raw = E.load_raw()
    folds = E.folds(5)
    clients, tests = E.make_fold(raw, folds[fold], seed=fold, **RX.CORPUS)
    cfg = {**RX.BASE, **RX.METHODS["STRATUM"]}
    res = trainer.run(clients, tests, cfg, seed=fold, dump="results/predictions.npz")
    print("saved", {k: v for k, v in res.items() if k in ("cpl_f1_macro", "cpl_ece")})


if __name__ == "__main__":
    main()
