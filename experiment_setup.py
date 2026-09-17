"""Fold construction and evaluation-set assembly.

Subject-disjoint and patient-disjoint boundaries are enforced jointly: the
wearers whose recordings host the evaluation beats are never seen during
training, and the donor morphology used at evaluation comes from the DS2
patients, which never contribute a training beat either.
"""

import pickle
import numpy as np

import config as C
import build_corpus as bc
import data_mhealth as dmh
import data_mitbih as dmit

CACHE = "cache/raw.pkl"


def load_raw(force=False):
    try:
        if force:
            raise FileNotFoundError
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        raw = {"subs": [dmh.load_subject(s) for s in C.MHEALTH_SUBJECTS],
               "ds1": dmit.load_split(C.DS1),
               "ds2": dmit.load_split(C.DS2)}
        with open(CACHE, "wb") as f:
            pickle.dump(raw, f)
        return raw


def folds(n_folds=5):
    """Leave-two-wearers-out partitions of the MHEALTH cohort."""
    order = list(range(len(C.MHEALTH_SUBJECTS)))
    per = len(order) // n_folds
    return [order[i * per:(i + 1) * per] for i in range(n_folds)]


def make_fold(raw, held, seed=0, kappa_train=None, graft_rate=None,
              cap=700, n_test_records=None):
    kappa_train = C.KAPPA_TRAIN if kappa_train is None else kappa_train
    subs = raw["subs"]
    tr_subs = [s for i, s in enumerate(subs) if i not in held]
    te_subs = [s for i, s in enumerate(subs) if i in held]
    ds1, ds2 = raw["ds1"], raw["ds2"]
    if n_test_records:
        ds2 = ds2[:n_test_records]

    clients = bc.build_clients(tr_subs, ds1, ds1, kappa_train, seed=seed,
                               cap_per_record=cap, graft_rate=graft_rate)

    tests = {}
    for tag, kap in (("cpl", C.KAPPA_EVAL), ("adv", C.KAPPA_ADVERSE)):
        te = bc.build_clients(te_subs, [], ds2, kap, seed=seed + 100,
                              graft_rate=C.GRAFT_RATE_EVAL,
                              base_prior=bc.EVAL_PRIOR)
        tests[tag] = bc.pooled(te)

    ext = bc.build_clients([], ds2, ds2, 0.0, seed=seed + 200, cap_per_record=cap)
    tests["ext"] = bc.pooled(ext)
    return clients, tests


def describe(clients, tests):
    tot = sum(len(c["ecg"]) for c in clients)
    lab = sum(int((c["rhythm"] >= 0).sum()) for c in clients)
    print(f"train clients={len(clients)} beats={tot} rhythm-labelled={lab}")
    for k, v in tests.items():
        m = v["rhythm"] >= 0
        print(f"  {k}: beats={len(v['rhythm'])} labelled={int(m.sum())} "
              f"classes={np.bincount(v['rhythm'][m], minlength=4).tolist()}")
