"""Descriptive statistics of the corpus, written to results/corpus_stats.json."""

import json
import numpy as np

import config as C
import build_corpus as bc
import data_mhealth as dmh
import experiment_setup as E


def main():
    raw = E.load_raw()
    folds = E.folds(5)
    clients, tests = E.make_fold(raw, folds[0], seed=0, cap=400)
    P = bc.pooled(clients)
    stats = {}

    mm = [c for c in clients if c["kind"] == "multimodal"]
    eo = [c for c in clients if c["kind"] == "ecg_only"]
    stats["n_clients"] = len(clients)
    stats["n_multimodal"] = len(mm)
    stats["n_ecg_only"] = len(eo)
    stats["train_beats"] = int(len(P["ecg"]))
    stats["train_labelled"] = int((P["rhythm"] >= 0).sum())
    stats["train_class"] = np.bincount(P["rhythm"][P["rhythm"] >= 0],
                                       minlength=C.N_RHYTHM).tolist()
    stats["train_activity_beats"] = int((P["activity"] >= 0).sum())
    for tag in ("cpl", "adv", "ext"):
        d = tests[tag]
        m = d["rhythm"] >= 0
        stats[tag] = {
            "beats": int(len(d["rhythm"])),
            "labelled": int(m.sum()),
            "class": np.bincount(d["rhythm"][m], minlength=C.N_RHYTHM).tolist(),
        }
        if (d["activity"] >= 0).any():
            inten = np.asarray(C.ACTIVITY_INTENSITY)[np.clip(d["activity"], 0, 11)]
            stats[tag]["ectopy_by_stratum"] = [
                round(float((d["rhythm"][m & (inten == s)] > 0).mean()), 3)
                for s in range(3)]
            stats[tag]["beats_by_stratum"] = [int((m & (inten == s)).sum())
                                             for s in range(3)]
    inten_tr = np.asarray(C.ACTIVITY_INTENSITY)[np.clip(P["activity"], 0, 11)]
    sel = (P["rhythm"] >= 0) & (P["activity"] >= 0)
    stats["train_ectopy_by_stratum"] = [
        round(float((P["rhythm"][sel & (inten_tr == s)] > 0).mean()), 3)
        for s in range(3)]

    hr, act, mot = [], [], []
    for s in raw["subs"]:
        hr.append(s["hr"]); act.append(s["activity"])
        mot.append(dmh.motion_energy(s["ctx"]))
    hr, act, mot = np.concatenate(hr), np.concatenate(act), np.concatenate(mot)
    it = np.asarray(C.ACTIVITY_INTENSITY)[act]
    stats["hr_by_stratum"] = [round(float(hr[it == s].mean()), 1) for s in range(3)]
    stats["hr_sd_by_stratum"] = [round(float(hr[it == s].std()), 1) for s in range(3)]
    stats["motion_by_stratum"] = [round(float(np.median(mot[it == s])), 3)
                                  for s in range(3)]
    stats["wearer_beats"] = int(len(hr))
    stats["n_wearers"] = len(raw["subs"])
    stats["n_records_train"] = len(raw["ds1"])
    stats["n_records_test"] = len(raw["ds2"])

    import torch
    from models import STRATUM
    m = STRATUM()
    stats["params_total"] = int(sum(p.numel() for p in m.parameters()))
    for g in ("phi", "psi", "omega"):
        stats["params_" + g] = int(sum(p.numel() for n, p in m.named_parameters()
                                       if n.startswith(g)))
    with open("results/corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
