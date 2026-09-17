"""Orchestration of hierarchical training and evaluation.

One entry point, :func:`run`, covers every configuration reported in the paper.
A variant is described by a dictionary, so the proposed framework, its
ablations and all competing federated schemes traverse exactly the same data
pipeline, the same number of optimisation steps and the same evaluation code.
"""

import copy
import time
import numpy as np
import torch

import config as C
import calibration as cal
import federated as fed
import metrics as M
from models import STRATUM

torch.set_num_threads(2)


DEFAULT = dict(
    topology="hier",        # hier | flat | local | central
    aggregation="strat",    # strat | fedavg
    mediated=True,
    adapter=True,
    personalise=True,       # keep omega and normalisation statistics on device
    calib="cond",           # cond | global | none
    task_surgery=False,     # PCGrad-style surgery across task gradients
    w_a=C.W_ACTIVITY, w_v=C.W_VITAL, w_adv=C.W_ADVERSARY, w_cf=C.W_COUNTERFACTUAL,
    w_prox=0.0, lam_grl=1.0, lr=C.LR, batch=C.BATCH_SIZE,
    rounds=25, tau2=2, local_steps=8, n_cells=C.N_EDGE_CELLS, mixed_cells=False,
    client_frac=0.5, task_coverage=True, irm=False, w_irm=1.0,
    warmup_frac=0.34, ema_frac=0.34, ema_decay=0.65,
)


# ---------------------------------------------------------------------------
def _split_client(client, rng, val_frac=0.15):
    n = len(client["ecg"])
    idx = rng.permutation(n)
    n_val = max(int(val_frac * n), 1)
    va, tr = idx[:n_val], idx[n_val:]
    keys = ("ecg", "ctx", "has_ctx", "activity", "hr", "rhythm")
    a = {k: client[k][tr] for k in keys}
    b = {k: client[k][va] for k in keys}
    for d in (a, b):
        d["cid"], d["kind"] = client["cid"], client["kind"]
    return a, b


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _load(model, *dicts):
    sd = model.state_dict()
    for d in dicts:
        for k, v in d.items():
            sd[k] = v.clone()
    model.load_state_dict(sd)


def _pick(state, prefixes):
    return {k: v.clone() for k, v in state.items()
            if any(k.startswith(p) for p in prefixes)}


def _buffers(model, state):
    names = {n for n, _ in model.named_buffers()}
    return {k: v.clone() for k, v in state.items() if k in names}


def _class_weights(clients):
    counts = np.zeros(C.N_RHYTHM)
    for c in clients:
        y = c["rhythm"]
        counts += np.bincount(y[y >= 0], minlength=C.N_RHYTHM)
    # Inverse square-root frequency: full inverse weighting is dominated by the
    # fusion class, which is two orders of magnitude rarer than normal beats and
    # would otherwise squash every other weight against the clipping bound.
    inv = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    w = inv / np.median(inv)
    return torch.tensor(np.clip(w, 0.3, 8.0), dtype=torch.float32)


# ---------------------------------------------------------------------------
def run(clients, tests, cfg=None, seed=0, verbose=False, dump=None):
    cfg = {**DEFAULT, **(cfg or {})}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    hr_all = np.concatenate([c["hr"] for c in clients])
    hr_mu, hr_sd = float(hr_all.mean()), float(hr_all.std() + 1e-6)

    train_parts, val_parts = zip(*[_split_client(c, rng) for c in clients])
    train_parts, val_parts = list(train_parts), list(val_parts)
    cw = _class_weights(train_parts)

    if cfg["topology"] == "central":
        # Pooling the federation into one client would otherwise give that client
        # the step budget of a single node. The local step count is scaled by the
        # number of nodes that a federated round would have visited, so the
        # reference point consumes the same number of gradient evaluations.
        n_visited = max(int(round(cfg["client_frac"] * len(train_parts))), 1)
        cfg = {**cfg, "local_steps": cfg["local_steps"] * n_visited}
        merged = {k: np.concatenate([c[k] for c in train_parts])
                  for k in ("ecg", "ctx", "has_ctx", "activity", "hr", "rhythm")}
        merged["cid"], merged["kind"] = "central", "multimodal"
        train_parts = [merged]

    # --- edge assignment -------------------------------------------------
    import build_corpus as bc
    if cfg["topology"] in ("flat", "central", "local"):
        cell_of = {c["cid"]: 0 for c in train_parts}
        n_cells = 1
    else:
        cell_of = bc.assign_edges(train_parts, cfg["n_cells"], cfg["mixed_cells"], seed)
        n_cells = cfg["n_cells"]

    cells = {e: [c for c in train_parts if cell_of[c["cid"]] == e]
             for e in range(n_cells)}
    cells = {e: v for e, v in cells.items() if v}

    # --- cell activity marginals for the importance correction -----------
    for e, members in cells.items():
        acts = np.concatenate([c["activity"][c["activity"] >= 0] for c in members]) \
            if any((c["activity"] >= 0).any() for c in members) else np.array([], int)
        if len(acts):
            pi = np.bincount(acts, minlength=C.N_ACTIVITY).astype(np.float64)
            pi = pi / pi.sum()
        else:
            pi = np.ones(C.N_ACTIVITY) / C.N_ACTIVITY
        for c in members:
            c["iw"] = (fed.importance_weights(c, pi) if cfg["aggregation"] == "strat"
                       else np.ones(len(c["ecg"]), np.float32))

    tensors = {c["cid"]: fed.to_tensors(c, hr_mu, hr_sd) for c in train_parts}
    sizes = {c["cid"]: (fed.effective_size(c["iw"]) if cfg["aggregation"] == "strat"
                        else float(len(c["ecg"]))) for c in train_parts}

    # --- state -----------------------------------------------------------
    model = STRATUM(use_adapter=cfg["adapter"], mediated=cfg["mediated"])
    init = _state(model)
    phi_g = _pick(init, ["phi"])
    psi_c = {e: _pick(init, ["psi"]) for e in cells}
    omega = {c["cid"]: _pick(init, ["omega"]) for c in train_parts}
    bufs = {c["cid"]: _buffers(model, init) for c in train_parts}
    phi_keys = sorted(phi_g.keys())

    history = {"eta": [], "round": [], "val_f1": []}
    t0 = time.time()
    uplink_floats = 0

    ema_start = int(round((1.0 - cfg["ema_frac"]) * cfg["rounds"]))
    phi_ema = None

    for rnd in range(cfg["rounds"]):
        # The invariance and consistency terms are introduced gradually. Both act
        # against a representation that is still forming, and imposing them from
        # the first round costs accuracy on every task without advancing the
        # constraint, because there is as yet nothing for the probe to remove.
        ramp = min(1.0, (rnd + 1) / max(cfg["warmup_frac"] * cfg["rounds"], 1.0))
        round_cfg = {**cfg, "w_adv": cfg["w_adv"] * ramp,
                     "w_cf": cfg["w_cf"] * ramp}
        client_delta, cell_delta, cell_w, cell_cover = {}, {}, {}, {}

        for e, members in cells.items():
            phi_e = {k: v.clone() for k, v in phi_g.items()}
            psi_e = psi_c[e]
            for inner in range(cfg["tau2"]):
                states, ws = [], []
                n_sel = max(int(round(cfg["client_frac"] * len(members))), 1)
                sel = members if len(members) <= 2 else [
                    members[i] for i in rng.choice(len(members), n_sel, replace=False)]
                for c in sel:
                    cid = c["cid"]
                    _load(model, phi_e, psi_e, omega[cid], bufs[cid])
                    fed.local_train(model, tensors[cid], cw, round_cfg,
                                    cfg["local_steps"], rng,
                                    global_phi=None if cfg["w_prox"] == 0 else
                                    {k: v for k, v in phi_e.items()})
                    st = _state(model)
                    states.append(st)
                    ws.append(sizes[cid])
                    omega[cid] = _pick(st, ["omega"]) if cfg["personalise"] else omega[cid]
                    bufs[cid] = _buffers(model, st)
                    uplink_floats += sum(v.numel() for k, v in st.items()
                                         if k.startswith("phi") or k.startswith("psi"))
                    if inner == cfg["tau2"] - 1:
                        client_delta[cid] = {k: st[k] - phi_e[k] for k in phi_keys}
                if cfg["task_coverage"] or cfg["aggregation"] == "strat":
                    cts = [fed.coverage_counts(c) for c in sel]
                    agg = fed.weighted_mean_blocks(states, cts, ["phi", "psi"], ws)
                else:
                    agg = fed.weighted_mean(states, ws, ["phi", "psi"])
                phi_e = {k: v for k, v in agg.items() if k.startswith("phi")}
                psi_e = {k: v for k, v in agg.items() if k.startswith("psi")}
                if not cfg["personalise"]:
                    shared_om = fed.weighted_mean(states, ws, ["omega"])
                    for c in members:
                        omega[c["cid"]] = shared_om
            psi_c[e] = psi_e
            cell_delta[e] = {k: phi_e[k] - phi_g[k] for k in phi_keys}
            cell_w[e] = sum(sizes[c["cid"]] for c in members)
            cts = [fed.coverage_counts(c) for c in members]
            cell_cover[e] = {t_: float(sum(c[t_] for c in cts))
                             for t_ in ("all", "rhythm", "activity", "context")}

        if cfg["topology"] == "local":
            phi_g = phi_g                     # no aggregation above the device
            for cid, d in client_delta.items():
                pass
        elif cfg["aggregation"] == "strat" and len(cell_delta) > 1:
            eta = np.mean([fed.dispersion_ratio(client_delta, cell_delta,
                                                 cell_of, gk)
                           for gk in fed.group_keys(phi_keys).values()])
            merged = fed.conflict_projection(
                cell_delta, cell_w, phi_keys, eta,
                cell_cover if cfg["task_coverage"] else None)
            phi_g = {k: phi_g[k] + merged[k] for k in phi_keys}
            history["eta"].append(float(eta))
        else:
            cells_l = list(cell_delta)
            base_w = np.array([cell_w[e] for e in cells_l], dtype=np.float64)
            base_w = base_w / base_w.sum()
            new = {}
            for k in phi_keys:
                if cfg["task_coverage"]:
                    task = fed.block_of(k)
                    w = np.array([cell_cover[e][task] for e in cells_l], dtype=np.float64)
                    w = base_w if w.sum() <= 0 else w / w.sum()
                else:
                    w = base_w
                new[k] = phi_g[k] + sum(float(wi) * cell_delta[e][k]
                                        for e, wi in zip(cells_l, w))
            phi_g = new
            history["eta"].append(float("nan"))
        history["round"].append(rnd)
        # Averaging the shared block over the closing rounds removes the
        # round-to-round oscillation that partial participation produces and is
        # applied identically to every scheme compared in this study.
        if rnd >= ema_start:
            if phi_ema is None:
                phi_ema = {k: v.detach().clone().float() for k, v in phi_g.items()}
            else:
                d = cfg["ema_decay"]
                for k in phi_ema:
                    phi_ema[k] = d * phi_ema[k] + (1.0 - d) * phi_g[k].float()

    if phi_ema is not None:
        phi_g = {k: phi_ema[k].to(phi_g[k].dtype) for k in phi_g}

    train_time = time.time() - t0

    # --- assemble the deployed model -------------------------------------
    # The normalisation statistics that ship with the deployed model are the
    # ones carried inside the aggregated shared block, because those were
    # combined with the per-block coverage weights. Averaging the device-local
    # statistics instead would fold in the context-branch statistics of nodes
    # that never observed an inertial channel, and the activity head would then
    # be evaluated against statistics estimated on constant input.
    psi_deploy = psi_c[0] if 0 in psi_c else list(psi_c.values())[0]
    om_mean = {k: torch.stack([omega[c][k].float() for c in omega]).mean(0)
               for k in next(iter(omega.values()))}
    _load(model, phi_g, psi_deploy, om_mean)
    model.eval()

    if cfg["topology"] == "local":
        # Local-only: every device keeps its own model; report the mean over devices.
        return _evaluate_local(model, phi_g, psi_deploy, omega, bufs, train_parts,
                               tensors, tests, hr_mu, hr_sd, cfg, cw, rng,
                               train_time, uplink_floats, history, val_parts)

    return _evaluate(model, tests, val_parts, hr_mu, hr_sd, cfg,
                     train_time, uplink_floats, history, dump=dump)


# ---------------------------------------------------------------------------
def _infer(model, data, hr_mu, hr_sd, bs=512):
    outs = {"logit_y": [], "logit_a": [], "v_hat": []}
    with torch.no_grad():
        n = data["ecg"].shape[0]
        for a in range(0, n, bs):
            b = slice(a, min(a + bs, n))
            o = model(torch.from_numpy(data["ecg"][b]).unsqueeze(1),
                      torch.from_numpy(data["ctx"][b]),
                      torch.from_numpy(data["has_ctx"][b]))
            for k in outs:
                outs[k].append(o[k].numpy())
    out = {k: np.concatenate(v) for k, v in outs.items()}
    out["hr_hat"] = out["v_hat"] * hr_sd + hr_mu
    return out


def _evaluate(model, tests, val_parts, hr_mu, hr_sd, cfg, train_time,
              uplink_floats, history, dump=None):
    val = {k: np.concatenate([c[k] for c in val_parts])
           for k in ("ecg", "ctx", "has_ctx", "activity", "hr", "rhythm")}
    ov = _infer(model, val, hr_mu, hr_sd)
    mv = val["rhythm"] >= 0
    strata_v = cal.predicted_strata(ov["logit_a"][mv])
    if cfg["calib"] == "cond":
        temps, _ = cal.fit_conditional(ov["logit_y"][mv], val["rhythm"][mv], strata_v)
    elif cfg["calib"] == "global":
        t = cal.fit_temperature(ov["logit_y"][mv], val["rhythm"][mv])
        temps = np.full(cal.N_STRATA, t)
    else:
        temps = np.ones(cal.N_STRATA)
    t_global = cal.fit_temperature(ov["logit_y"][mv], val["rhythm"][mv])

    res = {"train_time_s": train_time,
           "uplink_mfloats": uplink_floats / 1e6,
           "eta_mean": float(np.nanmean(history["eta"])) if history["eta"] else float("nan"),
           "eta_curve": history["eta"],
           "temps": temps.tolist()}

    for name, ds in tests.items():
        o = _infer(model, ds, hr_mu, hr_sd)
        m = ds["rhythm"] >= 0
        res.update(M.rhythm_metrics(ds["rhythm"][m], o["logit_y"][m], f"{name}_"))
        res.update(M.binary_metrics(ds["rhythm"][m], o["logit_y"][m], f"{name}_"))
        probs = cal.apply_temperature(o["logit_y"][m], temps,
                                      cal.predicted_strata(o["logit_a"][m]))
        res.update(M.calibration_metrics(ds["rhythm"][m], probs, prefix=f"{name}_"))
        res.update(M.vital_metrics(ds["hr"], o["hr_hat"], f"{name}_hr_"))
        ma = ds["activity"] >= 0
        if ma.any():
            res.update(M.activity_metrics(ds["activity"][ma], o["logit_a"][ma],
                                          f"{name}_act_"))
            inten = np.asarray(C.ACTIVITY_INTENSITY)[np.clip(ds["activity"], 0, 11)]
            for lvl, tag in ((0, "sed"), (1, "mod"), (2, "vig")):
                sel = m & (inten == lvl)
                if sel.sum() > 20:
                    res[f"{name}_f1_{tag}"] = M.rhythm_metrics(
                        ds["rhythm"][sel], o["logit_y"][sel])["f1_macro"]
                    res[f"{name}_far_{tag}"] = M.false_alarm_rate(
                        ds["rhythm"][sel], o["logit_y"][sel], inten[sel], lvl)
                    res[f"{name}_sens_{tag}"] = M.binary_metrics(
                        ds["rhythm"][sel], o["logit_y"][sel])["sens"]
        res[f"{name}_confusion"] = M.confusion(ds["rhythm"][m], o["logit_y"][m]).tolist()
        if dump and name == "cpl":
            np.savez(dump, logits=o["logit_y"][m], y=ds["rhythm"][m],
                     stratum=cal.predicted_strata(o["logit_a"][m]),
                     temps=np.asarray(temps), t_global=t_global)
    return res


def _evaluate_local(model, phi_g, psi_deploy, omega, bufs, train_parts, tensors,
                    tests, hr_mu, hr_sd, cfg, cw, rng, train_time, uplink,
                    history, val_parts):
    """Device-only baseline: no parameter leaves the node."""
    import federated as fedmod
    per = []
    for c in train_parts[:4]:
        cid = c["cid"]
        m2 = STRATUM(use_adapter=cfg["adapter"], mediated=cfg["mediated"])
        _load(m2, phi_g, psi_deploy, omega[cid], bufs[cid])
        fedmod.local_train(m2, tensors[cid], cw, cfg,
                           cfg["rounds"] * cfg["tau2"] * cfg["local_steps"], rng)
        m2.eval()
        per.append(_evaluate(m2, tests, [c], hr_mu, hr_sd, cfg, train_time, 0,
                             {"eta": []}))
    out = {}
    for k in per[0]:
        vals = [p[k] for p in per]
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            out[k] = float(np.nanmean(vals))
    out["uplink_mfloats"] = 0.0
    out["train_time_s"] = train_time
    out["eta_mean"] = float("nan")
    out["eta_curve"] = []
    return out
