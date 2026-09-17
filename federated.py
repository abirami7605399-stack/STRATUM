"""Federated optimisation: device objective, edge aggregation, cloud aggregation.

The three tiers do different work. A device minimises the coupled multi-task
objective on its own beats. A cell aggregates the shared representation and its
own context adapter using effective sample sizes computed after the activity
mixture of each device has been re-weighted towards the cell marginal. The
cloud aggregates only the shared representation across cells, resolving the
conflicts between cell updates with a strength read off an analysis of variance
of the update dispersion within and between cells.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from models import STRATUM


# ---------------------------------------------------------------------------
# Device-side data handling
# ---------------------------------------------------------------------------
def to_tensors(client, hr_mu, hr_sd):
    return {
        "e": torch.from_numpy(client["ecg"]).unsqueeze(1),
        "m": torch.from_numpy(client["ctx"]),
        "h": torch.from_numpy(client["has_ctx"]),
        "a": torch.from_numpy(client["activity"]),
        "v": torch.from_numpy((client["hr"] - hr_mu) / hr_sd),
        "y": torch.from_numpy(client["rhythm"]),
        "u": torch.from_numpy(client.get("iw", np.ones(len(client["ecg"]), np.float32))),
    }


def importance_weights(client, pi_cell, clip=(0.2, 5.0)):
    """Ratio between the cell activity marginal and the device marginal."""
    a = client["activity"]
    if (a < 0).all():
        return np.ones(len(a), np.float32)
    counts = np.bincount(a[a >= 0], minlength=C.N_ACTIVITY).astype(np.float64)
    pi_k = counts / max(counts.sum(), 1.0)
    w = np.ones(len(a), np.float64)
    ok = a >= 0
    w[ok] = pi_cell[a[ok]] / np.maximum(pi_k[a[ok]], 1e-6)
    return np.clip(w, *clip).astype(np.float32)


def effective_size(w):
    s1, s2 = float(w.sum()), float((w ** 2).sum())
    return (s1 ** 2) / max(s2, 1e-9)


# ---------------------------------------------------------------------------
# Local objective
# ---------------------------------------------------------------------------
def local_objective(model, batch, cw, cfg, global_phi=None):
    out = model(batch["e"], batch["m"], batch["h"])
    losses = {}

    y, u = batch["y"], batch["u"]
    my = y >= 0
    if my.any():
        ce = F.cross_entropy(out["logit_y"][my], y[my], weight=cw, reduction="none")
        losses["y"] = (ce * u[my]).sum() / u[my].sum().clamp_min(1e-6)
    else:
        losses["y"] = torch.zeros((), dtype=torch.float32)

    a = batch["a"]
    ma = a >= 0
    if ma.any() and cfg["w_a"] > 0:
        losses["a"] = F.cross_entropy(out["logit_a"][ma], a[ma])
    else:
        losses["a"] = torch.zeros((), dtype=torch.float32)

    if cfg["w_v"] > 0:
        losses["v"] = F.smooth_l1_loss(out["v_hat"], batch["v"])
    else:
        losses["v"] = torch.zeros((), dtype=torch.float32)

    if ma.any() and cfg["w_adv"] > 0:
        # The probe is fitted on detached features, so it tracks the residual
        # dependence without itself steering the encoder.
        probe = model.adversary(out["z_r"][ma], out["v_hat"][ma],
                                detach_features=True)
        losses["probe"] = F.cross_entropy(probe, a[ma])
        # The encoder is pushed towards a probe posterior that is uninformative.
        # A confusion objective is used rather than a reversed gradient because
        # maximising the probe loss is unbounded above and destabilises training
        # once the representation has become uninformative.
        free = model.adversary(out["z_r"][ma], out["v_hat"][ma])
        logq = F.log_softmax(free, dim=-1)
        losses["adv"] = -logq.mean() - float(np.log(C.N_ACTIVITY))
    else:
        losses["probe"] = torch.zeros((), dtype=torch.float32)
        losses["adv"] = torch.zeros((), dtype=torch.float32)

    if cfg["w_cf"] > 0 and batch["h"].sum() > 1:
        idx = torch.randperm(batch["e"].shape[0])
        cf_logit = model.counterfactual(out, idx)
        p = F.softmax(out["logit_y"], dim=-1)
        q = F.softmax(cf_logit, dim=-1)
        mmix = (0.5 * (p + q)).clamp_min(1e-8)
        # Jensen-Shannon divergence: symmetric and bounded by log 2, so a large
        # disagreement cannot drive the update to an arbitrary magnitude.
        js = 0.5 * ((p * (p.clamp_min(1e-8).log() - mmix.log())).sum(-1)
                    + (q * (q.clamp_min(1e-8).log() - mmix.log())).sum(-1))
        losses["cf"] = js[batch["h"] > 0].mean() if (batch["h"] > 0).any() \
            else js.mean()
    else:
        losses["cf"] = torch.zeros((), dtype=torch.float32)

    if cfg.get("irm", False) and ma.any() and my.any():
        losses["irm"] = _irm_penalty(out["logit_y"], y, batch["a"], cw)
    else:
        losses["irm"] = torch.zeros((), dtype=torch.float32)

    total = (losses["y"] + cfg["w_a"] * losses["a"] + cfg["w_v"] * losses["v"]
             + cfg["w_adv"] * losses["adv"] + cfg["w_cf"] * losses["cf"]
             + losses["probe"] + cfg.get("w_irm", 1.0) * losses["irm"])

    if global_phi is not None and cfg["w_prox"] > 0:
        prox = sum(((p - global_phi[n]) ** 2).sum()
                   for n, p in model.named_parameters() if n.startswith("phi"))
        total = total + cfg["w_prox"] * prox

    return total, losses


def _irm_penalty(logits, y, a, cw):
    """IRMv1 penalty over motion-intensity environments."""
    inten = torch.as_tensor(C.ACTIVITY_INTENSITY)[a.clamp(min=0)]
    pen = torch.zeros((), dtype=torch.float32)
    n_env = 0
    for env in range(3):
        m = (a >= 0) & (inten == env) & (y >= 0)
        if m.sum() < 8:
            continue
        scale = torch.ones(1, requires_grad=True)
        loss = F.cross_entropy(logits[m] * scale, y[m], weight=cw)
        g = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        pen = pen + (g ** 2).sum()
        n_env += 1
    return pen / max(n_env, 1)


# ---------------------------------------------------------------------------
# Device update
# ---------------------------------------------------------------------------
def local_train(model, data, cw, cfg, steps, rng, global_phi=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=C.WEIGHT_DECAY)
    n = data["e"].shape[0]
    bs = min(cfg["batch"], n)
    for _ in range(steps):
        idx = torch.from_numpy(rng.choice(n, bs, replace=n < bs))
        batch = {k: v[idx] for k, v in data.items()}
        if batch["e"].shape[0] < 2:
            continue
        opt.zero_grad()
        loss, _ = local_objective(model, batch, cw, cfg, global_phi)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
BLOCK_TASKS = {
    "phi_ecg": "all", "phi_head_v": "all", "phi_head_y": "rhythm",
    "phi_ctx": "context", "psi_": "context",
    "phi_head_a": "activity", "phi_adv": "activity",
    "omega": "all",
}


def coverage_counts(client):
    """Number of beats on a device that actually supervise each parameter block.

    Averaging a task head over devices that carry no label for that task pulls
    the head back towards its initialisation in proportion to how many such
    devices exist. Counting the supervising beats per block removes that pull
    without changing the aggregation rule itself.
    """
    n = len(client["ecg"])
    return {
        "all": float(n),
        "rhythm": float((client["rhythm"] >= 0).sum()),
        "activity": float((client["activity"] >= 0).sum()),
        "context": float(client["has_ctx"].sum()),
    }


def block_of(key):
    for pre, task in BLOCK_TASKS.items():
        if key.startswith(pre):
            return task
    return "all"


def weighted_mean_blocks(states, counts, prefixes, sizes=None):
    """Aggregate with a separate weight vector for every parameter block."""
    out = {}
    for k in states[0]:
        if not any(k.startswith(p) for p in prefixes):
            continue
        task = block_of(k)
        w = np.asarray([c[task] for c in counts], dtype=np.float64)
        if w.sum() <= 0:
            w = np.asarray(sizes if sizes is not None else
                           [c["all"] for c in counts], dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, wi in zip(states, w):
            acc += float(wi) * s[k].float()
        out[k] = acc.to(states[0][k].dtype)
    return out


def weighted_mean(states, weights, prefixes):
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    out = {}
    for k in states[0]:
        if not any(k.startswith(p) for p in prefixes):
            continue
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, wi in zip(states, w):
            acc += float(wi) * s[k].float()
        out[k] = acc.to(states[0][k].dtype)
    return out


def _flatten(delta, keys):
    return torch.cat([delta[k].reshape(-1) for k in keys])


def group_keys(keys):
    """Partition the shared parameters by the task their block serves."""
    groups = {}
    for k in keys:
        groups.setdefault(block_of(k), []).append(k)
    return groups


def dispersion_ratio(client_deltas, cell_deltas, cell_of, keys):
    """Share of the update dispersion that lies between cells rather than within.

    The statistic is an analysis of variance on the cell updates: when the
    spread between cells dominates the spread of devices inside a cell, the
    disagreement the cloud is asked to reconcile is a population effect and the
    conflict between cell updates is worth resolving; when devices inside a cell
    already disagree that much, the same disagreement is sampling noise and the
    resolution is damped accordingly.
    """
    flat_cell = {e: _flatten(d, keys) for e, d in cell_deltas.items()}
    if len(flat_cell) < 2:
        return 0.0
    mean_all = torch.stack(list(flat_cell.values())).mean(dim=0)
    s_b = float(torch.stack([((v - mean_all) ** 2).sum()
                             for v in flat_cell.values()]).mean())
    diffs = []
    for cid, d in client_deltas.items():
        e = cell_of[cid]
        if e in flat_cell:
            diffs.append(((_flatten(d, keys) - flat_cell[e]) ** 2).sum())
    s_w = float(torch.stack(diffs).mean()) if diffs else 0.0
    return float(np.clip(s_b / max(s_b + s_w, 1e-12), 0.0, 1.0))


def conflict_projection(cell_deltas, weights, keys, eta, block_weights=None):
    """Resolve opposing cell updates separately inside each parameter block.

    Blocks are treated independently because a cell that holds no label for a
    task carries no meaningful update for the block serving that task; mixing
    such a block into a single flattened direction would let an uninformative
    cell rotate the updates of every other block. Only the cells that actually
    supervise a block take part in that block's projection.
    """
    cells = list(cell_deltas.keys())
    default_w = np.asarray([weights[e] for e in cells], dtype=np.float64)
    default_w = default_w / max(default_w.sum(), 1e-12)

    out = {}
    for task, gkeys in group_keys(keys).items():
        if block_weights is None:
            active = list(cells)
            w = default_w.copy()
        else:
            active = [e for e in cells if block_weights[e].get(task, 0.0) > 0]
            if not active:
                active, w = list(cells), default_w.copy()
            else:
                w = np.asarray([block_weights[e][task] for e in active],
                               dtype=np.float64)
                w = w / w.sum()

        flat = {e: _flatten(cell_deltas[e], gkeys).clone() for e in active}
        if len(active) > 1 and eta > 0:
            for e in active:
                for g in active:
                    if e == g:
                        continue
                    ip = float(torch.dot(flat[e], flat[g]))
                    if ip < 0:
                        den = float(torch.dot(flat[g], flat[g])) + 1e-12
                        flat[e] = flat[e] - eta * (ip / den) * flat[g]

        merged = sum(float(wi) * flat[e] for e, wi in zip(active, w))
        off = 0
        for k in gkeys:
            n = cell_deltas[active[0]][k].numel()
            out[k] = merged[off:off + n].view_as(cell_deltas[active[0]][k])
            off += n
    return out
