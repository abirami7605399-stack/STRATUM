"""Construction of the coupled wearable corpus.

The corpus joins two real recordings rather than simulating either of them.
Host streams come from MHEALTH and supply the activity label, the inertial
context, the instantaneous rate and — critically — the motion artifact that a
chest electrode actually produces while the wearer is moving. Donor beats come
from MIT-BIH and supply arrhythmic morphology together with its expert AAMI
annotation.

A host beat is replaced by

    x = alpha * donor + (host_beat - template(subject, activity)),

so the artifact residual of the host recording is preserved and only the
morphology is exchanged. The same operator is applied to normal and to ectopic
donors, which means the splice itself carries no class information and cannot
be exploited as a cue.

The prevalence of ectopic donors is made to depend on the activity intensity
through a coupling coefficient kappa. Training uses kappa > 0, so that ectopy
is over-represented during vigorous movement exactly as an artifact-driven
shortcut would suggest; evaluation uses kappa = 0 or kappa < 0, which breaks or
reverses the association and exposes any model that learned it.
"""

import numpy as np

import config as C
import data_mitbih as dmit
import data_mhealth as dmh
import signal_utils as su

# Marginal AAMI prevalence observed in the training records, used as the
# baseline donor prior before the activity coupling is applied.
BASE_PRIOR = np.array([0.895, 0.028, 0.077])

# The evaluation hosts are class-enriched so that per-class recall and the
# macro-averaged scores are estimated from an adequate number of beats. Natural
# prevalence is retained on the external electrocardiogram-only partition.
EVAL_PRIOR = np.array([0.55, 0.18, 0.27])


def coupled_prior(intensity, kappa, base=None):
    """Donor class prior for a beat recorded at the given activity intensity."""
    p = (BASE_PRIOR if base is None else base).copy()
    ref = p.copy()
    shift = np.exp(kappa * (intensity - 1.0))
    # The ectopic mass is shifted jointly and bounded away from zero and one, so
    # that every stratum of every evaluation set retains beats of both kinds and
    # the stratified false alarm rate stays estimable.
    q = float(np.clip(ref[1:].sum() * shift, 0.03, 0.85))
    p[1:] = ref[1:] * (q / max(ref[1:].sum(), 1e-9))
    p[0] = 1.0 - q
    return p / p.sum()


def graft_subject(sub, pool, rng, kappa, graft_rate=None, base_prior=None):
    """Return the grafted beat matrix and the rhythm labels for one subject.

    An ectopic donor also carries its prematurity ratio. The host window is
    re-cut around a centre advanced by that ratio, so the preceding beat appears
    closer and the following beat further away inside the same window; the
    coupling between morphology and timing that defines an ectopic beat is
    therefore preserved rather than discarded by the graft.
    """
    graft_rate = C.GRAFT_RATE if graft_rate is None else graft_rate
    raw = sub["ecg_raw"]
    peaks = sub["peaks"]
    beats = su.zscore(sub["beats"]).astype(np.float32)
    templates, glob = dmh.beat_templates({"beats": beats,
                                          "activity": sub["activity"]})
    n = len(beats)
    y = np.full(n, -1, dtype=np.int64)
    out = beats.copy()
    half = C.BEAT_LEN // 2

    rr_host = np.diff(peaks, prepend=peaks[0] - int(0.8 * C.FS))
    if len(rr_host) > 1:
        rr_host[0] = rr_host[1]

    idx = rng.permutation(n)[: int(round(graft_rate * n))]
    intens = np.asarray(C.ACTIVITY_INTENSITY)[sub["activity"]]

    for i in idx:
        prior = coupled_prior(intens[i], kappa, base_prior)
        cls = int(rng.choice(C.N_RHYTHM, p=prior))
        d_beats, d_prem = pool[cls]
        if len(d_beats) == 0:
            continue
        j = rng.integers(len(d_beats))
        donor, prem = d_beats[j], float(d_prem[j])

        shift = int(round((1.0 - prem) * rr_host[i]))
        c = int(peaks[i]) - shift
        if c - half < 0 or c - half + C.BEAT_LEN > len(raw):
            c = int(peaks[i])
        host_win = su.zscore(raw[c - half:c - half + C.BEAT_LEN][None, :])[0]

        tmpl = templates.get(int(sub["activity"][i]), glob)
        resid = host_win - tmpl
        a_t = float(np.squeeze(su.robust_amplitude(tmpl)))
        a_d = float(np.squeeze(su.robust_amplitude(donor))) + 1e-6
        out[i] = (a_t / a_d) * donor + resid
        y[i] = cls

    return su.zscore(out).astype(np.float32), y


def build_clients(mhealth_subjects, mitdb_records, donor_records, kappa,
                  seed=0, cap_per_record=700, graft_rate=None, verbose=False,
                  base_prior=None):
    """Assemble the federation: multimodal host clients and ECG-only clients."""
    rng = np.random.default_rng(seed)
    pool = dmit.donor_pool(donor_records)
    clients = []

    for sub in mhealth_subjects:
        x, y = graft_subject(sub, pool, rng, kappa, graft_rate, base_prior)
        clients.append({
            "cid": f"A{sub['subject']:02d}",
            "kind": "multimodal",
            "ecg": x,
            "ctx": sub["ctx"].astype(np.float32),
            "has_ctx": np.ones(len(x), np.float32),
            "activity": sub["activity"].astype(np.int64),
            "hr": sub["hr"].astype(np.float32),
            "rhythm": y,
        })

    for rec in mitdb_records:
        n = len(rec["labels"])
        sel = rng.permutation(n)[:cap_per_record] if n > cap_per_record else np.arange(n)
        clients.append({
            "cid": f"B{rec['record']}",
            "kind": "ecg_only",
            "ecg": rec["beats"][sel],
            "ctx": np.zeros((len(sel), C.N_IMU_CH, C.CTX_LEN_DS), np.float32),
            "has_ctx": np.zeros(len(sel), np.float32),
            "activity": np.full(len(sel), -1, np.int64),
            "hr": rec["hr"][sel],
            "rhythm": rec["labels"][sel],
        })

    if verbose:
        na = sum(1 for c in clients if c["kind"] == "multimodal")
        tot = sum(len(c["ecg"]) for c in clients)
        print(f"  {len(clients)} clients ({na} multimodal), {tot} beats")
    return clients


def pooled(clients, keys=("ecg", "ctx", "has_ctx", "activity", "hr", "rhythm")):
    return {k: np.concatenate([c[k] for c in clients]) for k in keys}


def assign_edges(clients, n_cells=None, mixed=False, seed=0):
    """Map every client onto an edge cell.

    By default the cells follow deployment capability: multimodal hosts share a
    cell and the electrocardiogram-only nodes are distributed over the
    remaining cells. The mixed option interleaves both kinds and is used in the
    topology sensitivity study.
    """
    n_cells = n_cells or C.N_EDGE_CELLS
    rng = np.random.default_rng(seed)
    if mixed:
        order = rng.permutation(len(clients))
        return {clients[i]["cid"]: int(j % n_cells) for j, i in enumerate(order)}
    amap, bmap = {}, {}
    a = [c for c in clients if c["kind"] == "multimodal"]
    b = [c for c in clients if c["kind"] == "ecg_only"]
    for c in a:
        amap[c["cid"]] = 0
    for j, c in enumerate(rng.permutation(len(b))):
        bmap[b[c]["cid"]] = 1 + (j % max(n_cells - 1, 1))
    amap.update(bmap)
    return amap
