"""Publication figures.

Everything is drawn in grayscale. Categories are separated by hatch pattern,
line style and marker rather than by hue, so the figures survive monochrome
reproduction. Legends are placed outside the data area or in reserved space and
every panel is laid out with explicit padding, so no annotation, legend or
block overlaps another element.
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import config as C

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 7.2,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.8,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "savefig.dpi": 600,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": "0.85",
    "axes.axisbelow": True,
})

GREY = ["0.15", "0.40", "0.60", "0.78", "0.30", "0.50", "0.70", "0.88", "0.22", "0.65"]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo", "***", "|||", "---"]
LINES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
MARKS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

COL = 3.5          # single-column width in inches
DBL = 7.16         # full-width figure


def out(name):
    return os.path.join(C.FIGURE_DIR, name)


def load(tag):
    path = os.path.join(C.RESULT_DIR, f"{tag}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def agg(rows, metric, key="method"):
    d = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int, float)) and np.isfinite(v):
            name = r[key] + ("|" + r["setting"] if r.get("setting") else "")
            d[name].append(v)
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                len(v)) for k, v in d.items()}


# ===========================================================================
# Fig. 1 -- tiered architecture
# ===========================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(DBL, 2.95))
    ax.set_xlim(-5, 100); ax.set_ylim(-11, 46); ax.axis("off")

    def box(x, y, w, h, label, sub=None, fc="white", ls="-", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45",
                                    linewidth=0.8, edgecolor="black",
                                    facecolor=fc, linestyle=ls))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), label, ha="center",
                va="center", fontsize=fs)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=5.8, style="italic")

    def arrow(x1, y1, x2, y2, style="-|>", ls="-", lw=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=7, linewidth=lw,
                                     linestyle=ls, color="black",
                                     shrinkA=1, shrinkB=1))

    # device tier
    dev_x = [3, 21, 39, 60, 78]
    dev_lab = [("Node 1", "ECG + IMU"), ("Node 2", "ECG + IMU"),
               ("Node 3", "ECG + IMU"), ("Node m", "ECG only"),
               ("Node M", "ECG only")]
    for x, (l, s) in zip(dev_x, dev_lab):
        box(x, 2, 16, 8.5, l, s, fc="0.96")
    ax.text(-4.2, 6.2, "Device\ntier", fontsize=6.6, ha="left", va="center",
            rotation=90, fontweight="bold")

    # edge tier
    box(9, 18, 26, 9, "Edge cell 1", "multimodal population", fc="0.90")
    box(56, 18, 30, 9, "Edge cell E", "rhythm-only population", fc="0.90")
    ax.text(-4.2, 22.5, "Edge\ntier", fontsize=6.6, ha="left", va="center",
            rotation=90, fontweight="bold")

    # cloud tier
    box(26, 34, 46, 9.5, "Cloud aggregator",
        "variance-stratified conflict resolution", fc="0.82")
    ax.text(-4.2, 38.5, "Cloud\ntier", fontsize=6.6, ha="left", va="center",
            rotation=90, fontweight="bold")

    for x in dev_x[:3]:
        arrow(x + 8, 10.5, 22, 18)
    for x in dev_x[3:]:
        arrow(x + 8, 10.5, 71, 18)
    arrow(22, 27, 45, 34)
    arrow(71, 27, 56, 34)
    arrow(45, 34, 22, 27, ls=(0, (2, 1.6)))
    arrow(56, 34, 71, 27, ls=(0, (2, 1.6)))

    ax.text(36.5, 30.8, r"$\phi$ up / down", fontsize=6.0, ha="center")
    ax.text(63.0, 30.8, r"$\phi$ up / down", fontsize=6.0, ha="center")
    ax.text(22, 14.4, r"$\phi,\psi$", fontsize=6.0, ha="center")
    ax.text(71, 14.4, r"$\phi,\psi$", fontsize=6.0, ha="center")

    legend = ("Shared representation  $\\phi$ : aggregated at the cloud\n"
              "Context adapter  $\\psi$ : aggregated inside one cell only\n"
              "Device calibration  $\\omega$ : never transmitted")
    ax.text(47.5, -10.6, legend, fontsize=6.2, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.4", linewidth=0.5))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out("fig1_architecture.png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ===========================================================================
# Fig. 2 -- mediation structure and the model graph
# ===========================================================================
def fig_mediation():
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.45))

    # ---- panel (a): generative structure --------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(-1.4, 8.6); ax.axis("off")
    ax.set_title("(a) Generative structure of an ambulatory recording", pad=5)

    def node(x, y, t, r=0.66):
        ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor="black",
                                linewidth=0.8, zorder=3))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.4, zorder=4)

    def edge(p, q, ls="-", lw=0.9, lab=None, lab_xy=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=lw, linestyle=ls, color="black",
                                     shrinkA=11, shrinkB=11, zorder=2))
        if lab:
            ax.text(lab_xy[0], lab_xy[1], lab, ha="center", fontsize=6.2)

    node(1.5, 4.2, "$a$"); node(5.0, 7.0, "$v$")
    node(5.0, 1.4, r"$\xi$"); node(8.5, 4.2, "$y$")
    edge((1.5, 4.2), (5.0, 7.0), lab="physiological", lab_xy=(2.75, 6.25))
    edge((5.0, 7.0), (8.5, 4.2), lab="rate evidence", lab_xy=(7.35, 6.25))
    edge((1.5, 4.2), (5.0, 1.4), lab="motion artifact", lab_xy=(2.70, 1.85))
    edge((5.0, 1.4), (8.5, 4.2), ls=(0, (2.6, 1.5)), lw=1.3,
         lab="direct path", lab_xy=(7.45, 1.85))
    ax.text(1.5, 3.05, "activity", fontsize=6.0, ha="center")
    ax.text(5.0, 7.95, "rate", fontsize=6.0, ha="center")
    ax.text(5.0, 0.42, "artifact", fontsize=6.0, ha="center")
    ax.text(8.5, 3.05, "rhythm", fontsize=6.0, ha="center")
    ax.text(5.0, -1.25, "the dashed path carries no rhythm information and is\n"
                        "what the mediation constraint is designed to close",
            fontsize=6.1, ha="center", va="bottom", style="italic")

    # ---- panel (b): information flow ------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 11.0); ax.set_ylim(-1.4, 8.6); ax.axis("off")
    ax.set_title("(b) Information flow through the network", pad=5)

    def blk(x, y, w, h, t, fc="0.94", fs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                                    linewidth=0.8, edgecolor="black",
                                    facecolor=fc, zorder=3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=fs, zorder=4)

    def ar(p, q, ls="-", lw=0.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5,
                                     linewidth=lw, linestyle=ls, color="black",
                                     shrinkA=1.5, shrinkB=1.5, zorder=2))

    blk(0.10, 6.25, 1.65, 1.15, "ECG\nwindow")
    blk(0.10, 0.85, 1.65, 1.15, "inertial\nwindow")
    blk(2.35, 6.25, 1.75, 1.15, r"$f_{\phi}$ rhythm" + "\nencoder")
    blk(2.35, 0.85, 1.75, 1.15, r"$g_{\phi}$ context" + "\nencoder")
    blk(4.75, 3.45, 1.55, 1.05, r"$\psi$ adapter" + "\n" + r"$\oplus\ \omega$",
        fc="0.86")
    blk(7.05, 6.45, 1.70, 0.95, "rate head")
    blk(7.05, 4.45, 1.70, 0.95, "rhythm head", fc="0.86")
    blk(7.05, 2.30, 1.70, 0.95, "probe $q$", fc="white")
    blk(7.05, 0.85, 1.70, 0.95, "activity head")

    ar((1.75, 6.83), (2.35, 6.83))
    ar((1.75, 1.43), (2.35, 1.43))
    ar((4.10, 6.60), (4.75, 4.50))          # z^r into the adapter
    ar((4.10, 1.60), (4.75, 3.45), ls=(0, (2, 1.4)))   # context modulation
    ar((4.10, 1.25), (6.92, 1.25))          # context to the activity head
    ar((6.30, 4.35), (7.05, 6.62))          # calibrated embedding to rate head
    ar((6.30, 4.05), (7.05, 4.80))          # and to the rhythm head
    ar((6.30, 3.70), (7.05, 2.72), ls=(0, (1, 1.3)))
    ar((7.90, 6.45), (7.90, 5.40), lw=1.1)  # the mediator
    ar((8.98, 6.92), (9.70, 6.92))
    ar((8.98, 4.92), (9.70, 4.92))
    ar((8.98, 1.32), (9.70, 1.32))
    ax.text(9.95, 6.92, r"$\hat{v}$", fontsize=7.2, va="center")
    ax.text(9.95, 4.92, r"$\hat{y}$", fontsize=7.2, va="center")
    ax.text(9.95, 1.32, r"$\hat{a}$", fontsize=7.2, va="center")
    ax.text(4.50, 5.70, "$z^{r}$", fontsize=6.6)
    ax.text(4.12, 2.68, "$z^{c}$", fontsize=6.6)
    ax.text(8.02, 5.88, r"$\hat{v}$", fontsize=6.6, ha="left")
    ax.text(5.50, -1.25,
            "solid: learned signal path;  dashed: context modulation;\n"
            "dotted: the probe that measures residual context information",
            fontsize=6.1, ha="center", va="bottom")

    fig.subplots_adjust(wspace=0.10)
    fig.savefig(out("fig2_mediation.png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ===========================================================================
# Fig. 3 -- corpus construction
# ===========================================================================
def fig_corpus():
    import experiment_setup as E
    import build_corpus as bc
    import data_mhealth as dmh
    import data_mitbih as dmit

    raw = E.load_raw()
    sub = raw["subs"][1]
    pool = dmit.donor_pool(raw["ds1"])
    rng = np.random.default_rng(3)

    import signal_utils as su
    beats = su.zscore(sub["beats"])
    templates, glob = dmh.beat_templates({"beats": beats, "activity": sub["activity"]})
    vig = np.where(np.asarray(C.ACTIVITY_INTENSITY)[sub["activity"]] == 2)[0]
    i = int(vig[len(vig) // 2])
    tmpl = templates.get(int(sub["activity"][i]), glob)
    host = beats[i]
    donor = pool[2][0][rng.integers(len(pool[2][0]))]
    a_t = float(np.squeeze(su.robust_amplitude(tmpl)))
    a_d = float(np.squeeze(su.robust_amplitude(donor))) + 1e-6
    graft = su.zscore(((a_t / a_d) * donor + (host - tmpl))[None, :])[0]
    t = np.arange(C.BEAT_LEN) / C.FS

    fig = plt.figure(figsize=(DBL, 2.20))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    for k, (sig, lab, ls) in enumerate([
            (host, "host beat (moving wearer)", "-"),
            (donor, "donor beat (annotated ectopic)", "--"),
            (graft, "grafted observation", "-.")]):
        ax.plot(t, sig + 6.5 - 3.2 * k, ls, color=GREY[k], linewidth=0.9)
        ax.text(0.02, 6.5 - 3.2 * k + 1.55, lab, fontsize=6.0)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title("(a) Graft operator", pad=5)
    ax.set_ylim(-4.6, 9.2)
    ax.grid(axis="x")

    ax = fig.add_subplot(gs[0, 1])
    allact, allmot, allhr = [], [], []
    for s in raw["subs"]:
        allact.append(s["activity"]); allhr.append(s["hr"])
        allmot.append(dmh.motion_energy(s["ctx"]))
    act = np.concatenate(allact); mot = np.concatenate(allmot)
    hr = np.concatenate(allhr)
    inten = np.asarray(C.ACTIVITY_INTENSITY)[act]
    data = [np.log10(mot[inten == i] + 1e-3) for i in range(3)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for p, g, h in zip(bp["boxes"], GREY[:3], HATCH[:3]):
        p.set(facecolor=g, hatch=h, linewidth=0.7, edgecolor="black")
    for w in bp["medians"]:
        w.set(color="white", linewidth=1.1)
    ax.set_xticklabels(["sedentary", "moderate", "vigorous"])
    ax.set_ylabel("$\\log_{10}$ motion energy")
    ax.set_title("(b) Motion by stratum", pad=5)

    ax = fig.add_subplot(gs[0, 2])
    means = [hr[inten == i].mean() for i in range(3)]
    sds = [hr[inten == i].std() for i in range(3)]
    ax.bar(range(3), means, yerr=sds, capsize=2.2, width=0.55,
           color=GREY[:3], edgecolor="black", linewidth=0.7,
           hatch=HATCH[:3], error_kw=dict(elinewidth=0.7, capthick=0.7))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["sedentary", "moderate", "vigorous"])
    ax.set_ylabel("heart rate (bpm)")
    ax.set_ylim(0, max(means) + max(sds) + 18)
    ax.set_title("(c) Rate by stratum", pad=5)
    for i, m in enumerate(means):
        ax.text(i, m + sds[i] + 2.5, f"{m:.0f}", ha="center", fontsize=6.2)

    fig.savefig(out("fig3_corpus.png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ===========================================================================
# Fig. 4 -- main comparison
# ===========================================================================
ORDER = ["Local", "FedAvg", "FedProx", "FedBN-Per", "HierFAVG", "HierFAVG-TC",
         "Hier-MTL-Adapter", "Hier-IRM", "STRATUM", "Centralised"]
SHORT = {"Local": "Local", "FedAvg": "FedAvg", "FedProx": "FedProx",
         "FedBN-Per": "FedBN-Per", "HierFAVG": "HierFAVG",
         "HierFAVG-TC": "HierFAVG-TC", "Hier-MTL-Adapter": "Hier-Adapter",
         "Hier-IRM": "Hier-IRM", "STRATUM": "STRATUM", "Centralised": "Pooled"}


def fig_main():
    rows = load("main")
    if not rows:
        return
    panels = [("cpl_f1_macro", "macro $F_1$, coupled", True),
              ("cpl_far_vig", "FAR, vigorous", False),
              ("cpl_sens", "detection sensitivity", True),
              ("ext_mcc", "MCC, external set", True)]
    fig, axes = plt.subplots(2, 2, figsize=(DBL, 3.60))
    for ax, (metric, label, hib) in zip(axes.ravel(), panels):
        st = agg(rows, metric)
        names = [n for n in ORDER if n in st]
        mu = [st[n][0] for n in names]
        sd = [st[n][1] for n in names]
        xs = np.arange(len(names))
        cols = ["0.25" if n == "STRATUM" else ("0.92" if n == "Centralised" else "0.68")
                for n in names]
        hats = ["" if n == "STRATUM" else ("///" if n == "Centralised" else "")
                for n in names]
        ax.bar(xs, mu, yerr=sd, width=0.68, color=cols, edgecolor="black",
               linewidth=0.7, hatch=hats, capsize=2.0,
               error_kw=dict(elinewidth=0.65, capthick=0.65))
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[n] for n in names], rotation=38, ha="right")
        ax.set_ylabel(label)
        top = max(m + s for m, s in zip(mu, sd))
        ax.set_ylim(0, top * 1.16)
        ax.margins(x=0.02)
    fig.subplots_adjust(hspace=0.78, wspace=0.28)
    fig.savefig(out("fig4_main.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ===========================================================================
# Fig. 5 -- behaviour across motion strata
# ===========================================================================
def fig_strata():
    rows = load("main")
    if not rows:
        return
    subset = ["FedAvg", "HierFAVG-TC", "Hier-MTL-Adapter", "Hier-IRM", "STRATUM"]
    strata = ["sed", "mod", "vig"]
    labels = ["sedentary", "moderate", "vigorous"]

    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.18))

    ax = axes[0]
    for i, m in enumerate(subset):
        st = [agg(rows, f"cpl_far_{s}").get(m, (np.nan, 0, 0)) for s in strata]
        ax.errorbar(range(3), [v[0] for v in st], yerr=[v[1] for v in st],
                    linestyle=LINES[i % len(LINES)], marker=MARKS[i], ms=3.2,
                    color=GREY[i], capsize=2.0, elinewidth=0.6, linewidth=1.0,
                    label=SHORT.get(m, m))
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("false alarm rate"); ax.set_title("(a) False alarms", pad=4)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for i, m in enumerate(subset):
        st = [agg(rows, f"cpl_f1_{s}").get(m, (np.nan, 0, 0)) for s in strata]
        ax.errorbar(range(3), [v[0] for v in st], yerr=[v[1] for v in st],
                    linestyle=LINES[i % len(LINES)], marker=MARKS[i], ms=3.2,
                    color=GREY[i], capsize=2.0, elinewidth=0.6, linewidth=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("macro $F_1$"); ax.set_title("(b) Rhythm accuracy", pad=4)

    ax = axes[2]
    w = 0.34
    xs = np.arange(len(subset))
    for j, (tag, lab) in enumerate((("cpl", "association removed"),
                                    ("adv", "association reversed"))):
        st = [agg(rows, f"{tag}_f1_macro").get(m, (np.nan, 0, 0)) for m in subset]
        ax.bar(xs + (j - 0.5) * w, [v[0] for v in st], yerr=[v[1] for v in st],
               width=w, color=GREY[j * 3], edgecolor="black", linewidth=0.7,
               hatch=HATCH[j], capsize=1.8, label=lab,
               error_kw=dict(elinewidth=0.6, capthick=0.6))
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(m, m) for m in subset], rotation=38, ha="right")
    ax.set_ylabel("macro $F_1$")
    ax.set_title("(c) Confound shift", pad=4)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="0.4",
              borderpad=0.35, handlelength=1.5)
    ax.set_ylim(0, 0.78)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.38, bottom=0.30, top=0.90)
    fig.savefig(out("fig5_strata.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ===========================================================================
# Fig. 6 -- ablation
# ===========================================================================
def fig_ablation():
    rows = load("ablation")
    if not rows:
        return
    f1 = agg(rows, "cpl_f1_macro")
    far = agg(rows, "cpl_far_vig")
    if "Full" not in f1:
        return
    names = [n for n in f1 if n != "Full"]
    names.sort(key=lambda n: (far.get(n, (0, 0, 0))[0] - far["Full"][0]))
    d_f1 = [f1[n][0] - f1["Full"][0] for n in names]
    e_f1 = [np.hypot(f1[n][1], f1["Full"][1]) for n in names]
    d_far = [far.get(n, (np.nan, 0, 0))[0] - far["Full"][0] for n in names]
    e_far = [np.hypot(far.get(n, (0, 0, 0))[1], far["Full"][1]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.45), sharey=True)
    ys = np.arange(len(names))
    for ax, d, e, lab, sign in ((axes[0], d_f1, e_f1, "$\\Delta$ macro $F_1$", 1),
                                (axes[1], d_far, e_far,
                                 "$\\Delta$ false alarm rate (vigorous)", -1)):
        cols = ["0.35" if sign * v < 0 else "0.78" for v in d]
        ax.barh(ys, d, xerr=e, height=0.62, color=cols, edgecolor="black",
                linewidth=0.7, capsize=1.8,
                error_kw=dict(elinewidth=0.6, capthick=0.6))
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(lab)
        ax.margins(x=0.20)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=0.10)
    fig.savefig(out("fig6_ablation.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ===========================================================================
# Fig. 7 -- sensitivity
# ===========================================================================
def fig_sensitivity():
    rows = load("sensitivity")
    topo = load("topology")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.10))

    ax = axes[0]
    kaps = [0.4, 0.6, 0.8, 1.0]
    far = agg(rows, "cpl_far_vig")
    for i, m in enumerate(("HierFAVG-TC", "STRATUM")):
        mu = [far.get(f"{m}|kappa={k}", (np.nan, 0, 0))[0] for k in kaps]
        sd = [far.get(f"{m}|kappa={k}", (np.nan, 0, 0))[1] for k in kaps]
        ax.errorbar(kaps, mu, yerr=sd, linestyle=LINES[i], marker=MARKS[i],
                    ms=3.4, color=("0.12" if i == 0 else "0.52"), capsize=2.0,
                    elinewidth=0.6, label=SHORT.get(m, m))
    ax.set_xlabel("training coupling $\\kappa$")
    ax.set_ylabel("FAR, vigorous")
    ax.set_title("(a) Confound strength", pad=4)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="0.4",
              borderpad=0.3, handlelength=1.8)

    ax = axes[1]
    ws = [0.0, 0.15, 0.35, 0.6]
    f1a = agg(rows, "cpl_f1_macro")
    abl = agg(load("ablation"), "cpl_f1_macro")
    abl_far = agg(load("ablation"), "cpl_far_vig")

    def pick(store, w):
        if abs(w - 0.35) < 1e-9:
            return store.get("Full", (np.nan, 0, 0))
        return store.get(f"STRATUM|w_adv={w}", (np.nan, 0, 0))

    st_far = [pick(far if f"STRATUM|w_adv={w}" in far else abl_far, w) for w in ws]
    st_f1 = [pick(f1a if f"STRATUM|w_adv={w}" in f1a else abl, w) for w in ws]
    ax.errorbar(ws, [v[0] for v in st_far], yerr=[v[1] for v in st_far],
                linestyle="-", marker="o", ms=3.4, color="0.15", capsize=2.0,
                elinewidth=0.6, label="FAR, vigorous")
    ax2 = ax.twinx()
    ax2.errorbar(ws, [v[0] for v in st_f1], yerr=[v[1] for v in st_f1],
                 linestyle="--", marker="s", ms=3.4, color="0.60", capsize=2.0,
                 elinewidth=0.6, label="macro $F_1$")
    ax2.grid(False)
    ax.set_xlabel("invariance weight $\\lambda_{c}$")
    ax.set_ylabel("FAR, vigorous")
    ax2.set_ylabel("macro $F_1$", labelpad=1.0)
    ax.set_ylim(0, max(v[0] for v in st_far) * 1.45)
    ax.set_title("(b) Invariance weight", pad=4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True,
              framealpha=1.0, edgecolor="0.4", borderpad=0.3, handlelength=1.8)

    ax = axes[2]
    cells = [2, 4, 6]
    tf1 = agg(topo, "cpl_f1_macro")
    teta = agg(topo, "eta_mean")
    v = [tf1.get(f"STRATUM|cells={c}", (np.nan, 0, 0)) for c in cells]
    ax.bar(range(3), [x[0] for x in v], yerr=[x[1] for x in v], width=0.5,
           color=GREY[:3], edgecolor="black", linewidth=0.7, hatch=HATCH[:3],
           capsize=2.0, error_kw=dict(elinewidth=0.6, capthick=0.6))
    ax.set_xticks(range(3)); ax.set_xticklabels([str(c) for c in cells])
    ax.set_xlabel("number of cells $E$")
    ax.set_ylabel("macro $F_1$", labelpad=1.0)
    top = max(x[0] + x[1] for x in v if np.isfinite(x[0]))
    ax.set_ylim(0, top * 1.30)
    ax3 = ax.twinx()
    e = [teta.get(f"STRATUM|cells={c}", (np.nan, 0, 0))[0] for c in cells]
    ax3.plot(range(3), e, linestyle="-", marker="D", ms=3.4, color="0.10",
             linewidth=1.0)
    ax3.set_ylabel("mean $\\eta$")
    ax3.set_ylim(0, max([x for x in e if np.isfinite(x)] or [1]) * 1.9)
    ax3.grid(False)
    ax.set_title("(c) Cells and dispersion", pad=4)
    fig.subplots_adjust(wspace=0.78)
    fig.savefig(out("fig7_sensitivity.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ===========================================================================
# Fig. 8 -- calibration
# ===========================================================================
def fig_calibration(npz="results/predictions.npz"):
    if not os.path.exists(npz):
        return
    import calibration as cal
    import metrics as M
    d = np.load(npz, allow_pickle=True)
    logits, y, stratum = d["logits"], d["y"], d["stratum"]
    temps, t_glob = d["temps"], float(d["t_global"])

    variants = [("uncalibrated", np.ones(3)),
                ("single temperature", np.full(3, t_glob)),
                ("per-stratum temperature", temps)]
    fig, axes = plt.subplots(1, 4, figsize=(DBL, 2.00),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
    for ax, (name, tt) in zip(axes[:3], variants):
        probs = cal.apply_temperature(logits, tt, stratum)
        ax.plot([0, 1], [0, 1], color="0.55", linestyle=(0, (3, 2)), linewidth=0.8)
        for s, (lab, mk) in enumerate((("sedentary", "o"), ("moderate", "s"),
                                       ("vigorous", "^"))):
            m = stratum == s
            if m.sum() < 30:
                continue
            xs, ys, ns = M.reliability_curve(y[m], probs[m], n_bins=8)
            ax.plot(xs, ys, marker=mk, ms=3.0, linestyle=LINES[s],
                    color=GREY[s], linewidth=1.0, label=lab)
        ax.set_xlim(0.3, 1.02); ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("confidence")
        ax.set_title(name, pad=4)
    axes[0].set_ylabel("empirical accuracy")
    axes[0].legend(loc="upper left", frameon=True, framealpha=1.0,
                   edgecolor="0.4", borderpad=0.3, handlelength=1.6)

    ax = axes[3]
    eces = []
    for name, tt in variants:
        probs = cal.apply_temperature(logits, tt, stratum)
        eces.append(M.calibration_metrics(y, probs)["ece"])
    ax.bar(range(3), eces, width=0.55, color=GREY[:3], edgecolor="black",
           linewidth=0.7, hatch=HATCH[:3])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["none", "single", "per-stratum"], rotation=24, ha="right")
    ax.set_ylabel("expected calibration error")
    ax.set_ylim(0, max(eces) * 1.32)
    for i, v in enumerate(eces):
        ax.text(i, v + max(eces) * 0.045, f"{v:.3f}", ha="center", fontsize=6.2)
    ax.set_title("calibration error", pad=4)
    fig.subplots_adjust(wspace=0.40)
    fig.savefig(out("fig8_calibration.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ===========================================================================
# Fig. 9 -- aggregation diagnostics and error structure
# ===========================================================================
def fig_diagnostics():
    rows = load("main")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.10),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    ax = axes[0]
    curves = [r["eta_curve"] for r in rows
              if r["method"] == "STRATUM" and r.get("eta_curve")]
    if curves:
        L = min(len(c) for c in curves)
        arr = np.array([c[:L] for c in curves])
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        xs = np.arange(1, L + 1)
        ax.plot(xs, mu, color="0.15", linewidth=1.1, marker="o", ms=2.6)
        ax.fill_between(xs, mu - sd, mu + sd, color="0.80", linewidth=0)
        ax.set_xlabel("cloud round")
        ax.set_ylabel("stratification coefficient $\\eta_{t}$")
        ax.set_ylim(0, 1.0)
        ax.set_title("(a) Aggregation diagnostic", pad=4)

    for ax, method, tag in ((axes[1], "HierFAVG-TC", "(b) Coverage-weighted nested"),
                            (axes[2], "STRATUM", "(c) STRATUM")):
        mats = [np.array(r["cpl_confusion"]) for r in rows
                if r["method"] == method and "cpl_confusion" in r]
        if not mats:
            continue
        cm = np.sum(mats, axis=0).astype(float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(C.AAMI_CLASSES); ax.set_yticklabels(C.AAMI_CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("reference")
        ax.set_title(tag, pad=4)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if cmn[i, j] > 0.55 else "black")
    fig.subplots_adjust(wspace=0.46)
    fig.savefig(out("fig9_diagnostics.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_all():
    fig_architecture()
    fig_mediation()
    fig_corpus()
    fig_main()
    fig_strata()
    fig_ablation()
    fig_sensitivity()
    fig_calibration()
    fig_diagnostics()


if __name__ == "__main__":
    build_all()
