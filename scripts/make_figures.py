"""HW1 report figures. Run from DeepDataMiningLearning/llm after posthoc.py / posthoc2.py:
    python make_figures.py <results_dir> <figures_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, F = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = Path("outputs/nanolm/sweeps")
C = {"rnn": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a", "lstm_512": "#eda100",
     "gpt_textbook": "#e87ba4", "gpt_modern": "#008300"}
LABEL = {"rnn": "RNN 2×256", "gru": "GRU 2×256", "lstm": "LSTM 2×256", "lstm_512": "LSTM 2×512",
         "gpt_textbook": "GPT-2-style", "gpt_modern": "Modern Transformer"}
ORDER = list(C)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.8,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "savefig.dpi": 220, "savefig.bbox": "tight"})


def curve(name, seeds=(1337, 42), sweep="char_models"):
    frames = [pd.read_json(ROOT / sweep / f"{name}_s{s}" / "eval_log.jsonl", lines=True) for s in seeds]
    x = frames[0].train_tokens.values / 1e6
    return x, np.stack([f.bpb.values for f in frames]), np.stack([f.loss.values for f in frames])


def train_loss(name, seeds=(1337, 42), sweep="char_models", window=5):
    frames = [pd.read_json(ROOT / sweep / f"{name}_s{s}" / "train_log.jsonl", lines=True) for s in seeds]
    step = frames[0].step.values
    y = np.mean([f.loss.rolling(window, min_periods=1).mean().values for f in frames], axis=0)
    return step * 8192 / 1e6, y


s = pd.read_csv(R / "char_models" / "summary.csv")
g = s.groupby("name").val_bpb.agg(["mean", "std"])

# ------------------------------------------------------------------ Fig 1
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1.15]})
a = ax[0]
for n in ORDER:
    x, b, _ = curve(n)
    a.plot(x, b.mean(0), color=C[n], label=LABEL[n])
    a.fill_between(x, b.min(0), b.max(0), color=C[n], alpha=0.2, lw=0)
a.set_ylim(2.05, 2.75)
a.set_xlabel("training tokens (millions)")
a.set_ylabel("validation bits per byte")
a.set_title("A. Learning curves")
h_models, l_models = a.get_legend_handles_labels()

a = ax[1]
order = g.loc[ORDER].sort_values("mean").index[::-1]
ypos = np.arange(len(order))
for i, n in enumerate(order):
    a.errorbar(g.loc[n, "mean"], i, xerr=g.loc[n, "std"], fmt="o", color=C[n], ms=4, capsize=2, elinewidth=1)
    a.text(g.loc[n, "mean"] + 0.012, i + 0.2, f"{g.loc[n, 'mean']:.3f}", fontsize=6.5, color=INK2)
a.set_yticks(ypos)
a.set_yticklabels([LABEL[n] for n in order], fontsize=7)
a.set_xlim(2.08, 2.5)
a.set_xlabel("final val bpb (mean ± sd)")
a.set_title("B. Final bpb")
a.grid(axis="y", visible=False)

a = ax[2]
for n in ("lstm", "lstm_512"):
    x, _, L = curve(n)
    a.plot(x, L.mean(0), color=C[n], label=f"{LABEL[n]} val")
    tx, tl = train_loss(n)
    a.plot(tx, tl, color=C[n], ls=":", label=f"{LABEL[n]} train")
a.set_ylim(0.95, 2.3)
a.set_xlabel("training tokens (millions)")
a.set_ylabel("loss (nats/char)")
a.set_title("C. Train vs. validation")
a.legend(loc="upper right", fontsize=6.3)
fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1))
fig.legend(h_models, l_models, loc="lower center", ncol=6, fontsize=7, columnspacing=1.2, handlelength=1.6)
fig.savefig(F / "fig1_models.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 2: LR sensitivity
rows = []
for sw in ("char_models", "hw1_rnn_controls", "hw1_lr_controls", "hw1_lr_controls2"):
    rows.append(pd.read_csv(R / sw / "summary.csv"))
A = pd.concat(rows)
def series(names):
    out = []
    for lr, n in names:
        v = A[A.name == n].val_bpb
        out.append((lr, v.mean(), v.std(ddof=1)))
    return np.array(out)
S = {"RNN 2×256 (0.28M)": ("rnn", "-", [(5e-4, "rnn_lr5e-4"), (1e-3, "rnn_lr1e-3"), (2e-3, "rnn"), (4e-3, "rnn_lr4e-3"), (8e-3, "rnn_lr8e-3"), (1.6e-2, "rnn_lr1.6e-2")]),
     "RNN 2×512 (1.08M)": ("rnn", "--", [(2e-3, "rnn_512_paramsmatch"), (4e-3, "rnn512_lr4e-3"), (8e-3, "rnn512_lr8e-3")]),
     "GRU 2×256 (0.81M)": ("gru", "-", [(1e-3, "gru_lr1e-3"), (2e-3, "gru"), (4e-3, "gru_lr4e-3"), (8e-3, "gru_lr8e-3")]),
     "LSTM 2×256 (1.07M)": ("lstm", "-", [(1e-3, "lstm_lr1e-3"), (2e-3, "lstm"), (4e-3, "lstm_lr4e-3"), (8e-3, "lstm_lr8e-3")]),
     "Modern Transformer (3.16M)": ("gpt_modern", "-", [(5e-4, "gpt_modern_lr5e-4"), (1e-3, "gpt_modern"), (2e-3, "gpt_modern_lr2e-3"), (4e-3, "gpt_modern_lr4e-3")])}
fig, a = plt.subplots(figsize=(7.2, 2.5))
for lab, (c, ls, names) in S.items():
    d = series(names)
    a.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2], color=C[c], ls=ls, marker="o", ms=3.5, capsize=2, elinewidth=0.8, label=lab)
a.set_xscale("log")
a.set_ylim(2.05, 2.75)
a.set_xticks([5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
a.set_xticklabels(["5e-4", "1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"])
a.minorticks_off()
a.set_xlabel("learning rate (log scale)")
a.set_ylabel("final val bits per byte")
a.legend(ncol=2, loc="upper right")
a.set_title("Final bpb vs. learning rate (2 seeds, mean ± sd; y axis clipped at 2.75)")
fig.tight_layout()
fig.savefig(F / "fig2_lr.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 3: context
c = pd.read_csv(R / "hw1_context" / "summary.csv")
G = pd.read_csv(R / "context_grid.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=False)
Ts = [64, 128, 256, 512]
a = ax[0]
for arch, col, lab in (("lstm", C["lstm"], "LSTM 2×256"), ("gpt", C["gpt_modern"], "Modern Transformer")):
    m = [c[c.name == f"{arch}_T{T}"].val_bpb.mean() for T in Ts]
    e = [c[c.name == f"{arch}_T{T}"].val_bpb.std(ddof=1) for T in Ts]
    a.errorbar(Ts, m, yerr=e, color=col, marker="o", ms=4, capsize=2, label=lab)
a.set_xscale("log", base=2)
a.set_xticks(Ts)
a.set_xticklabels(Ts)
a.minorticks_off()
a.set_xlabel("context length train.seq_len (chars); eval window = train window")
a.set_ylabel("validation bits per byte")
a.set_title("A. Assignment metric (3 seeds, mean ± sd)")
a.legend()
a = ax[1]
Ws = np.array(Ts)
fit = {}
for arch, col, lab, Tt in (("lstm", C["lstm"], "LSTM", 512), ("gpt", C["gpt_modern"], "Transformer", 512)):
    x = G[(G.arch == arch) & (G.train_T == Tt)].groupby("eval_W").bpb.agg(["mean", "std"])
    b, cc = np.polyfit(1 / Ws, x["mean"].values, 1)[::-1]
    fit[arch] = (b, cc)
    a.errorbar(Ws, x["mean"], yerr=x["std"], color=col, marker="o", ms=4, capsize=2, ls="none", label=f"{lab}, trained T=512")
    fw = np.geomspace(60, 540, 40)
    a.plot(fw, b + cc / fw, color=col, lw=1.1, ls="--", label=f"{lab} fit: {b:.3f} + {cc:.1f}/W")
a.set_xscale("log", base=2)
a.set_xticks(Ts)
a.set_xticklabels(Ts)
a.minorticks_off()
a.set_xlabel("evaluation window W (chars), same model")
a.set_title("B. Same checkpoints re-scored: bpb ≈ a + c/W")
a.legend(fontsize=6.3)
fig.tight_layout(w_pad=1.5)
fig.savefig(F / "fig3_context.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 4: gradient reach
d = json.loads((R / "grad_reach.json").read_text())
fig, a = plt.subplots(figsize=(3.5, 2.4))
for n, lab in (("rnn", "RNN"), ("gru", "GRU"), ("lstm", "LSTM"), ("gpt_modern", "Modern Transformer")):
    y = np.array(d[n]["mean"])
    k = np.arange(len(y))
    a.plot(k[1:], y[1:], color=C[n], label=lab, lw=1.4)
a.set_yscale("log")
a.set_ylim(1e-9, 2)
a.set_xlabel("distance k back from the predicted token (chars)")
a.set_ylabel("gradient norm rel. to k=0")
a.set_title("Input-gradient reach (trained models)")
a.legend(loc="lower left")
fig.tight_layout()
fig.savefig(F / "fig4_gradreach.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig 5: selected attention heads
from matplotlib.colors import LinearSegmentedColormap
from nanolm.generate import load_run
from nanolm.viz import attention_maps
import torch
A_ = json.loads((R / "attention.json").read_text())
text = A_["texts"]["dialogue"]
cfg, model, tok, dinfo, _ = load_run(str(ROOT / "char_models" / "gpt_modern_s1337"), device="cpu")
maps = attention_maps(model.float(), torch.tensor([tok.encode(text)]))
heads = [(0, 1, "L0·H1  previous char"), (0, 3, "L0·H3  last space"), (2, 3, "L2·H3  word start"), (3, 2, "L3·H2  speaker / punctuation")]
cm = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6", "#0b2f5e"])
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95))
for a, (l, h, t) in zip(ax, heads):
    a.imshow(maps[l][h], cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    a.set_title(t, fontsize=7.2)
    a.grid(False)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel("attended position", fontsize=6.5)
ax[0].set_ylabel("query position", fontsize=6.5)
fig.tight_layout(w_pad=0.6)
fig.savefig(F / "fig5_heads.png")
plt.close(fig)
print({k: (round(v[0], 4), round(v[1], 2)) for k, v in fit.items()})
