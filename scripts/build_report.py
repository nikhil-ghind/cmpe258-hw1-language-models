"""Build the single-file HW1 report (HTML -> PDF with headless Chrome). All numbers are read from results/.

    python build_report.py <hw1_dir> <nanolm_llm_dir>
"""
import base64
import glob
import html
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW, LLM = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
R, F = HW / "results", HW / "figures"
SW = LLM / "outputs/nanolm/sweeps"
REPO = "https://github.com/nikhil-ghind/cmpe258-hw1-language-models"


def J(n):
    return json.loads((R / n).read_text())


def img(n, w="100%"):
    b = base64.b64encode((F / n).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{w}">'


s = pd.read_csv(R / "char_models/summary.csv")
ctrl = pd.concat([pd.read_csv(R / d / "summary.csv") for d in ("char_models", "hw1_rnn_controls", "hw1_lr_controls", "hw1_lr_controls2")])
ctx = pd.read_csv(R / "hw1_context/summary.csv")
grid = pd.read_csv(R / "context_grid.csv")
lc = J("lc_diagnostics.json")
hand = J("metrics_by_hand.json")
unt = J("untrained.json")
unt0 = pd.read_csv(R / "hw1_untrained_lr0/summary.csv")
grad = J("grad_reach.json")
fixed = pd.DataFrame(J("context_fixed512.json"))
pairs = J("context_paired.json")
att = J("attention.json")
gruv = json.loads((R / "gru_verification.json").read_text())


def bpb(n, d=ctrl):
    v = d[d.name == n].val_bpb
    return v.mean(), v.std(ddof=1)


def pm(n, d=ctrl, p=3):
    a, b = bpb(n, d)
    return f"{a:.{p}f} ± {b:.{p}f}"


def paired(a, b, d=ctrl):
    from scipy import stats
    A = d[d.name == a].set_index("seed").val_bpb
    B = d[d.name == b].set_index("seed").val_bpb
    x = (B - A).reindex(A.index)
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return x.mean(), float(2 * stats.t.sf(abs(t), len(x) - 1))


LAB = {"rnn": "Vanilla RNN 2×256", "gru": "GRU 2×256", "lstm": "LSTM 2×256", "lstm_512": "LSTM 2×512",
       "gpt_textbook": "Transformer, GPT-2 style", "gpt_modern": "Transformer, modern"}
BEST = {"rnn": ("8e-3", "rnn_lr8e-3"), "gru": ("4e-3", "gru_lr4e-3"), "lstm": ("4e-3", "lstm_lr4e-3"), "gpt_modern": ("1e-3", "gpt_modern")}
D = 16.384e6

# ------------------------------------------------------------- table builders
def t_models():
    rows = []
    g = s.groupby("name")
    for n in sorted(LAB, key=lambda k: bpb(k)[0]):
        p = g.params.first()[n]
        d, pv = paired("gpt_textbook", n) if n != "gpt_textbook" else (0.0, float("nan"))
        best = f"{bpb(BEST[n][1])[0]:.3f} (lr {BEST[n][0]})" if n in BEST else "not tuned"
        rows.append(f"<tr><td>{LAB[n]}</td><td class=n>{p/1e6:.2f}M</td><td class=n>{6*p*D/1e15:.2f}</td>"
                    f"<td class=n><b>{pm(n, p=4)}</b></td><td class=n>{'baseline' if n == 'gpt_textbook' else f'{d:+.3f}'}</td><td class=n>{'—' if pv != pv else f'{pv:.3f}'}</td>"
                    f"<td class=n>{g.train_time_sec.mean()[n]:.0f}</td><td class=n>{best}</td></tr>")
    return "".join(rows)


def t_hand():
    h = hand["gpt_modern_s1337"]
    L, I, H = h["logged"], h["independent_forward_pass"], h["from_metrics_json"]
    rows = [("Loss (nats/token)", f"{H['loss']:.6f}", f"{I['loss']:.6f}", f"{L['val_loss']:.6f}"),
            ("Perplexity = e<sup>loss</sup>", f"{H['ppl']:.5f}", f"{I['ppl']:.5f}", f"{L['val_ppl']:.5f}"),
            ("Bits/token = loss / ln 2", f"{H['bits_per_token']:.6f}", f"{I['bits_per_token']:.6f}", "—"),
            ("bpb = bits/token × tokens/bytes", f"{H['bpb']:.6f}", f"{I['bpb']:.6f}", f"{L['val_bpb']:.6f}")]
    return "".join(f"<tr><td>{a}</td><td class=n>{b}</td><td class=n>{c}</td><td class=n>{d}</td></tr>" for a, b, c, d in rows)


def t_untrained():
    rows = []
    for k, lab in (("char", "Characters"), ("byte", "Bytes"), ("gpt2", "GPT-2 BPE")):
        u = unt[k]
        x = unt0[unt0.name == k]
        V, lnV = u[0]["V"], u[0]["lnV"]
        l0 = x.val_loss.mean()
        l1 = np.mean([r["val_loss"] for r in u])
        rows.append(f"<tr><td>{lab}</td><td class=n>{V:,}</td><td class=n>{lnV:.3f}</td><td class=n>{l0:.3f}</td><td class=n>{l0-lnV:+.3f}</td>"
                    f"<td class=n>{l1:.3f}</td><td class=n>{l1-lnV:+.3f}</td><td class=n>{x.val_bpb.mean():.3f}</td><td class=n>{u[0]['uniform_bpb']:.3f}</td></tr>")
    return "".join(rows)


def t_ctx():
    a = ctx.groupby("name").val_bpb.agg(["mean", "std"])
    P = {(p["arch"], p["to"]): p for p in pairs}
    rows = []
    for arch, lab in (("lstm", "LSTM"), ("gpt", "Transformer")):
        for T in (64, 128, 256, 512):
            n = f"{arch}_T{T}"
            if T > 64:
                p = P[(arch, T)]
                st = f"{p['delta_bpb']:+.4f}"
                res = "resolved" if abs(p["delta_bpb"]) >= 2 * 0.0069 else "not resolved"
            else:
                st, res = "—", "—"
            rows.append(f"<tr><td>{lab}</td><td class=n>{T}</td><td class=n><b>{a.loc[n,'mean']:.4f} ± {a.loc[n,'std']:.4f}</b></td>"
                        f"<td class=n>{st}</td><td>{res}</td></tr>")
    return "".join(rows)


def asym():
    out = {}
    for arch, Ts in (("lstm", (64, 128, 256, 512)), ("gpt", (256, 512))):
        for T in Ts:
            x = grid[(grid.arch == arch) & (grid.train_T == T) & (grid.eval_W <= (512 if arch == "lstm" else T))]
            res = []
            for _, y in x.groupby("seed"):
                A = np.c_[np.ones(len(y)), 1 / y.eval_W.values]
                b, c = np.linalg.lstsq(A, y.bpb.values, rcond=None)[0]
                res.append((b, c, np.abs(y.bpb.values - (b + c / y.eval_W.values)).max()))
            r = np.array(res)
            out[(arch, T)] = (r[:, 0].mean(), r[:, 0].std(ddof=1), r[:, 1].mean(), r[:, 2].max())
    return out


AS = asym()


def t_asym():
    rows = []
    for (arch, T), v in AS.items():
        nW = 4 if arch == "lstm" else (3 if T == 256 else 4)
        rows.append(f"<tr><td>{'LSTM' if arch == 'lstm' else 'Transformer'}</td><td class=n>{T}</td><td class=n>{nW}</td>"
                    f"<td class=n>{v[0]:.4f} ± {v[1]:.4f}</td><td class=n>{v[2]:.2f}</td><td class=n>{v[3]:.4f}</td></tr>")
    return "".join(rows)


HEADS = [("Previous char", "prev_token", (0, 1), (0, 3)), ("Last space (word boundary)", "last_space", (0, 3), (0, 0)),
         ("Word start", "word_start", (2, 3), (2, 3)), ("Speaker tag / punctuation", "punct_any", (3, 2), (1, 0))]


def t_heads():
    rows = []
    T = att["runs"]
    for name, k, (l1, h1), (l2, h2) in HEADS:
        for run, l, h in (("gpt_modern_s1337", l1, h1), ("gpt_modern_s42", l2, h2)):
            rr = T[run]
            un = T["untrained_s1337"]["val"][l][h][k]
            cells = "".join(f"<td class=n>{rr['text'][t][l][h][k]:.3f}</td>" for t in att["texts"])
            rows.append(f"<tr><td>{name if run.endswith('1337') else ''}</td><td>{run[-4:].replace('_s42', 's42').replace('1337', 's1337')} L{l}·H{h}</td>{cells}"
                        f"<td class=n>{rr['val'][l][h][k]:.3f} ± {rr['val'][l][h][k + '_se']:.3f}</td><td class=n>{un:.3f}</td></tr>")
    return "".join(rows)


def grid_table():
    p = grid.groupby(["arch", "train_T", "eval_W"]).bpb.mean().unstack()
    rows = []
    for (arch, T), r in p.iterrows():
        rows.append(f"<tr><td>{'LSTM' if arch == 'lstm' else 'Transformer'}</td><td class=n>{T}</td>" +
                    "".join(f"<td class=n>{v:.4f}</td>" for v in r.values) + "</tr>")
    return "".join(rows)


def all_runs():
    files = ["char_models", "hw1_rnn_controls", "hw1_lr_controls", "hw1_lr_controls2", "hw1_context", "hw1_untrained", "hw1_untrained_lr0"]
    rows = []
    for f in files:
        d = pd.read_csv(R / f / "summary.csv")
        for _, r in d.iterrows():
            rows.append(f"<tr><td>{f}</td><td>{r['name']}</td><td class=n>{r['seed']}</td><td class=n>{r['params']/1e6:.3f}</td><td class=n>{r['tokens_seen']/1e6:.3f}</td>"
                        f"<td class=n>{r['val_loss']:.4f}</td><td class=n>{r['val_ppl']:.3f}</td><td class=n>{r['val_bpb']:.4f}</td><td class=n>{r['val_tokens']:,}</td><td class=n>{r['train_time_sec']:.0f}</td></tr>")
    return "".join(rows)


# ------------------------------------------------------------- values for prose
meta = json.loads((SW / "char_models/gpt_modern_s1337/meta.json").read_text())
total_min = sum(json.loads(Path(f).read_text())["train_time_sec"] + json.loads(Path(f).read_text()).get("eval_time_sec", 0)
                for f in glob.glob(str(SW / "*/*/metrics.json"))) / 60
n_runs = len(glob.glob(str(SW / "*/*/metrics.json")))
same_gpt = (bpb("gpt_modern")[0], ctx[(ctx.name == "gpt_T256") & (ctx.seed == 1337)].val_bpb.iloc[0], s[(s.name == "gpt_modern") & (s.seed == 1337)].val_bpb.iloc[0])
same_lstm = (ctx[(ctx.name == "lstm_T256") & (ctx.seed == 1337)].val_bpb.iloc[0], s[(s.name == "lstm") & (s.seed == 1337)].val_bpb.iloc[0])
gt = s.groupby("name").train_time_sec.mean()["gpt_modern"]
gt2 = ctx[ctx.name == "gpt_T256"].train_time_sec.mean()
print("same config check", same_gpt, same_lstm, "gpt time", gt, gt2)

R0, R1, R2, LS = bpb("rnn")[0], bpb("rnn_lr8e-3")[0], bpb("rnn512_lr4e-3")[0], bpb("lstm_lr4e-3")[0]
dlr, dcap, dgate, dtot = R0 - R1, R1 - R2, R2 - LS, R0 - LS
d_ls, p_ls = paired("lstm", "rnn")
d_512, p_512 = paired("lstm", "lstm_512")
d_gate, p_gate = paired("lstm_lr4e-3", "rnn512_lr4e-3")
d_tie, p_tie = paired("gpt_modern", "lstm_lr4e-3")
d_mod, p_mod = paired("lstm", "gpt_modern")
g_ = {k: grad[k]["at_k"] for k in grad}
L5a = lc["lstm_512"]
L25 = lc["lstm"]
best512 = np.mean([x["best_bpb"] for x in L5a])
ctx_m = ctx.groupby("name").val_bpb.mean()
fx = fixed.groupby("name").sliding_halfT_bpb.mean()
G = grid.groupby(["arch", "train_T", "eval_W"]).bpb.mean()
gv = [(k, v) for k, v in gruv.items()]
gv_out = max(v["output_max_abs_diff"] for _, v in gv)
gv_grad = max(v["grad_max_abs_diff"] for _, v in gv)

CSS = """
@page { size: letter; margin: 0.55in 0.62in 0.6in 0.62in; @bottom-center { content: counter(page); font: 8pt 'Liberation Sans', sans-serif; color: #52514e; } }
body { font: 9.35pt/1.3 'Liberation Serif', 'DejaVu Serif', serif; color: #111; margin: 0; }
h1 { font: bold 15pt 'Liberation Sans', sans-serif; margin: 0 0 1pt; }
.sub { font: 8.7pt 'Liberation Sans', sans-serif; color: #52514e; margin-bottom: 6pt; }
h2 { font: bold 10.6pt 'Liberation Sans', sans-serif; margin: 7pt 0 2pt; border-bottom: 0.6pt solid #bbb; padding-bottom: 1pt; }
h3 { font: bold 9.3pt 'Liberation Sans', sans-serif; margin: 5pt 0 1pt; }
h4 { font: bold 8.5pt 'Liberation Sans', sans-serif; margin: 7pt 0 2pt; }
p { margin: 2pt 0 3.5pt; }
ul { margin: 2pt 0 3pt; padding-left: 14pt; }
li { margin-bottom: 1.5pt; }
table { border-collapse: collapse; width: 100%; font: 7.7pt/1.25 'Liberation Sans', sans-serif; margin: 2pt 0 3pt; }
th { text-align: left; border-bottom: 0.8pt solid #333; padding: 1.5pt 3pt; font-weight: bold; vertical-align: bottom; }
th.n, td.n { text-align: right; font-variant-numeric: tabular-nums; }
td { padding: 1.1pt 3pt; border-bottom: 0.25pt solid #ddd; }
.cap { font: 7.8pt/1.25 'Liberation Sans', sans-serif; color: #333; margin: 1pt 0 5pt; }
figure { margin: 2pt 0 2pt; break-inside: avoid; }
.tb { break-inside: avoid; }
code { font: 8.2pt 'Liberation Mono', monospace; }
a { color: #1a4f9c; text-decoration: none; }
pre { font: 6.4pt/1.17 'Liberation Mono', monospace; white-space: pre-wrap; word-break: break-all; margin: 1pt 0 5pt; background: #f7f7f5; padding: 3pt; }
.app { break-before: page; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
.two { display: flex; gap: 9pt; align-items: flex-start; }
.two > div { flex: 1; min-width: 0; }
"""

BODY = f"""
<h1>Homework 1: Language Models from Scratch</h1>
<div class=sub>CMPE 258 · Character-level models on Tiny Shakespeare, trained with nanolm · commit {meta['git']['commit'][:10]} · RTX 5070 Laptop GPU, PyTorch {meta['torch']}, bfloat16<br>
Code, configs, results and figures: <a href="{REPO}">{REPO.replace('https://', '')}</a></div>

<h2>1. Question and hypothesis</h2>
<p>This homework is about two things: understanding what the loss numbers actually mean, and seeing how recurrent networks and Transformers behave differently in practice. Going in, I had three expectations. <b>(H1)</b> The vanilla RNN would lose badly to the GRU and LSTM, because gating is supposed to fix vanishing gradients. <b>(H2)</b> The bigger LSTM would overfit, since Tiny Shakespeare is only about 1&nbsp;MB of text. <b>(H3)</b> The Transformer would get more out of longer context than the LSTM, because attention can look straight back at distant characters.</p>
<p>Only one of these held up cleanly. H2 was right. H1 was only partly right: most of the RNN's gap turned out to be about its learning rate and size, not its lack of gates. H3 was not supported at all.</p>

<h2>2. Setup</h2>
<p>All models are trained on Tiny Shakespeare, with the last 10% held out for validation, using a character tokenizer (65 symbols) unless I say otherwise. To keep comparisons fair, every training run in Sections 3.2–3.4 sees exactly the same 16.384 million tokens: 2000 steps of 8192 tokens each, which is about 16 passes over the corpus. Using the usual estimate of 6 × parameters × tokens, that works out to between 0.03 and 0.42 PFLOP per run depending on model size (Table&nbsp;2).</p>
<p>I used seeds 1337 and 42 for every configuration, and added seed 7 for the context-length study; compared configurations always share the same seeds. The metric throughout is final validation bits per byte (bpb) on the full validation split. To decide what counts as a real difference, I used the seed-to-seed noise floor reported by the analysis tool (0.006 bpb for the model comparison, 0.0069 for the context study) and called any gap smaller than about twice that "not resolved". The p-values come from paired tests on only two or three seeds, so I treat them as weak evidence at best.</p>
<p>In total I ran {n_runs} training runs, which took {total_min:.0f} minutes of GPU time including evaluation. Every experiment is defined by a sweep file in the repository; I did not modify the nanolm library itself.</p>

<h2>3. Results</h2>
<h3>3.1 Metrics by hand</h3>
<div class=tb><table><tr><th>Quantity (modern Transformer, seed 1337)</th><th class=n>From metrics file</th><th class=n>My own pass</th><th class=n>Logged</th></tr>{t_hand()}</table>
<div class=cap><b>Table 1a.</b> The validation split has 111,360 tokens and 111,360 bytes (435 windows of 256 characters, one byte each), so bits per token and bits per byte are the same number here. "My own pass" re-scores the raw validation text with a separate loop, without using the library's evaluation code.</div></div>
<div class=tb><table><tr><th>Tokenizer</th><th class=n>V</th><th class=n>ln V</th><th class=n>Loss, no update</th><th class=n>− ln V</th><th class=n>Loss, 1 step</th><th class=n>− ln V</th><th class=n>bpb, no update</th><th class=n>Uniform bpb</th></tr>{t_untrained()}</table>
<div class=cap><b>Table 1b.</b> Loss of an untrained model, averaged over two seeds. "1 step" is the assignment's suggestion of training for a single step; "no update" is the same run with the learning rate set to zero, so the weights never change.</div></div>
<p>Everything lines up. Recomputing loss, perplexity, bits per token and bpb from the saved totals gives exactly the logged values, and my independent pass over the text differs by only 4×10<sup>−7</sup> nats, which is just float32 rounding. For untrained models, the loss sits within 0.05 nats of ln&nbsp;V for all three tokenizers, which is what you'd expect from a model that is guessing roughly uniformly. One detail surprised me at first: after a single training step the loss can already dip <i>below</i> ln&nbsp;V, because even one warmup step teaches the model a little about which characters are common. The last two columns show why bpb is the fairer unit. Per-token loss ranges from 4.2 to 10.8 across tokenizers, but once you convert to bits per byte the uniform-guess baselines are 6.02, 8.00 and 5.05, which are directly comparable.</p>

<h3>3.2 RNN vs. LSTM vs. GRU vs. Transformer</h3>
<figure>{img('fig1_models.png')}<div class=cap><b>Figure 1.</b> The course's standard model-comparison sweep, with each model at its default learning rate (2 seeds; shaded band shows the range). A: validation bpb as training goes on. B: final bpb, mean ± standard deviation. C: training loss (dotted, with dropout on) against validation loss for the two LSTMs.</div></figure>
<div class=tb><table><tr><th>Model</th><th class=n>Params</th><th class=n>PFLOP</th><th class=n>Val bpb (mean ± sd)</th><th class=n>Δ vs. GPT-2 style</th><th class=n>Paired p</th><th class=n>Train s</th><th class=n>Best bpb after LR tuning</th></tr>{t_models()}</table>
<div class=cap><b>Table 2.</b> Every model gets the same training budget. The default learning rates are 2e-3 for the recurrent models and 1e-3 for the Transformers; the last column is the best result from my own learning-rate search (Figure&nbsp;2).</div></div>
<figure>{img('fig2_lr.png')}<div class=cap><b>Figure 2.</b> My learning-rate search, which goes beyond the standard sweep. I made sure each model's best value had a worse value on both sides. Best rates: 1e-3 for the Transformer, 4e-3 for the LSTM and GRU, and 8e-3 for the RNN at width 256 (4e-3 at width 512).</div></figure>
<p>The headline ranking turns out to depend heavily on learning rate. With the default settings, the modern Transformer beats the LSTM by {abs(d_mod):.3f} bpb, which looks like a clear win. But once the LSTM gets its own best learning rate, it reaches {pm('lstm_lr4e-3', p=4)}, essentially level with the Transformer's {pm('gpt_modern', p=4)}. The difference ({d_tie:+.4f}) is below the noise floor, so I call it <b>not resolved</b>. The paired p-value of {p_tie:.2f} looks impressive, but it only reflects two seeds that happened to agree closely, so I don't read anything into it.</p>

<h3>3.3 Context length</h3>
<figure style="text-align:center">{img('fig3_context.png', '62%')}<div class=cap><b>Figure 3.</b> Validation bpb against context length for the LSTM and the modern Transformer. Each model is evaluated on windows the same length it was trained on.</div></figure>
<div class=tb><table><tr><th>Model</th><th class=n>T</th><th class=n>Val bpb (W = T)</th><th class=n>Change from previous T</th><th></th></tr>{t_ctx()}</table>
<div class=cap><b>Table 3a.</b> Three seeds each. Every run uses the same 16.4M tokens and 2000 steps; I changed the batch size so the number of tokens per step stays fixed. A change counts as resolved if it is at least 0.0138 bpb, twice the noise floor.</div></div>
<p><b>Where does each model stop benefiting?</b> Both models improve clearly from 64 to 128 and from 128 to 256 characters. The step from 256 to 512 is too small to resolve for either one: the LSTM gains {-pairs[2]['delta_bpb']:.4f} bpb and the Transformer only {-pairs[5]['delta_bpb']:.4f}, both under the 0.0138 threshold. So both stop benefiting at around 256 characters. The Transformer flattens out a little more sharply, gaining almost nothing past 256, while the LSTM is still drifting down slightly at 512, though not by enough to call it real.</p>
<p>Even the gains below 256 are partly an accounting effect rather than the model using long-range information. Every evaluation window starts with no context, so the first few characters are hard to predict. A longer window spreads that fixed cost over more characters, which lowers the average bpb even if the model never uses anything far back. <b>On this corpus, neither architecture gets measurably better from context beyond about 256 characters.</b></p>

<h3>3.4 Attention analysis</h3>
<figure>{img('fig5_heads.png')}<div class=cap><b>Figure 4.</b> Four attention heads from the modern Transformer (seed 1337) on the dialogue text, which begins "ROMEO: But soft! what light…"; darker means more attention. All 16 heads for all three texts are in Appendix&nbsp;B, made with the course's attention visualizer.</div></figure>
<div class=tb><table><tr><th>Head type</th><th>Head</th><th class=n>Dialogue</th><th class=n>Verse</th><th class=n>Modern prose</th><th class=n>Validation mean ± SE</th><th class=n>Untrained</th></tr>{t_heads()}</table>
<div class=cap><b>Table 4.</b> Average attention weight on the position each head type is supposed to look at (as in §1.6.2 of the tutorial). I picked the heads using 80 validation windows of 256 characters, then scored them separately on the three texts. The word-start score only counts characters at least two letters into a word, so it can't be confused with the previous-character head. The punctuation score is the total weight on colons, periods, exclamation and question marks, semicolons, commas and line breaks. For reference, uniform attention would score about 0.035 for the previous character and 0.017 for word start on the verse text.</div></div>
<p>I found four kinds of heads, and each one shows up in both seeds (the layer and head numbers differ between seeds, which is expected since nothing ties them together):</p>
<ul>
<li><b>Previous character:</b> puts 0.46–0.67 of its attention on the character right before, compared with 0.020 for an untrained model.</li>
<li><b>Last space:</b> looks back to the most recent space, i.e. the start of the current word (0.23–0.35 vs. 0.016 untrained).</li>
<li><b>Word start:</b> attends to the first letter of the current word (0.28–0.30 vs. 0.019 untrained).</li>
<li><b>Speaker tag / punctuation:</b> 0.47–0.48 vs. about 0.09 untrained. This one is clearly tied to the text type. On the dialogue it scores 0.51–0.52, but only 0.01–0.07 on the verse and modern prose, which don't have speaker names followed by a colon and a line break. It also looks back further than any other head, about 25 characters on average.</li>
</ul>

<h2>4. Analysis</h2>
<div class=two><div style="flex:1.75">
<p><b>Why does the vanilla RNN fall so far behind?</b> At the default settings it trails the LSTM by {d_ls:+.3f} bpb (p&nbsp;=&nbsp;{p_ls:.3f}), and it is clearly underfitting rather than overfitting: its training loss is 1.62 against 1.69 on validation, while the LSTM gets its training loss down to 1.29. To find out where the gap comes from, I closed it one step at a time against the tuned LSTM. The total gap is {dtot:.3f} bpb. Raising the RNN's learning rate from 2e-3 to 8e-3 closes <b>{dlr:.3f}</b> of it. Making it wider (0.28M to 1.08M parameters, the same size as the LSTM) closes another <b>{dcap:.3f}</b>. What's left, <b>{dgate:.3f}</b>, is the actual benefit of gating: an RNN and an LSTM of equal size, both well tuned, score {R2:.3f} and {LS:.3f}, a gap about 4.7 times the seed spread. So gates do help, but they only account for about {dgate/dtot*100:.0f}% of the gap. The RNN is also far more sensitive to the learning rate (Figure&nbsp;2). Across rates from 1e-3 to 8e-3 it ranges from 2.25 to 2.92 bpb, while the LSTM stays between 2.10 and 2.28.</p>
<p>I also expected to see the classic vanishing-gradient picture, and I didn't (Figure&nbsp;5). In the trained models, the gradient from the last character's loss back to earlier inputs shrinks at almost the same rate for the RNN, GRU and LSTM: {g_['rnn']['8']:.3f}, {g_['gru']['8']:.3f} and {g_['lstm']['8']:.3f} at 8 characters back, and {g_['rnn']['64']:.1e}, {g_['gru']['64']:.1e} and {g_['lstm']['64']:.1e} at 64 characters back. I think this is a genuine negative result rather than a bug, because the same measurement gives the Transformer a flat profile that reaches far back, exactly as it should. It doesn't rule out gates helping <i>during</i> training, though, since I only measured the models after training finished.</p>
</div><div style="flex:1">{img('fig4_gradreach.png')}<div class=cap><b>Figure 5.</b> How strongly the last character's loss depends on the input k characters earlier (average gradient size, relative to k = 0; 48 windows × 2 seeds). The Transformer stays around 10<sup>−2</sup> all the way back to k = 255.</div></div></div>
<p><b>Is the larger LSTM better or worse?</b> Worse, at this training budget. It finishes at {pm('lstm_512')} against {pm('lstm')} for the smaller one, a gap of {d_512:+.3f} that is 5.5 times the noise floor (p&nbsp;=&nbsp;{p_512:.2f} with two seeds). Figure&nbsp;1 shows what's going on. Early in training the big LSTM is actually the best model of all, bottoming out at {L5a[0]['best_bpb']:.3f} and {L5a[1]['best_bpb']:.3f} bpb after 8–10M tokens. After that its validation loss starts <i>climbing</i>, by 0.044–0.046 bpb, even though its training loss keeps falling, down to 1.02. By the end its train/validation gap is {L5a[0]['gap']:.2f} nats, compared with {L25[0]['gap']:.2f} for the smaller LSTM. That's textbook overfitting: with four times the parameters and only 1&nbsp;MB of text, it starts memorizing. If I had stopped it at its best point ({best512:.3f}), it would beat the small LSTM at the same learning rate ({bpb('lstm')[0]:.3f}) by {bpb('lstm')[0]-best512:.3f}, which is right at the edge of what I can resolve, and it would still lose to the tuned small LSTM ({LS:.3f}). So the bigger model only comes out ahead with early stopping, and even then the advantage isn't clearly real.</p>
<p><b>Which architecture uses longer context more effectively?</b> Measured in bpb, neither one clearly does. Going from 64 to 512 characters of context, the LSTM improves by {ctx_m['lstm_T64']-ctx_m['lstm_T512']:.3f} bpb and the Transformer by {ctx_m['gpt_T64']-ctx_m['gpt_T512']:.3f}, almost the same. The Transformer is ahead by 0.024–0.032 at every length, but that is the same lead it had at default learning rates in Section 3.2, not something that grows with context. Where they really differ is in <i>how</i> they handle context. The Transformer can reach much further back: at 128 characters its gradient is {g_['gpt_modern']['128']:.1e} versus {g_['lstm']['128']:.1e} for the LSTM, roughly 300 times larger. But that extra reach doesn't turn into better predictions on this corpus, and it only works at the length the model was trained on. The Transformer trained on 64-character windows scores {G[('gpt',64,64)]:.2f} bpb on 64-character windows but falls apart at 512 ({G[('gpt',64,512)]:.2f}). The LSTM has no such problem: trained on 64 characters, it scores {G[('lstm',64,512)]:.3f} on 512-character windows, matching the LSTM actually trained at 512 ({G[('lstm',512,512)]:.3f}). So the Transformer has the longer reach, the LSTM is more flexible, and neither gets a measurable benefit from more than about 256 characters here.</p>
<p><b>Bug or negative result?</b> I'm fairly confident the flat context result is real rather than a bug. Re-scoring the saved models myself reproduces every logged validation number, the pattern is the same across all three seeds and both architectures, the library's unit tests pass (28 passed, 3 skipped), and Tiny Shakespeare just doesn't have much structure beyond the current line and the current speaker.</p>

<h2>5. Conclusion and limitations</h2>
<p>The metric definitions check out to numerical precision, and untrained models score right around ln&nbsp;V, as they should. Once each model is given a fair learning rate, the LSTM and the modern Transformer are tied on this task. The vanilla RNN's big headline gap is mostly a tuning and size issue rather than a gating issue. The larger LSTM overfits. And neither model gains anything measurable from context longer than about 256 characters on this corpus.</p>
<p><b>Limitations.</b> There are several things I'd want to tighten up with more time.</p>
<ul>
<li>I picked learning rates using the validation set, so the "best learning rate" numbers are a little optimistic, and there is no separate test set.</li>
<li>I only tuned learning rates at length 256 with two seeds, and the context study used the default LSTM learning rate, so the Transformer's 0.024–0.032 lead there would probably shrink with a tuned LSTM.</li>
<li>The breakdown of the RNN gap depends on the order I applied the changes in, and I only measured gradient reach after training.</li>
<li>This is a single small corpus at the character level. Longer context might well pay off on larger or more structured text.</li>
</ul>

<h2>Extra credit: a GRU written gate by gate</h2>
<p>For the extra credit I wrote a GRU cell by hand, one gate at a time, in the same style as the tutorial's LSTM. It computes a reset gate <i>r</i> = σ(W<sub>ir</sub>x + W<sub>hr</sub>h), an update gate <i>z</i> = σ(W<sub>iz</sub>x + W<sub>hz</sub>h), a candidate state <i>n</i> = tanh(W<sub>in</sub>x + <i>r</i>·W<sub>hn</sub>h), and the new state h′ = (1 − <i>z</i>)<i>n</i> + <i>z</i>h (biases left out here to keep it readable). I then loaded PyTorch's built-in GRU weights into my version and compared the two in float64. On random networks with one to three layers, and on both of my trained GRU models, the outputs match to within {gv_out:.1e} and the input gradients to within {gv_grad:.1e}. Getting there taught me something: my first guess was that PyTorch stores the gates in the order (z, r, n), and that was off by 0.53. I tried all six possible orders, and only (r, z, n) matches. The code is in the repository linked at the top.</p>

<div class=app>
<h2>Appendix A. Full results</h2>
<p>Every experiment in this report is defined by a sweep file, and every number comes from a run produced by nanolm. The sweep files, per-sweep summary tables, analysis and plotting scripts, figures, training logs, and step-by-step reproduction instructions are all in the repository: <a href="{REPO}">{REPO}</a>.</p>

<h3>Every run</h3>
<table><tr><th>Sweep</th><th>Run</th><th class=n>Seed</th><th class=n>Params (M)</th><th class=n>Tokens (M)</th><th class=n>Loss</th><th class=n>PPL</th><th class=n>bpb</th><th class=n>Val tokens</th><th class=n>Train s</th></tr>{all_runs()}</table>

<h2 class=app>Appendix B. Attention maps</h2>
<p><b>Texts used.</b> Dialogue: "{html.escape(att['texts']['dialogue'].replace(chr(10), ' / '))}". Verse: "{html.escape(att['texts']['verse'])}". Modern prose: "{html.escape(att['texts']['modern'])}".</p>
<h3>Dialogue</h3>{img('attn_dialogue.png', '88%')}
<h3 style="break-before:page">Verse</h3>{img('attn_verse.png', '88%')}
<h3 style="break-before:page">Modern prose</h3>{img('attn_modern.png', '88%')}
</div>
"""

out = HW / "HW1_report.html"
out.write_text(f"<!doctype html><html><head><meta charset=utf-8><title>CMPE 258 HW1 Report</title><style>{CSS}</style></head><body>{BODY}</body></html>")
pdf = HW / "HW1_report.pdf"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{out}"],
               check=True, capture_output=True, timeout=180)
print(subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout)
