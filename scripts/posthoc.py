"""HW1 post-hoc analysis of finished nanolm runs (library code is imported, never edited).

Run from DeepDataMiningLearning/llm:  python <this file> <results_dir>
Writes JSON files to results_dir:
  metrics_by_hand.json    independent recomputation of loss/ppl/bits-per-token/bpb from raw val text
  untrained.json          loss after train.max_steps=1 vs ln V
  context_fixed512.json   every context-study checkpoint re-scored on the same 512-char windows
  grad_reach.json         ||d loss_T / d e_(T-k)|| vs k  for trained recurrent models and the transformer
  attention.json          head statistics on three text kinds + validation windows (trained vs untrained)
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from nanolm.generate import load_run
from nanolm.viz import attention_maps
from nanolm.evaluate import evaluate
from nanolm.data import TokenData, byte_lut_tensors, prepare_dataset
from nanolm.device import autocast_ctx

ROOT = Path("outputs/nanolm/sweeps")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1)); print("wrote", name)

def val_data(cfg):
    meta = prepare_dataset(cfg.data)
    return TokenData(meta, sampler=cfg.data.sampler, seed=cfg.seed), meta

# ------------------------------------------------------------- 1. metrics by hand
def metrics_by_hand():
    out = {}
    for run in ["gpt_modern_s1337", "lstm_s1337"]:
        d = ROOT / "char_models" / run
        m = json.loads((d / "metrics.json").read_text())
        # (a) arithmetic from metrics.json only
        L, N, B = m["val_loss"], m["val_tokens"], m["val_bytes"]
        total_nats = L * N
        hand = {"total_nats": total_nats, "loss": total_nats / N, "ppl": math.exp(total_nats / N),
                "bits_per_token": total_nats / N / math.log(2), "tokens_per_byte": N / B,
                "bpb": total_nats / math.log(2) / B}
        # (b) independent forward pass over the validation text (no nanolm.evaluate)
        cfg, model, tok, dinfo, meta = load_run(str(d), device="auto")
        data, _ = val_data(cfg)
        T = cfg.train.seq_len
        ids = torch.as_tensor(np.asarray(data.val).astype("int64"))
        n_win = (ids.numel() - 1) // T
        nats, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, n_win, 64):
                idx = range(s, min(s + 64, n_win))
                win = torch.stack([ids[i * T: i * T + T + 1] for i in idx]).to(dinfo.device)
                with autocast_ctx(dinfo):
                    logits = model(win[:, :-1])
                nats += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), win[:, 1:].reshape(-1),
                                        reduction="sum").item()
                n += win[:, 1:].numel()
        text_bytes = len(tok.decode(ids[1: n + 1].tolist()).encode("utf-8"))
        indep = {"total_nats": nats, "tokens": n, "bytes_utf8": text_bytes, "loss": nats / n,
                 "ppl": math.exp(nats / n), "bits_per_token": nats / n / math.log(2),
                 "bpb": nats / math.log(2) / text_bytes}
        out[run] = {"logged": {k: m[k] for k in ("val_loss", "val_ppl", "val_bpb", "val_tokens", "val_bytes")},
                    "from_metrics_json": hand, "independent_forward_pass": indep,
                    "ln_ppl_equals_loss": abs(math.log(m["val_ppl"]) - L)}
    dump("metrics_by_hand.json", out)

def untrained():
    out = {}
    for tok_name in ["char", "byte", "gpt2"]:
        rows = []
        for seed in (1337, 42):
            d = ROOT / "hw1_untrained" / f"{tok_name}_s{seed}"
            m = json.loads((d / "metrics.json").read_text())
            ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
            V = ck["data_meta"]["vocab_size"]
            rows.append({"seed": seed, "V": V, "lnV": math.log(V), "val_loss": m["val_loss"],
                         "excess": m["val_loss"] - math.log(V), "val_ppl": m["val_ppl"],
                         "val_bpb": m["val_bpb"], "bytes_per_token": m["val_bytes_per_token"],
                         "uniform_bpb": math.log2(V) * m["val_tokens"] / m["val_bytes"], "val_acc": m["val_acc"],
                         "chance_acc": 1 / V, "steps": m["steps"], "tokens": m["val_tokens"], "params": m["params"]})
        out[tok_name] = rows
    dump("untrained.json", out)

# ---------------------------------------------------- 2. fixed-window context evaluation
def context_fixed():
    out = []
    for d in sorted((ROOT / "hw1_context").glob("*_s*")):
        m = json.loads((d / "metrics.json").read_text())
        cfg, model, tok, dinfo, meta = load_run(str(d), device="auto")
        data, dm = val_data(cfg)
        luts = byte_lut_tensors(dm, dinfo.device)
        rows = {"run": d.name, "name": m["name"], "seed": m["seed"], "train_T": cfg.train.seq_len,
                "val_bpb_at_train_window": m["val_bpb"], "val_tokens": m["val_tokens"],
                "train_time_sec": m["train_time_sec"], "params": m["params"]}
        for W in (512,):
            r = evaluate(model, data.val_windows(W, 0), W, 16, luts, dinfo)
            rows[f"bpb_window{W}"], rows[f"tokens_window{W}"] = r["bpb"], r["tokens"]
        # last-half-only scoring at the training window: every scored token has >= T/2 context
        r = evaluate(model, data.val_windows(cfg.train.seq_len, 0), cfg.train.seq_len, 32, luts, dinfo,
                     stride=max(1, cfg.train.seq_len // 2))
        rows["sliding_halfT_bpb"], rows["sliding_halfT_tokens"] = r["bpb"], r["tokens"]
        out.append(rows); print(rows)
    dump("context_fixed512.json", out)

# ---------------------------------------------------- 3. gradient reach
def grad_reach():
    """Mean over val windows of ||d loss(y_T) / d e_{T-k}||_2 (k=0..255), per model; eval mode, CPU."""
    res = {}
    for name in ["rnn", "gru", "lstm", "gpt_modern"]:
        per_seed = []
        for seed in (1337, 42):
            d = ROOT / "char_models" / f"{name}_s{seed}"
            cfg, model, tok, dinfo, meta = load_run(str(d), device="cpu")
            data, _ = val_data(cfg)
            T = 256
            ids = torch.as_tensor(np.asarray(data.val).astype("int64"))
            n = min(48, (ids.numel() - 1) // T)
            win = torch.stack([ids[i * T: i * T + T + 1] for i in range(n)])
            x, y = win[:, :-1], win[:, 1:]
            model.float()
            emb_mod = model.embed if hasattr(model, "embed") else model.tok_emb
            cap = {}
            def hook(_, __, out):
                out.retain_grad(); cap["e"] = out
            h = emb_mod.register_forward_hook(hook)
            with torch.backends.cudnn.flags(enabled=False):
                logits = model(x)
            h.remove()
            loss = F.cross_entropy(logits[:, -1], y[:, -1], reduction="sum")
            model.zero_grad(); loss.backward()
            g = cap["e"].grad.norm(dim=-1).mean(0).flip(0).numpy()      # index k = distance back
            g_norm = g / g[0]
            per_seed.append(g_norm.tolist())
        arr = np.array(per_seed)
        res[name] = {"mean": arr.mean(0).tolist(), "per_seed": per_seed,
                     "k_below_1pct": int(np.argmax(arr.mean(0)[1:] < 0.01)) + 1 if (arr.mean(0)[1:] < 0.01).any() else None,
                     "at_k": {str(k): float(arr.mean(0)[k]) for k in (1, 2, 4, 8, 16, 32, 64, 128, 255)}}
    dump("grad_reach.json", res)

# ---------------------------------------------------- 4. attention statistics
TEXTS = {
    "dialogue": "ROMEO:\nBut soft! what light through yonder window breaks?\n\nJULIET:\nO Romeo, Romeo! wherefore art thou Romeo?",
    "verse":    "To be, or not to be, that is the question: whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune",
    "modern":   "The students trained a small language model on their laptop, and then they measured how well it could predict the next letter.",
}
def is_letter(c): return c.isalpha()

def word_start_index(text, t):
    """Index of the first letter of the word containing position t, or None if t is not inside a word."""
    if not is_letter(text[t]): return None
    i = t
    while i > 0 and is_letter(text[i - 1]): i -= 1
    return i

def stats_for_map(A, text):
    """A [T,T] one head. Returns average attention weight on interpretable positions."""
    T = len(text); ks, s = np.arange(T), {}
    s["prev_token"] = float(np.mean([A[t, t - 1] for t in range(1, T)]))
    s["self_token"] = float(np.mean([A[t, t] for t in range(T)]))
    w = [(t, word_start_index(text, t)) for t in range(T)]
    w = [(t, i) for t, i in w if i is not None and t - i >= 2]      # inside word, >=2 chars after the start, so ws != prev/self
    s["word_start"] = float(np.mean([A[t, i] for t, i in w])) if w else float("nan")
    s["n_word_start_positions"] = len(w)
    # weight on the most recent space/newline (word boundary) from any letter position
    b = []
    for t in range(1, T):
        j = t
        while j >= 0 and text[j] not in " \n": j -= 1
        if j >= 0 and j < t - 1: b.append((t, j))
    s["last_space"] = float(np.mean([A[t, j] for t, j in b])) if b else float("nan")
    s["n_last_space_positions"] = len(b)
    # weight on line-ending punctuation/newline (':' '.' '!' '?' ';' ',' newline) anywhere before t
    pm = [i for i, c in enumerate(text) if c in ":.!?;\n,"]
    p = [A[t, [i for i in pm if i <= t]].sum() for t in range(T) if any(i <= t for i in pm)]
    s["punct_any"] = float(np.mean(p)) if p else float("nan")
    s["mean_lookback"] = float(np.mean([(A[t] * (t - ks)).sum() for t in range(T)]))
    return s

def uniform_baseline(text):
    """What a head that attends uniformly over the causal window would score (control number)."""
    T = len(text); A = np.zeros((T, T))
    for t in range(T): A[t, :t + 1] = 1 / (t + 1)
    return stats_for_map(A, text)

def attention_all():
    from nanolm.models import build_model
    runs = {f"gpt_modern_s{s}": str(ROOT / "char_models" / f"gpt_modern_s{s}") for s in (1337, 42)}
    runs["untrained_s1337"] = str(ROOT / "hw1_untrained" / "char_s1337")
    runs["untrained_s42"] = str(ROOT / "hw1_untrained" / "char_s42")
    out = {"texts": TEXTS, "uniform_baseline": {k: uniform_baseline(v) for k, v in TEXTS.items()}, "runs": {}}
    val_text = None
    for rname, rd in runs.items():
        cfg, model, tok, dinfo, _ = load_run(rd, device="cpu")
        model.float()
        data, _ = val_data(cfg)
        ids = np.asarray(data.val).astype("int64")
        if val_text is None:
            val_text = tok.decode(ids[:21000].tolist())
        rr = {"text": {}, "val": {}}
        for kind, text in TEXTS.items():
            maps = attention_maps(model, torch.tensor([tok.encode(text)]))
            rr["text"][kind] = [[stats_for_map(maps[l][h], text) for h in range(maps[l].shape[0])] for l in range(len(maps))]
        # validation windows: 80 consecutive windows of 256 chars
        L = len(maps); H = maps[0].shape[0]
        acc = [[[] for _ in range(H)] for _ in range(L)]
        for w in range(80):
            text = val_text[w * 256: (w + 1) * 256]
            m = attention_maps(model, torch.tensor([tok.encode(text)]))
            for l in range(L):
                for h in range(H):
                    acc[l][h].append(stats_for_map(m[l][h], text))
        rr["val"] = [[{k: float(np.nanmean([d[k] for d in acc[l][h]])) for k in acc[l][h][0]} |
                      {f"{k}_se": float(np.nanstd([d[k] for d in acc[l][h]], ddof=1) / np.sqrt(80))
                       for k in ("prev_token", "word_start", "last_space", "punct_any")}
                      for h in range(H)] for l in range(L)]
        out["runs"][rname] = rr
        print("attention done", rname)
    dump("attention.json", out)

if __name__ == "__main__":
    metrics_by_hand(); untrained(); grad_reach(); attention_all(); context_fixed()
