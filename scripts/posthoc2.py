"""HW1 follow-up: train-window x eval-window grid, LR-0 untrained, overfitting and paired-difference tables."""
import json, math, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
from nanolm.generate import load_run
from nanolm.evaluate import evaluate
from nanolm.data import TokenData, byte_lut_tensors, prepare_dataset
ROOT = Path("outputs/nanolm/sweeps"); OUT = Path(sys.argv[1])
def m(run): return json.loads((run/"metrics.json").read_text())
# ---- train-window x eval-window grid
rows=[]
for arch in ("lstm","gpt"):
  for T in (64,128,256,512):
    for seed in (1337,42,7):
      d=ROOT/"hw1_context"/f"{arch}_T{T}_s{seed}"
      cfg,model,tok,dinfo,_=load_run(str(d),device="auto")
      dm=prepare_dataset(cfg.data); data=TokenData(dm,sampler=cfg.data.sampler,seed=cfg.seed)
      luts=byte_lut_tensors(dm,dinfo.device)
      for W in (64,128,256,512):
        r=evaluate(model,data.val_windows(W,0),W,16,luts,dinfo)
        rows.append({"arch":arch,"train_T":T,"seed":seed,"eval_W":W,"bpb":r["bpb"],"tokens":r["tokens"]})
g=pd.DataFrame(rows); g.to_csv(OUT/"context_grid.csv",index=False)
piv=g.groupby(["arch","train_T","eval_W"]).bpb.agg(["mean","std"]).reset_index()
print(piv.pivot_table(index=["arch","train_T"],columns="eval_W",values="mean").round(4))
# ---- paired consecutive-T differences (official metric)
c=pd.read_csv(OUT/"hw1_context/summary.csv")
out=[]
from scipy import stats
for arch in ("lstm","gpt"):
  for a,b in [(64,128),(128,256),(256,512)]:
    A=c[c.name==f"{arch}_T{a}"].set_index("seed").val_bpb; B=c[c.name==f"{arch}_T{b}"].set_index("seed").val_bpb
    d=(B-A).reindex(A.index); t=d.mean()/(d.std(ddof=1)/math.sqrt(3))
    out.append({"arch":arch,"from":a,"to":b,"delta_bpb":d.mean(),"sd":d.std(ddof=1),"p":float(2*stats.t.sf(abs(t),2)),"per_seed":d.tolist()})
print(pd.DataFrame(out).drop(columns="per_seed").round(4)); (OUT/"context_paired.json").write_text(json.dumps(out,indent=1))
# ---- learning-curve diagnostics
diag={}
for name in ("lstm","lstm_512","gru","rnn","gpt_textbook","gpt_modern"):
  for seed in (1337,42):
    d=ROOT/"char_models"/f"{name}_s{seed}"
    ev=pd.read_json(d/"eval_log.jsonl",lines=True); tr=pd.read_json(d/"train_log.jsonl",lines=True)
    i=ev.bpb.idxmin()
    diag.setdefault(name,[]).append({"seed":seed,"best_bpb":float(ev.bpb.min()),"best_step":int(ev.step[i]),"best_tokens_M":float(ev.train_tokens[i]/1e6),"final_periodic_bpb":float(ev.bpb.iloc[-1]),
       "final_train_loss":float(tr.loss.iloc[-1]),"final_val_loss":float(m(d)["val_loss"]),"gap":float(m(d)["val_loss"]-tr.loss.iloc[-1]),
       "val_bpb":m(d)["val_bpb"],"curve":list(zip(ev.step.tolist(),[round(float(b),4) for b in ev.bpb]))})
(OUT/"lc_diagnostics.json").write_text(json.dumps(diag,indent=1))
for k,v in diag.items(): print(k,[{a:(round(b,4) if isinstance(b,float) else b) for a,b in x.items() if a!="curve"} for x in v])
