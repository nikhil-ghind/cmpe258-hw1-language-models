# CMPE 258 HW1: Language Models from Scratch

Character-level language models (vanilla RNN, GRU, LSTM and Transformers) trained on Tiny Shakespeare with
nanolm from the course repository, [lkk688/DeepDataMiningLearning](https://github.com/lkk688/DeepDataMiningLearning).
The full write-up is **[HW1_report.pdf](HW1_report.pdf)**.

## What's here

| Folder | Contents |
|---|---|
| `HW1_report.pdf` | The report |
| `configs/sweeps/` | One sweep file per experiment; every number in the report comes from these runs |
| `results/<sweep>/` | `summary.csv` and `summary.md` from `nanolm.analyze`, plus its learning-curve and bar plots |
| `results/*.json`, `results/*.csv` | Post-hoc analysis: metrics by hand, untrained loss, context grid, gradient reach, attention statistics, GRU check |
| `figures/` | Figures used in the report |
| `scripts/` | Analysis, plotting and report-building scripts, and the extra-credit GRU cell (`gru_scratch.py`) |
| `logs/` | Raw logs from the sweep runs |

Checkpoints and datasets are not included.

## Reproducing

Place this folder next to a checkout of the course repository (commit `8418fc1bae`), so the layout is:

```
<project>/
  hw1/    <- this repository
  repo/   <- DeepDataMiningLearning checkout
```

Then, from `repo/DeepDataMiningLearning/llm`:

```bash
HW=../../../hw1
bash $HW/scripts/run_sweeps.sh                                   # all 7 sweeps (~1 GPU-hour on an RTX 5070 Laptop)
python -m nanolm.analyze outputs/nanolm/sweeps/<name> --baseline <baseline> --metric val_bpb --out $HW/results/<name>
PYTHONPATH=. python $HW/scripts/posthoc.py  $HW/results
PYTHONPATH=. python $HW/scripts/posthoc2.py $HW/results
PYTHONPATH=. python $HW/scripts/make_figures.py $HW/results $HW/figures
python $HW/scripts/gru_scratch.py outputs/nanolm/sweeps/char_models/gru_s1337 outputs/nanolm/sweeps/char_models/gru_s42
python $HW/scripts/build_report.py $HW .                          # needs google-chrome for the PDF
```

Attention maps come from `python -m nanolm.viz attention --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337 --text "..."`
on the three texts listed in Appendix B of the report.

Environment: Python 3.12.8, PyTorch 2.11.0+cu128, CUDA with bfloat16 and `torch.compile`.
