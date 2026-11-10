#!/usr/bin/env bash
# Runs every HW1 sweep. Run from repo/DeepDataMiningLearning/llm (the nanolm checkout), with this
# hw1/ folder next to repo/. Resumable: finished runs are skipped.
set -u
HW=${HW:-../../../hw1}
PY=${PY:-python}
for s in hw1_untrained 01_char_models hw1_rnn_controls hw1_context; do
  echo "=== $s $(date)"; $PY -m nanolm.sweep --sweep $HW/configs/sweeps/$s.yaml
done
# control sweeps added after the first results came in
for s in hw1_lr_controls hw1_untrained_lr0 hw1_lr_controls2; do
  echo "=== $s $(date)"; $PY -m nanolm.sweep --sweep $HW/configs/sweeps/$s.yaml
done
echo ALL_DONE $(date)
