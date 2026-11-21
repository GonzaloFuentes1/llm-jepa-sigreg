#!/bin/bash
#SBATCH --job-name=llm-jepa-sigreg
#SBATCH --qos=regular
#SBATCH --comment="LLM-JEPA reproduction and SIGReg extension: fine-tune Llama-3.2-1B on NL-RX-SYNTH, 1 GPU"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#
# One training + evaluation run (also called by scripts/worker.sbatch):
#   sbatch scripts/run.sbatch TAG SEED [train args...]
# Only metrics, the step log and predictions are kept; no checkpoint is ever written.
set -euo pipefail

REPO=${REPO:-$HOME/perfil/llm-jepa-sigreg}
RUNS=${RUNS:-$HOME/perfil/runs}
PYTHON=${PYTHON:-$HOME/perfil/.venv/bin/python}
REVISION=5a8abab4a5d6f164389b1079fb721cfab8d7126c  # unsloth/Llama-3.2-1B-Instruct
export HF_HOME=${HF_HOME:-$RUNS/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

TAG=$1 SEED=$2
shift 2

OUT=$RUNS/$TAG/s$SEED
cd "$REPO"
"$PYTHON" -m llm_jepa_sigreg.train --tag "$TAG" --seed "$SEED" --out-dir "$OUT" \
  --data-dir "$RUNS/data" --revision "$REVISION" "$@"

DEST=$REPO/results/runs/$TAG
mkdir -p "$DEST"
cp "$OUT/metrics.json" "$DEST/s$SEED.json"
cp "$OUT/train_log.jsonl" "$DEST/s$SEED.train_log.jsonl"
cp "$OUT/predictions.jsonl.gz" "$DEST/s$SEED.predictions.jsonl.gz"
rm -rf "$OUT"
