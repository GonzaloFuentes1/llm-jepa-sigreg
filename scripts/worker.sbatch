#!/bin/bash
#SBATCH --job-name=llm-jepa-sigreg-worker
#SBATCH --qos=regular
#SBATCH --comment="LLM-JEPA reproduction and SIGReg extension: worker running grid lines sequentially on 1 GPU"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Runs the lines of a grid file one after another on one GPU. Several workers can share
# a grid: each line is claimed atomically (mkdir), and runs whose metrics already exist
# in results/runs are skipped, so a grid can be resumed by submitting new workers.
#   sbatch scripts/worker.sbatch scripts/grid_main.txt     (submit it twice for 2 GPUs)
# Grid lines: TAG SEED [train args...]; blank lines and # comments are ignored.
set -uo pipefail

REPO=${REPO:-$HOME/perfil/llm-jepa-sigreg}
RUNS=${RUNS:-$HOME/perfil/runs}
GRID=$1
CLAIMS=$RUNS/claims/$(basename "$GRID" .txt)
mkdir -p "$CLAIMS"

i=0
while read -r tag seed rest; do
  i=$((i + 1))
  if [[ -f "$REPO/results/runs/$tag/s$seed.json" ]] || ! mkdir "$CLAIMS/$i-$tag-s$seed" 2>/dev/null; then
    continue
  fi
  echo "[$(date -Is)] start line $i: $tag seed $seed $rest"
  # shellcheck disable=SC2086
  if bash "$REPO/scripts/run.sbatch" "$tag" "$seed" $rest < /dev/null; then
    echo "[$(date -Is)] done $tag seed $seed"
  else
    echo "[$(date -Is)] FAILED $tag seed $seed"
    rmdir "$CLAIMS/$i-$tag-s$seed"
  fi
done < <(grep -v -e '^\s*#' -e '^\s*$' "$REPO/$GRID")
