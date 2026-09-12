#!/bin/bash
# Regenerate the headline figures from the shipped evaluation results.
# No GPU needed. Output lands in work/g_plots/.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

mkdir -p work
cd work
ln -sf "$ROOT"/results/raw_eval/results*.json .
for f in "$ROOT"/data/annotators/* "$ROOT"/data/questionnaires/* "$ROOT"/data/dpo/*.jsonl; do
  ln -sf "$f" .
done
ln -sfn "$ROOT"/data/dpo/user_dpo user_dpo
for d in "$ROOT"/results/downstream/* "$ROOT"/results/eval_dirs/*; do
  ln -sfn "$d" "$(basename "$d")"
done
mkdir -p g_plots
cp -r "$ROOT"/results/plot_inputs/. g_plots/

plot() {
  local s=$1; shift
  cp -f "$ROOT/pipeline/08_plots/$s" .
  python3 "$s" "$@"
}

# 1000-annotator transfer scatters + correlation tables, all five models
plot user_corr_plots_1000.py
SWEEP_SHORT=WizardLM-33B-Uncensored SWEEP_FAMILY=WizardLM plot user_corr_plots_1000.py
plot user_corr_plots_1000_tulu7b.py
plot user_corr_plots_1000_vicuna13b.py
plot user_corr_plots_1000_hermes.py

# synthetic-agent HIGH/LOW panels + macro delta summary
plot wiz_agent_plots.py WizardLM-33B-Uncensored

# downstream behavior vs trait shift, averaged over models
plot downstream_trait_scatter_model_average.py

echo "DONE — figures are in work/g_plots/"
