#!/usr/bin/env bash
# ObjectCore replication - runs the paper protocol (k = 1,2,4 x 3 random support draws) on MVTec LOCO.
#
# Every launch creates a NEW folder:  ~/objectcore/runs/<YYYY-MM-DD_HHMM>_<mode>/
#   results/      per-run results.json + scores.jsonl + log.txt
#   logs/         console logs, summaries
#   code/         exact copy of the code used (+ sha256)
#   run_record.txt  GPU / driver / library versions, start + end time
#   report/       comparison CSVs vs the paper, graphs, paper-style figures, report.html (built at the end)
#
#   bash run_all.sh smoke     # 4-shot, 1 draw, juice_bottle only, 40 test images  (~5 min, checks everything works)
#   bash run_all.sh A         # Phase A: no detector fine-tuning (all categories)   target ~69 at k=4
#   bash run_all.sh B         # Phase B: detector fine-tuned on the support set     target 80.8 +- 1.3 at k=4
#   bash run_all.sh all       # A then B
#
# Resume an interrupted run INTO ITS EXISTING FOLDER (finished draws are skipped):
#   RESUME=~/objectcore/runs/2026-10-01_2015_all bash run_all.sh all
#
# Run inside tmux so it survives disconnects:  tmux new -s oc  ->  bash run_all.sh all  ->  Ctrl+B D
set -uo pipefail
cd ~/objectcore
source ~/objectcore/env/bin/activate
export HF_HOME=~/objectcore/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=${DATA:-~/objectcore/data/mvtec_loco}
EXTRA=${EXTRA_ARGS:-}   # optional extra flags for objectcore.py, e.g. EXTRA_ARGS="--nms 0.7"
CATS="breakfast_box juice_bottle pushpins screw_bag splicing_connectors"
MODE=${1:-all}
case $MODE in smoke|A|B|all) ;; *) echo "usage: bash run_all.sh [smoke|A|B|all]"; exit 1 ;; esac

# ---- run folder -------------------------------------------------------------------------------
if [ -n "${RESUME:-}" ]; then
  RUN=$(realpath "$RESUME")
  [ -d "$RUN/results" ] || { echo "RESUME folder $RUN has no results/ - check the path"; exit 1; }
  echo "Resuming into existing run folder: $RUN"
  echo "resumed    : $(date) | $(TZ=Asia/Kuala_Lumpur date '+%F %T MYT')" >> "$RUN/run_record.txt"
  if ! diff -q objectcore.py "$RUN/code/objectcore.py" >/dev/null; then
    echo "WARNING: objectcore.py differs from the copy used when this run started ($RUN/code/)."
    echo "         Using the run's own copy to keep the run consistent."
  fi
else
  RUN=~/objectcore/runs/$(date +%Y-%m-%d_%H%M)_$MODE
  [ -e "$RUN" ] && RUN=${RUN}_$(date +%S)
  mkdir -p "$RUN"/{results,logs,code}
  cp objectcore.py summarize.py make_report.py run_all.sh setup.sh viz_steps.py diag_finetune.py README.md "$RUN/code/" 2>/dev/null
  (cd "$RUN/code" && sha256sum *.py *.sh > SHA256SUMS)
  {
    echo "run folder : $RUN"; echo "mode       : $MODE"; echo "started    : $(date) | $(TZ=Asia/Kuala_Lumpur date '+%F %T MYT')"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    python -c "import torch,transformers,numpy,scipy,sklearn;print('torch',torch.__version__,'cuda',torch.version.cuda,'| transformers',transformers.__version__,'| numpy',numpy.__version__,'| scipy',scipy.__version__,'| sklearn',sklearn.__version__)"
  } > "$RUN/run_record.txt"
  echo "New run folder: $RUN"
fi
ln -sfn "$RUN" ~/objectcore/runs/latest   # ~/objectcore/runs/latest always points to the newest run
CODE="$RUN/code"
OUT="$RUN/results"
LOG="$RUN/logs"

FAILED=""
oc () {
  python "$CODE/objectcore.py" --data_root $DATA --out_dir "$OUT" $EXTRA "$@" 2>&1 | grep -v -i "warn"
  local st=${PIPESTATUS[0]}
  [ "$st" -ne 0 ] && FAILED="$FAILED $*" && echo "!!! FAILED (exit $st): $*"
  return 0
}

run_phase () {  # $1 = A|B
  local flag=""; [ "$1" = "B" ] && flag="--finetune"
  for c in $CATS; do
    echo "=== Phase $1 | $c | $(date)"
    oc --category $c --shots 1 2 4 --draws 3 $flag > >(tee -a "$LOG/phase$1_$c.log")
  done
  python "$CODE/summarize.py" "$OUT" | tee "$LOG/summary_after_phase$1.txt"
}

case $MODE in
  smoke) oc --category juice_bottle --shots 4 --draws 1 --max_test 40 > >(tee -a "$LOG/smoke.log")
         oc --category juice_bottle --shots 4 --draws 1 --max_test 40 --finetune > >(tee -a "$LOG/smoke.log") ;;
  A) run_phase A ;;
  B) run_phase B ;;
  all) run_phase A; run_phase B ;;
esac
sleep 1
echo "=== Building report (CSVs, graphs, paper-style figures) $(date)"
python "$CODE/make_report.py" "$RUN" "$DATA" 2>&1 | grep -v -i warn | tee "$LOG/make_report.log" \
  || echo "!!! report failed - rerun: python make_report.py $RUN"
if [ -n "$FAILED" ]; then
  echo "finished WITH ERRORS: $(date) | failed:$FAILED" >> "$RUN/run_record.txt"
  echo "FINISHED WITH ERRORS - failed:$FAILED  (see $LOG). Fix, then resume: RESUME=$RUN bash run_all.sh $MODE"
  exit 1
fi
echo "finished   : $(date) | $(TZ=Asia/Kuala_Lumpur date '+%F %T MYT')" >> "$RUN/run_record.txt"
echo "ALL DONE $(date) - results in $RUN"
