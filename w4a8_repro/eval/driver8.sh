#!/bin/bash
# driver8: w3_g4 (gptq recipe + long-CoT calib, 268k tokens) post-quant eval chain.
# Waits for quant EXIT -> validate -> CEval gate -> AIME (0-3/18206, tag w3_g4)
# + GPQA (symlink gpqa_link4, 4-7/18207, tag w3_g4g) parallel -> sentinels/err summary.
LOGD=/home/w30022218/w4a8_tuning/logs
CE=/home/w30022218/w4a8_tuning/ceval
QOUT=/home/w30022218/w4a8_tuning/quant_out
R=$CE/results
ARM=w3_g4
OUT=$QOUT/$ARM
GLINK=gpqa_link4

errs() { # predictions file -> error record count
  if [ -f "$1" ]; then local c; c=$(grep -c '"error":' "$1" || true); echo "${c:-0}"; else echo 999; fi
}

n=0
while ! grep -q 'EXIT=' $LOGD/quant_${ARM}.log 2>/dev/null; do
  sleep 300; n=$((n+300))
  if [ $n -ge 90000 ]; then
    echo "[$(date '+%F %T')] ERROR: quant $ARM not finished after 25h - aborting"
    exit 1
  fi
done
QRC=$(grep -o 'EXIT=[0-9]*' $LOGD/quant_${ARM}.log | tail -1 | cut -d= -f2)
echo "[$(date '+%F %T')] quant $ARM finished rc=$QRC"
if [ "$QRC" != "0" ] || ! ls $OUT/*.safetensors >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] ERROR: quant $ARM failed or no export - aborting"
  tail -n 30 $LOGD/quant_${ARM}.log
  exit 1
fi
echo "[$(date '+%F %T')] export OK: $(ls $OUT/*.safetensors | wc -l) shards, $(du -sh $OUT | cut -f1)"

docker exec wkp-vllm-030 bash -c "CEVAL_PORT=18099 TP_SIZE=2 bash $CE/run_ceval.sh $OUT $ARM 5,7 1 0" \
  > $LOGD/ceval_${ARM}.log 2>&1
echo "[$(date '+%F %T')] $ARM ceval rc=$?"
grep '^RESULT' $LOGD/ceval_${ARM}.log || true
if ! grep -q '^RESULT' $LOGD/ceval_${ARM}.log; then
  echo "[$(date '+%F %T')] $ARM ceval FAILED - no evals, aborting"
  exit 1
fi

ln -sfn $OUT $QOUT/$GLINK
(
  docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18206 bash run_eval2.sh $OUT $ARM 0,1,2,3 aime" \
    > $LOGD/eval2_${ARM}_aime.log 2>&1
  echo "[$(date '+%F %T')] $ARM AIME rc=$?"
) &
(
  sleep 60
  docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18207 bash run_eval2.sh $QOUT/$GLINK ${ARM}g 4,5,6,7 gpqa" \
    > $LOGD/eval2_${ARM}g_gpqa.log 2>&1
  echo "[$(date '+%F %T')] $ARM GPQA rc=$?"
) &
wait

echo "===== $ARM summaries ====="
grep -E 'aime2024|aime2025|mean' $R/eval2-$ARM-summary.json 2>/dev/null || true
grep -E 'gpqa' $R/eval2-${ARM}g-summary.json 2>/dev/null || true
echo "===== sentinels (w3_g3 runaways: aime2024 id=22, aime2025 id=17) ====="
grep -E 'id=(22|17)\b' $CE/logs/${ARM}-client.log | tail -6 || true
echo "===== err counts ====="
for f in $R/eval2-$ARM-aime2024-predictions.jsonl $R/eval2-$ARM-aime2025-predictions.jsonl $R/eval2-${ARM}g-gpqa-predictions.jsonl; do
  [ -f "$f" ] && echo "$(basename $f): err=$(errs $f)"
done
echo "[$(date '+%F %T')] driver8 complete"
