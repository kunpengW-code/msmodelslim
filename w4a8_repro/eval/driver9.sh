#!/bin/bash
# driver9: w3_g4 full stability confirmation round (round 2), all on cards 4-7 sequential.
# CEval gate -> AIME (tag w3g4_r2) -> auto-makeup dead years -> GPQA (tag w3g4g_r2)
# -> auto-makeup dead gpqa -> final summary with sentinels + err counts.
LOGD=/home/w30022218/w4a8_tuning/logs
CE=/home/w30022218/w4a8_tuning/ceval
QOUT=/home/w30022218/w4a8_tuning/quant_out
R=$CE/results
M=$QOUT/w3_g4

errs() {
  if [ -f "$1" ]; then local c; c=$(grep -c '"error":' "$1" || true); echo "${c:-0}"; else echo 999; fi
}

echo "[$(date '+%F %T')] driver9: w3_g4 round-2 stability check (cards 4-7)"

# ---------- 1) CEval ----------
docker exec wkp-vllm-030 bash -c "CEVAL_PORT=18099 TP_SIZE=2 bash $CE/run_ceval.sh $M w3_g4_ce2 4,5 1 0" \
  > $LOGD/ceval_w3_g4_ce2.log 2>&1
echo "[$(date '+%F %T')] ceval rc=$?"
grep '^RESULT' $LOGD/ceval_w3_g4_ce2.log || true
if ! grep -q '^RESULT' $LOGD/ceval_w3_g4_ce2.log; then
  echo "[$(date '+%F %T')] ceval FAILED - aborting"; exit 1
fi

# ---------- 2) AIME ----------
docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18206 bash run_eval2.sh $M w3g4_r2 4,5,6,7 aime" \
  > $LOGD/eval2_w3g4_r2_aime.log 2>&1
echo "[$(date '+%F %T')] AIME rc=$?"

# auto-makeup dead years
for bb in aime2024 aime2025; do
  n=$(errs $R/eval2-w3g4_r2-$bb-predictions.jsonl)
  if [ "$n" -ge 10 ]; then
    echo "[$(date '+%F %T')] INVALID w3g4_r2/$bb (err=$n) -> makeup"
    docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18206 EVAL2_EXTRA='--only $bb' bash run_eval2.sh $M w3g4_r2x_$bb 4,5,6,7 aime" \
      > $LOGD/eval2_w3g4_r2x_$bb.log 2>&1
    echo "[$(date '+%F %T')] makeup $bb rc=$?"
  fi
done

# ---------- 3) GPQA ----------
docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18206 bash run_eval2.sh $M w3g4g_r2 4,5,6,7 gpqa" \
  > $LOGD/eval2_w3g4g_r2_gpqa.log 2>&1
echo "[$(date '+%F %T')] GPQA rc=$?"

n=$(errs $R/eval2-w3g4g_r2-gpqa-predictions.jsonl)
if [ "$n" -ge 10 ]; then
  echo "[$(date '+%F %T')] INVALID w3g4g_r2/gpqa (err=$n) -> makeup"
  docker exec wkp-vllm-030 bash -c "cd $CE && EVAL2_PORT=18206 EVAL2_EXTRA='--only gpqa' bash run_eval2.sh $M w3g4g_r2x 4,5,6,7 gpqa" \
    > $LOGD/eval2_w3g4g_r2x.log 2>&1
  echo "[$(date '+%F %T')] gpqa makeup rc=$?"
fi

# ---------- 4) final summary ----------
echo "===== ROUND-2 SUMMARIES ====="
grep -E 'aime2024|aime2025|mean' $R/eval2-w3g4_r2-summary.json 2>/dev/null || true
for bb in aime2024 aime2025; do
  f=$R/eval2-w3g4_r2x_$bb-summary.json
  [ -f "$f" ] && grep -E "$bb" "$f"
done
grep -E 'gpqa' $R/eval2-w3g4g_r2-summary.json 2>/dev/null || true
[ -f $R/eval2-w3g4g_r2x-summary.json ] && grep -E 'gpqa' $R/eval2-w3g4g_r2x-summary.json
echo "===== sentinels ====="
grep -E 'id=(22|17)\b' $CE/logs/w3g4_r2-client.log 2>/dev/null | tail -6
echo "===== err counts ====="
for bb in aime2024 aime2025; do
  echo "w3g4_r2/$bb: $(errs $R/eval2-w3g4_r2-$bb-predictions.jsonl)"
done
echo "w3g4g_r2/gpqa: $(errs $R/eval2-w3g4g_r2-gpqa-predictions.jsonl)"
echo "[$(date '+%F %T')] driver9 complete"
