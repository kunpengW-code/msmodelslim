#!/bin/bash
# C-Eval runner on 20002 (wkp-vllm container).
# Ported from y00884608's ceval-w8a8-w4a8c8-attribution-20261020/pipeline.py
# (2026-10-20) with an identical serving/eval protocol:
#   TP1/DP2, eager, MTP0 (no spec-decode flags), ctx 4096, max-num-seqs 8,
#   max-num-batched-tokens 2048, gpu-mem-util 0.9, seed 0, bfloat16 KV;
#   client: 0-shot answer-only, temperature=0, max_tokens=32,
#   enable_thinking=False, concurrency 8, first-standalone-ABCD scoring.
#
# Usage:
#   TP_SIZE=2 bash run_ceval.sh <model_path> <tag> [cards] [dp] [limit]
#     cards  default "4,5"    -> ASCEND_RT_VISIBLE_DEVICES (free cards only!)
#     dp     default 2        -> --data-parallel-size
#     TP_SIZE env, default 1  -> --tensor-parallel-size; cards must equal TP*DP
#                                (this box has 32GB cards: W8A8/C3 weights do
#                                 not fit at TP1, use TP_SIZE=2 minimum)
#     limit  default 0 = all 1346 questions; N > 0 = first N (smoke only)
#   Extra/overriding vllm flags: EXTRA_SERVE_ARGS="--no-async-scheduling ..."
#   Timeouts (seconds): WAIT_TIMEOUT (default 3600), EVAL_TIMEOUT (default 10800)
#
# Example (full run on the recommended candidate):
#   nohup bash run_ceval.sh /home/w30022218/w4a8_tuning/quant_out/c3_w4a8_mixed \
#          c3_w4a8_mixed 4,5 2 0 > logs/run-c3.log 2>&1 &
#
# Safety: only the cards in $cards are used; teardown kills ONLY this run's
# process tree (pgid + env/cmdline markers via stop_serve_tree.py) — NEVER a
# global VLLM:: sweep, because other services share this container.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:?usage: run_ceval.sh <model_path> <tag> [cards] [dp] [limit]}
TAG=${2:?usage: run_ceval.sh <model_path> <tag> [cards] [dp] [limit]}
CARDS=${3:-4,5}
DP=${4:-2}
LIMIT=${5:-0}
TP=${TP_SIZE:-1}
PORT=${CEVAL_PORT:-18089}
URL=http://127.0.0.1:$PORT
SERVED=qwen36-ceval
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10800}

# ---------------- preflight (no special env yet) ----------------
[ -f "$MODEL/config.json" ] || { echo "ERROR: model config not found: $MODEL"; exit 1; }
[ -f "$HERE/data/validation.jsonl" ] || { echo "ERROR: dataset missing: $HERE/data/validation.jsonl"; exit 1; }
N=$(wc -l < "$HERE/data/validation.jsonl")
[ "$N" = "1346" ] || { echo "ERROR: dataset line count $N != 1346"; exit 1; }
if [ -f "$HERE/results/ceval-$TAG-summary.json" ]; then
  echo "ERROR: results/ceval-$TAG-summary.json already exists; remove it or pick another tag"; exit 1
fi
NCARDS=$(awk -F, '{print NF}' <<< "$CARDS")
if [ "$NCARDS" != "$((TP * DP))" ]; then
  echo "ERROR: cards=$CARDS ($NCARDS gpus) != TP*DP=$((TP * DP)); set TP_SIZE or fix card list"; exit 1
fi
PYTHONPATH= python3 - "$PORT" <<'EOF'
import socket, sys
s = socket.socket()
try:
    if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0:
        raise SystemExit(f"ERROR: port {sys.argv[1]} already in use")
finally:
    s.close()
EOF
mkdir -p "$HERE/logs" "$HERE/results" "$HERE/configs" "$HERE/cache/$TAG"

# ---------------- serve command (mirrors the 10007 pipeline) ----------------
SERVE_PROG="${CEVAL_SERVE:-serve_with_port.py}"
SERVE_CMD=(python3 "$HERE/$SERVE_PROG" serve "$MODEL"
  --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP" --data-parallel-size "$DP" --dtype bfloat16
  --language-model-only --safetensors-load-strategy prefetch
  --no-enable-prefix-caching --enable-chunked-prefill --async-scheduling
  --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.9 --enforce-eager --seed 0
  --kv-cache-dtype bfloat16)
if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  SERVE_CMD+=(${EXTRA_SERVE_ARGS})
fi

echo "[$(date '+%F %T')] run_ceval: model=$MODEL tag=$TAG cards=$CARDS tp=$TP dp=$DP limit=$LIMIT"
echo "[$(date '+%F %T')] serve: ${SERVE_CMD[*]}"

# ---------------- teardown ----------------
SRV_PID=""
cleanup() {
  rc=$?
  trap - EXIT
  echo "[cleanup] stopping serving tree..."
  PYTHONPATH= python3 "$HERE/stop_serve_tree.py" "${SRV_PID:-0}" "$MODEL" \
    "$PORT" >/dev/null 2>&1 || true
  exit $rc
}
trap cleanup EXIT

# ---------------- start server (own session; env propagates registration) ----
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export PYTHONPATH="${CEVAL_BOOTSTRAP:-$HERE/bootstrap}:/home/w30022218/w4a8_tuning/scripts${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export VLLM_CACHE_ROOT="$HERE/cache/$TAG"

setsid "${SERVE_CMD[@]}" > "$HERE/logs/$TAG-server.log" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$HERE/logs/$TAG-server.pid"
sleep 2
if [ -e "/proc/$SRV_PID" ]; then
  echo "[$(date '+%F %T')] server pid=$SRV_PID pgid=$(ps -o pgid= -p "$SRV_PID" | tr -d ' ')"
else
  echo "ERROR: server exited immediately; see logs/$TAG-server.log"; exit 1
fi

# audit manifest (mirrors the 10007 configs/ trail)
PYTHONPATH= python3 - "$MODEL" "$TAG" "$CARDS" "$DP" "$TP" "$SRV_PID" "$HERE" "${SERVE_CMD[@]}" <<'EOF'
import datetime, hashlib, json, sys
model, tag, cards, dp, tp, pid, here, *cmd = sys.argv[1:]
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(
    tag=tag, model=model, command=cmd, cards=cards, dp=int(dp), tp=int(tp), port=int(__import__('os').environ.get('CEVAL_PORT', '18089')),
    mode='eager', mtp=0, server_pid=int(pid),
    dataset_sha256=sha(f'{here}/data/validation.jsonl'),
    client_sha256=sha(f'{here}/ceval_client.py'),
    port_sha256=sha('/home/w30022218/w4a8_tuning/scripts/w4a8_dense_port.py'),
    at=datetime.datetime.now().astimezone().isoformat(),
), open(f'{here}/configs/{tag}.json', 'w'), indent=2)
EOF

# ---------------- wait for /health ----------------
echo "[$(date '+%F %T')] waiting for /health (timeout ${WAIT_TIMEOUT}s; log: logs/$TAG-server.log)"
PYTHONPATH= python3 - "$SRV_PID" "$WAIT_TIMEOUT" "$URL" <<'EOF'
import os, sys, time, urllib.request
pid, timeout, url = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + timeout
while True:
    try:
        if urllib.request.urlopen(url + '/health', timeout=2).status == 200:
            print('server healthy')
            break
    except Exception:
        pass
    if time.monotonic() > deadline:
        raise SystemExit('ERROR: startup timeout')
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit('ERROR: server exited before health (see server log)')
    time.sleep(5)
EOF

# ---------------- run client ----------------
echo "[$(date '+%F %T')] running C-Eval client (timeout ${EVAL_TIMEOUT}s)"
if [ "$LIMIT" = "0" ]; then LIMIT_ARG=(); else LIMIT_ARG=(--limit "$LIMIT"); fi
timeout "$EVAL_TIMEOUT" python3 "$HERE/ceval_client.py" \
  --url "$URL" --model "$SERVED" --tag "$TAG" \
  --data "$HERE/data/validation.jsonl" --out "$HERE/results" "${LIMIT_ARG[@]}" \
  2>&1 | tee "$HERE/logs/$TAG-client.log"

# ---------------- verify summary ----------------
PYTHONPATH= python3 - "$HERE/results/ceval-$TAG-summary.json" "$LIMIT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
expect = 1346 if sys.argv[2] == '0' else int(sys.argv[2])
assert d['total'] == expect, (d['total'], expect)
if expect == 1346:
    assert len(d['subjects']) == 52, len(d['subjects'])
print(f"RESULT tag={d['tag']} accuracy={d['accuracy']*100:.4f}% "
      f"macro={d['macro_accuracy']*100:.4f}% correct={d['correct']}/{d['total']} "
      f"invalid={d['invalid']} seconds={d['seconds']:.0f}")
EOF
echo "[$(date '+%F %T')] run_ceval done: results/ceval-$TAG-summary.json"
