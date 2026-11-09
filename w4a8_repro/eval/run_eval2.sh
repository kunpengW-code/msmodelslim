#!/bin/bash
# AIME/GPQA runner (ais_bench protocol replica via eval2_client.py) - 131 edition.
# Prompts/sampling/scoring replicate the 10007/20002 ais_bench runs (m3 protocol);
# only the client differs (self-contained aiohttp instead of the missing ais_bench venv).
# usage: bash run_eval2.sh <model_path> <tag> <cards> <bench:aime|gpqa>
# env: EVAL2_PORT (default 18201), EVAL2_EAGER (0|1), WAIT_TIMEOUT (3600), EVAL_TIMEOUT (28800)
# Safety: only $cards used; teardown kills ONLY this run's tree (pgid + model/port markers).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:?usage: run_eval2.sh <model> <tag> <cards> <bench>}
TAG=${2:?tag}
CARDS=${3:?cards}
BENCH=${4:?bench:aime|gpqa}
PORT=${EVAL2_PORT:-18201}
URL=http://127.0.0.1:$PORT
SERVED=qwen36-eval2
TP=2
DP=${EVAL2_DP:-2}

if [ "${EVAL2_EAGER:-0}" = "1" ]; then
  MODE_FLAGS=(--enforce-eager); MODE_DESC=eager
else
  MODE_FLAGS=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,24,32,48,64]}')
  MODE_DESC=graph_FULL_DECODE_ONLY
fi

SRV_PID=""
cleanup() {
  rc=$?
  trap - EXIT
  echo "[cleanup] stopping serving tree..."
  PYTHONPATH= python3 "$HERE/stop_serve_tree.py" "${SRV_PID:-0}" "$MODEL" "port $PORT" >/dev/null 2>&1 || true
  exit $rc
}
trap cleanup EXIT

export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export PYTHONPATH="$HERE/bootstrap:/home/w30022218/w4a8_tuning/scripts${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export VLLM_CACHE_ROOT="$HERE/cache/$TAG"

SERVE_CMD=(python3 "$HERE/serve_with_port.py" serve "$MODEL"
  --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP" --data-parallel-size "$DP" --dtype bfloat16
  --language-model-only --safetensors-load-strategy prefetch
  --no-enable-prefix-caching --enable-chunked-prefill --async-scheduling
  --max-model-len 131072 --max-num-seqs 64 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.9 "${MODE_FLAGS[@]}" --seed 0
  --kv-cache-dtype bfloat16)

mkdir -p "$HERE/logs" "$HERE/results" "$HERE/cache/$TAG"
setsid "${SERVE_CMD[@]}" > "$HERE/logs/$TAG-server.log" 2>&1 &
SRV_PID=$!
sleep 2
[ -e "/proc/$SRV_PID" ] || { echo "ERROR: server exited immediately; see logs/$TAG-server.log"; exit 1; }
echo "[$(date '+%F %T')] run_eval2: model=$MODEL tag=$TAG cards=$CARDS bench=$BENCH mode=$MODE_DESC port=$PORT"
echo "[$(date '+%F %T')] server pid=$SRV_PID pgid=$(ps -o pgid= -p "$SRV_PID" | tr -d ' ')"

echo "[$(date '+%F %T')] waiting for /health (timeout ${WAIT_TIMEOUT:-3600}s)"
PYTHONPATH= python3 - "$SRV_PID" "${WAIT_TIMEOUT:-3600}" "$URL" <<'EOF'
import os, sys, time, urllib.request
pid, timeout, url = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + timeout
while True:
    try:
        if urllib.request.urlopen(url + '/health', timeout=2).status == 200:
            print('server healthy'); break
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

PYTHONPATH= python3 - "$URL" "$SERVED" <<'EOF'
import json, sys, urllib.request
url, served = sys.argv[1], sys.argv[2]
data = json.load(urllib.request.urlopen(url + '/v1/models', timeout=10))
ids = [m['id'] for m in data['data']]
if served not in ids:
    raise SystemExit(f"ERROR: served model {served!r} not in {ids}")
print('served model ok:', ids)
EOF

echo "[$(date '+%F %T')] running eval2 client ($BENCH, timeout ${EVAL_TIMEOUT:-28800}s)"
( cd "$HERE" && timeout "${EVAL_TIMEOUT:-28800}" python3 "$HERE/eval2_client.py" \
    --bench "$BENCH" --model "$SERVED" --port "$PORT" --tag "$TAG" \
    --out-dir "$HERE/results" ${EVAL2_EXTRA:-} ) 2>&1 | tee "$HERE/logs/$TAG-client.log"

echo "[$(date '+%F %T')] run_eval2 done: results/eval2-$TAG-summary.json"
