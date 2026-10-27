#!/bin/sh
# E-39 session (README Sections 3 to 6): warm-ups, then 6 rounds of R and C (order alternates), then the
# analysis. Same safety as E-38: CUDA bin on PATH, caches under the vLLM cache root, resume, GPU check,
# telemetry, AUTO_STOP with a copy window and a watchdog. Code layout as in E-38 ($CODE/<folder>).
set -u
CODE=${CODE:-/workspace/code}
E39=$CODE/marlin_moe_speed
RC=$CODE/safedispatch_testA/run_cell.py
RES=${RES:-/workspace/results/e39}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-FP8}
ARGS=${ARGS:-"--model $MODEL --batch-sizes 1 2 4 8 16 32 64 128 256 --max-num-seqs 256 --repeats 5 --throughput-max-tokens 128"}
ROUNDS=${ROUNDS:-10}
ROUND_RESERVE=${ROUND_RESERVE:-480}  # A1: start a round only if this many seconds remain before NEW_RUN_CAP
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
NEW_RUN_CAP=${NEW_RUN_CAP:-5700}
POD_LIMIT=${POD_LIMIT:-6300}
AUTO_STOP=${AUTO_STOP:-0}; AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-600}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton TRTLLM_DG_CACHE_DIR=$VLLM_CACHE_ROOT/trtllm_dg
fi
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
if [ -r /proc/1/environ ]; then
    for v in RUNPOD_POD_ID RUNPOD_API_KEY; do
        eval "cur=\${$v:-}"; [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1); [ -n "$val" ] && export "$v=$val"
    done
fi
mkdir -p "$RES"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ)
elapsed() { echo $(( $(date +%s) - START )); }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }
done_ok() { [ -f "$1/run.json" ] && grep -q '"status": "ok"' "$1/run.json"; }
stop_pod() {
    [ "$AUTO_STOP" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || return 0
    python3 -c "
import json, os, urllib.request
q = {'query': 'mutation { podStop(input: {podId: \\\"%s\\\"}) { id } }' % os.environ['RUNPOD_POD_ID']}
r = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps(q).encode(), method='POST',
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']})
print(urllib.request.urlopen(r, timeout=30).status)
" > "$RES/stop.txt" 2>&1; log "AUTO_STOP ($1): $(tail -1 "$RES/stop.txt")"
}
TEL=""; WD=""
finish() {
    [ -n "$TEL" ] && kill "$TEL" 2>/dev/null; [ -n "$WD" ] && kill "$WD" 2>/dev/null
    [ "$AUTO_STOP" = 1 ] && { log "AUTO_STOP in $AUTO_STOP_DELAY s; copy the results now"; sleep "$AUTO_STOP_DELAY"; }
    stop_pod exit
}
trap finish EXIT; trap 'exit 130' INT TERM
[ "$AUTO_STOP" = 1 ] && { ( sleep "$POD_LIMIT"; log "watchdog"; stop_pod watchdog ) & WD=$!; }
run() {  # $1 out dir, then run_cell.py arguments
    out=$1; shift
    done_ok "$out" && { log "$(basename "$out"): ok before, skip"; return; }
    [ "$(elapsed)" -ge "$NEW_RUN_CAP" ] && { log "time cap: skip $(basename "$out")"; return; }
    rm -rf "$out"; mkdir -p "$out"
    timeout -k 30 "$RUN_TIMEOUT" python3 "$RC" --out "$out" "$@" > "$out/log.txt" 2>&1; rc=$?
    log "$(basename "$out"): exit $rc after $(elapsed) s"
    case "$(basename "$out")" in warm_*) ;; *) grep -q "Directly load AOT" "$out/log.txt" || log "WARNING: $(basename "$out") cold";; esac
    sleep 10
}
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
[ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active > "$ENV/telemetry.csv" 2>&1 &
TEL=$!
pip install --quiet datasets > "$ENV/pip.txt" 2>&1
P="$RES/gsm8k"
[ -f "$P/prompts.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$MODEL" --out "$P" > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
log "setup done after $(elapsed) s"
run "$RES/warm_R" --mode warmup --prompts-dir "$P" $ARGS
run "$RES/warm_C" --mode warmup --prompts-dir "$P" $ARGS --moe-backend marlin
i=0
while [ "$i" -lt "$ROUNDS" ]; do
    i=$((i + 1))
    [ $(( $(elapsed) + ROUND_RESERVE )) -ge "$NEW_RUN_CAP" ] && { log "time cap: stop before round $i"; break; }
    if [ $((i % 2)) = 1 ]; then
        run "$RES/R_$i" --mode throughput --prompts-dir "$P" $ARGS
        run "$RES/C_$i" --mode throughput --prompts-dir "$P" $ARGS --moe-backend marlin
    else
        run "$RES/C_$i" --mode throughput --prompts-dir "$P" $ARGS --moe-backend marlin
        run "$RES/R_$i" --mode throughput --prompts-dir "$P" $ARGS
    fi
done
kill "$TEL" 2>/dev/null
python3 "$E39/analyze_e39.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
