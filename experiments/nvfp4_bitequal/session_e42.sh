#!/bin/sh
# E-42 session (README Sections 3 to 6, A1): top-20 logprobs for each NVFP4 linear backend, with a positive
# (R_2), a negative (marlin) and a small-change (R_m) control. Same safety as E-41: CUDA bin on PATH, caches on
# the volume, resume, telemetry, AUTO_STOP with a copy window and a watchdog, and a log line for every non-zero exit.
set -u
CODE=${CODE:-/workspace/code}
E42=$CODE/nvfp4_bitequal
RC=$CODE/safedispatch_testA/run_cell.py
RES=${RES:-/workspace/results/e42}
DENSE=${DENSE:-nvidia/Qwen3-8B-NVFP4}
OUT=${OUT:-"--limit 256 --max-num-seqs 256 --max-tokens 64 --logprobs 20"}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
NEW_RUN_CAP=${NEW_RUN_CAP:-2100}        # no new work after 35 min
POD_LIMIT=${POD_LIMIT:-2400}            # A1 (R-51): watchdog at 40 min
AUTO_STOP=${AUTO_STOP:-0}; AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-600}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton TRTLLM_DG_CACHE_DIR=$VLLM_CACHE_ROOT/trtllm_dg
    export FLASHINFER_WORKSPACE_BASE=$VLLM_CACHE_ROOT/flashinfer  # A1: keep FlashInfer JIT builds on the volume
fi
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
if [ -r /proc/1/environ ]; then
    for v in RUNPOD_POD_ID RUNPOD_API_KEY HF_TOKEN; do
        eval "cur=\${$v:-}"; [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1); [ -n "$val" ] && export "$v=$val"
    done
fi
mkdir -p "$RES"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ)
elapsed() { echo $(( $(date +%s) - START )); }
capped() { [ $(( $(elapsed) + ${1:-0} )) -ge "$NEW_RUN_CAP" ]; }
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
run() {  # $1 out dir, $2 script, then its arguments; skip ok runs; log every exit code
    out=$1; script=$2; shift 2
    done_ok "$out" && { log "${out#$RES/}: ok before, skip"; return; }
    capped && { log "time cap: skip ${out#$RES/}"; return; }
    rm -rf "$out"; mkdir -p "$out"
    timeout -k 30 "$RUN_TIMEOUT" python3 "$script" --out "$out" "$@" > "$out/log.txt" 2>&1; rc=$?
    log "${out#$RES/}: exit $rc after $(elapsed) s"
    sleep 10
}
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
[ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active > "$ENV/telemetry.csv" 2>&1 &
TEL=$!
pip install --quiet datasets > "$ENV/pip.txt" 2>&1
python3 -c "
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]: print(m, snapshot_download(m), flush=True)
" "$DENSE" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed"
P="$RES/gsm8k"
[ -f "$P/gsm8k.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$DENSE" --out "$P" ${GSM8K_ARGS:-} > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
log "setup done after $(elapsed) s"

o() { run "$RES/$1" "$RC" --mode outputs --model "$DENSE" --prompts-dir "$P" $OUT $2; }
o R_1 --enforce-eager
o R_2 --enforce-eager
o flashinfer_cudnn "--enforce-eager --linear-backend flashinfer_cudnn"
o flashinfer_cutlass "--enforce-eager --linear-backend flashinfer_cutlass"
o marlin "--enforce-eager --linear-backend marlin"
# A1 (R-51): graph mode next (the E-41 claim is in graph mode), then the small-change control and the rest.
o R_g ""
o flashinfer_cutlass_g "--linear-backend flashinfer_cutlass"
run "$RES/R_m" "$RC" --mode outputs --model "$DENSE" --prompts-dir "$P" --limit 256 --max-num-seqs 64 --max-tokens 64 --logprobs 20 --enforce-eager
o cutlass "--enforce-eager --linear-backend cutlass"
o flashinfer_trtllm "--enforce-eager --linear-backend flashinfer_trtllm"
kill "$TEL" 2>/dev/null
python3 "$E42/analyze_e42.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
