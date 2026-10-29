#!/bin/sh
# E-45 session (README Sections 3 to 6): accuracy and profiles for A0/A1 on Qwen3-8B-FP8, then paired rounds P1 (M8,
# A0 against A1), P2 (M14, A0 against A1), P3 (M14, B1 against B2). Same safety as E-44: CUDA bin on PATH, caches on
# the volume, resume, telemetry, AUTO_STOP with a copy window and a session watchdog, a log line for every non-zero
# exit. The Pod start command adds the boot watchdog (experiments/runpod/pod_cmd.py).
set -u
CODE=${CODE:-/workspace/code}
E45=$CODE/fp8_quant_cause
RC=$CODE/safedispatch_testA/run_cell.py
A2=$CODE/safedispatch_testA2/run_a2.py
RES=${RES:-/workspace/results/e45}
DENSE=${DENSE:-Qwen/Qwen3-8B-FP8}
MOE=${M14:-Qwen/Qwen3-14B-FP8}          # the setup block downloads "$MOE" and "$DENSE"; here MOE is the 14B dense model
TP=${TP:-"--batch-sizes 1 8 64 256 --max-num-seqs 256 --repeats 5 --throughput-max-tokens 128"}
PROF=${PROF:-"--profile-kernels --profile-batches 8 64 256"}
ACC=${ACC:-""}
NATIVE='{"custom_ops": ["-quant_fp8"]}'
NOFUSE='{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}'
ROUNDS=${ROUNDS:-5}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
NEW_RUN_CAP=${NEW_RUN_CAP:-5700}        # no new work after 95 min
POD_LIMIT=${POD_LIMIT:-6300}            # session watchdog at 105 min
ROUND_RESERVE=${ROUND_RESERVE:-420}
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
" "$MOE" "$DENSE" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed"
P="$RES/gsm8k"
[ -f "$P/gsm8k.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$DENSE" --out "$P" ${GSM8K_ARGS:-} > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
log "setup done after $(elapsed) s"

# cfg name -> engine arguments (A0, A1, B1, B2); every call goes through run() for resume and exit logging.
rc() {  # $1 out dir, $2 model, $3 config, then extra run_cell arguments
    o=$1; m=$2; c=$3; shift 3
    case "$c" in
        A0) run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP "$@" ;;
        A1) run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP --compilation-config "$NATIVE" "$@" ;;
        B1) run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP --linear-backend deep_gemm "$@" ;;
        B2) run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP --linear-backend deep_gemm --compilation-config "$NOFUSE" "$@" ;;
    esac
}
acc() {  # $1 out dir, $2 mode, $3 config
    if [ "$3" = A1 ]; then
        run "$1" "$A2" --mode "$2" --no-score --model "$DENSE" --prompts-dir "$P" $ACC --compilation-config "$NATIVE"
    else
        run "$1" "$A2" --mode "$2" --no-score --model "$DENSE" --prompts-dir "$P" $ACC
    fi
}
for c in A0 A1; do
    acc "$RES/acc/warm_$c" warmup "$c"
    acc "$RES/acc/$c" full "$c"
done
for c in A0 A1; do
    rc "$RES/prof/$c" "$DENSE" "$c" --mode throughput $PROF
done
pair() {  # $1 pair name, $2 model, $3 first config, $4 second config
    d=$RES/paired/$1
    rc "$d/warm_$3" "$2" "$3" --mode warmup
    rc "$d/warm_$4" "$2" "$4" --mode warmup
    i=0
    while [ "$i" -lt "$ROUNDS" ]; do
        i=$((i + 1))
        capped "$ROUND_RESERVE" && { log "time cap: stop $1 before round $i"; break; }
        if [ $((i % 2)) = 1 ]; then
            rc "$d/X_$i" "$2" "$3" --mode throughput
            rc "$d/Y_$i" "$2" "$4" --mode throughput
        else
            rc "$d/Y_$i" "$2" "$4" --mode throughput
            rc "$d/X_$i" "$2" "$3" --mode throughput
        fi
    done
}
pair P1 "$DENSE" A0 A1
pair P2 "$MOE" A0 A1
pair P3 "$MOE" B1 B2
kill "$TEL" 2>/dev/null
python3 "$E45/analyze_e45.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
