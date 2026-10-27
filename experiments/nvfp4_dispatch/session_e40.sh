#!/bin/sh
# E-40 session (README Sections 3 to 6): screen, accuracy, paired speed; MoE cell first. Same safety as
# E-38 and E-39: CUDA bin on PATH, caches under the vLLM cache root, resume, telemetry, AUTO_STOP with a
# copy window and a watchdog, and a log line for every non-zero exit. Code layout: $CODE/<folder>.
set -u
CODE=${CODE:-/workspace/code}
E40=$CODE/nvfp4_dispatch
RC=$CODE/safedispatch_testA/run_cell.py
A2=$CODE/safedispatch_testA2/run_a2.py
RES=${RES:-/workspace/results/e40}
DENSE=${DENSE:-nvidia/Qwen3-8B-NVFP4}; MOE=${MOE:-nvidia/Qwen3-30B-A3B-NVFP4}
DENSE_CANDS=${DENSE_CANDS:-"cutlass flashinfer_cutlass flashinfer_trtllm flashinfer_cudnn flashinfer_cutedsl b12x marlin humming"}
MOE_CANDS=${MOE_CANDS:-"flashinfer_trtllm flashinfer_cutlass flashinfer_cutedsl cutlass b12x marlin humming"}
TP=${TP:-"--batch-sizes 1 4 8 32 128 256 --max-num-seqs 256 --repeats 3 --throughput-max-tokens 128"}  # A1 (R-47): budget
ACC=${ACC:-""}                          # dry run: smaller engine settings for run_a2.py
ROUNDS=${ROUNDS:-6}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
NEW_RUN_CAP=${NEW_RUN_CAP:-9900}        # no new work after 2 h 45 min
POD_LIMIT=${POD_LIMIT:-10800}           # watchdog at 3 h
ROUND_RESERVE=${ROUND_RESERVE:-480}
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

phase_screen() {  # $1 cell dir, $2 model, $3 flag, $4 candidates
    d=$1/screen; m=$2; f=$3
    run "$d/warm_R" "$RC" --mode warmup --model "$m" --prompts-dir "$P" $TP
    run "$d/R_1" "$RC" --mode throughput --model "$m" --prompts-dir "$P" $TP
    grep -qE "MoE backend|Selected [^ ]+ for" "$d/R_1/log.txt" 2>/dev/null || log "WARNING: no backend selection line in $d/R_1"
    n=0; half=$(( $(echo $4 | wc -w) / 2 ))
    for v in $4; do  # A1 (R-47): R_2 runs in the middle, so a cut screen still has two R runs
        n=$((n + 1))
        run "$d/cand_$v" "$RC" --mode throughput --model "$m" --prompts-dir "$P" $TP "$f" "$v"
        [ "$n" = "$half" ] && run "$d/R_2" "$RC" --mode throughput --model "$m" --prompts-dir "$P" $TP
    done
    done_ok "$d/R_2" || run "$d/R_2" "$RC" --mode throughput --model "$m" --prompts-dir "$P" $TP
    sel=$(python3 "$E40/analyze_e40.py" screen --cell "$1" 2>>"$RES/session.log"); rc=$?
    log "$(basename "$1") screen selected: ${sel:-none} (analysis exit $rc)"
}
phase_accuracy() {  # $1 cell dir, $2 model, $3 flag
    sel=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['selected']))" "$1/screen.json" 2>/dev/null)
    [ -n "$sel" ] || return 0
    run "$1/acc/warm_R" "$A2" --mode warmup --no-score --model "$2" --prompts-dir "$P" $ACC
    run "$1/acc/R_a" "$A2" --mode full --no-score --model "$2" --prompts-dir "$P" $ACC
    for v in $sel; do
        run "$1/acc/warm_$v" "$A2" --mode warmup --no-score --model "$2" --prompts-dir "$P" $ACC "$3" "$v"
        run "$1/acc/cand_$v" "$A2" --mode full --no-score --model "$2" --prompts-dir "$P" $ACC "$3" "$v"
    done
    ok=$(python3 "$E40/analyze_e40.py" accept --cell "$1" 2>>"$RES/session.log"); rc=$?
    log "$(basename "$1") accuracy pass: ${ok:-none} (analysis exit $rc)"
}
phase_paired() {  # $1 cell dir, $2 model, $3 flag
    ok=$(python3 -c "import json,sys; print(' '.join(k for k,v in json.load(open(sys.argv[1])).items() if v.get('A1')))" "$1/accept.json" 2>/dev/null)
    for v in $ok; do
        d=$1/paired/$v
        run "$d/warm_R" "$RC" --mode warmup --model "$2" --prompts-dir "$P" $TP
        run "$d/warm_C" "$RC" --mode warmup --model "$2" --prompts-dir "$P" $TP "$3" "$v"
        i=0
        while [ "$i" -lt "$ROUNDS" ]; do
            i=$((i + 1))
            capped "$ROUND_RESERVE" && { log "time cap: stop $v before round $i"; break; }
            if [ $((i % 2)) = 1 ]; then
                run "$d/R_$i" "$RC" --mode throughput --model "$2" --prompts-dir "$P" $TP
                run "$d/C_$i" "$RC" --mode throughput --model "$2" --prompts-dir "$P" $TP "$3" "$v"
            else
                run "$d/C_$i" "$RC" --mode throughput --model "$2" --prompts-dir "$P" $TP "$3" "$v"
                run "$d/R_$i" "$RC" --mode throughput --model "$2" --prompts-dir "$P" $TP
            fi
        done
    done
}
M=$RES/cell_M; D=$RES/cell_D
# A1 (R-47): both screens first, then the MoE cell, then the dense cell.
phase_screen "$M" "$MOE" --moe-backend "$MOE_CANDS"
phase_screen "$D" "$DENSE" --linear-backend "$DENSE_CANDS"
phase_accuracy "$M" "$MOE" --moe-backend
phase_paired "$M" "$MOE" --moe-backend
phase_accuracy "$D" "$DENSE" --linear-backend
phase_paired "$D" "$DENSE" --linear-backend
kill "$TEL" 2>/dev/null
python3 "$E40/analyze_e40.py" decide --cells "$D" "$M" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
