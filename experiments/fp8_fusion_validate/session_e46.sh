#!/bin/sh
# E-46 session (README Sections 3 to 6 and Amendment A1): version record, accuracy and profiles for P1, an early V0
# gate, then P1 (8B deep_gemm), the extras (default-engine profile, vllm bench latency reproducer), P2 (14B deep_gemm)
# and P3 (8B cutlass), fusions on (X) against off (Y). Same safety as E-44. The Pod start command adds the boot
# watchdog (experiments/runpod/pod_cmd.py).
set -u
CODE=${CODE:-/workspace/code}
E46=$CODE/fp8_fusion_validate
RC=$CODE/safedispatch_testA/run_cell.py
A2=$CODE/safedispatch_testA2/run_a2.py
RES=${RES:-/workspace/results/e46}
DENSE=${DENSE:-Qwen/Qwen3-8B-FP8}
MOE=${M14:-Qwen/Qwen3-14B-FP8}          # the setup block downloads "$MOE" and "$DENSE"; here MOE is the 14B dense model
TP=${TP:-"--batch-sizes 1 8 64 256 --max-num-seqs 256 --repeats 5 --throughput-max-tokens 128"}
PROF=${PROF:-"--profile-kernels --profile-batches 8 64"}
ACC=${ACC:-""}
OFF='{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}'
ROUNDS=${ROUNDS:-5}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
WARM_TIMEOUT=${WARM_TIMEOUT:-1200}      # A1 (R-59): cold compiles get more time
BENCH=${BENCH:-1}
NEW_RUN_CAP=${NEW_RUN_CAP:-7200}        # A1: no new work after 120 min
POD_LIMIT=${POD_LIMIT:-7800}            # A1: session watchdog at 130 min
ROUND_RESERVE=${ROUND_RESERVE:-420}
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-300}
AUTO_STOP=${AUTO_STOP:-0}
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
{ python3 -m pip install --quiet datasets || pip install --quiet datasets || uv pip install --system datasets; } > "$ENV/pip.txt" 2>&1  # A1 (R-59)
python3 -c "
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]: print(m, snapshot_download(m), flush=True)
" "$MOE" "$DENSE" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed"
P="$RES/gsm8k"
[ -f "$P/gsm8k.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$DENSE" --out "$P" ${GSM8K_ARGS:-} > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
log "setup done after $(elapsed) s"
{ python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)"
  echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; pip show vllm 2>/dev/null | sed -n '1,3p'; } > "$RES/versions.txt" 2>&1
python3 -m vllm.collect_env > "$RES/collect_env.txt" 2>&1 || true
nvidia-smi > "$RES/nvidia-smi.txt" 2>&1
log "versions: $(tr '\n' ' ' < "$RES/versions.txt")"

rc() {  # $1 out dir, $2 model, $3 backend (auto = default), $4 on|off, then extra run_cell arguments
    o=$1; m=$2; b=$3; s=$4; shift 4
    if [ "$s" = off ]; then
        run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP --linear-backend "$b" --compilation-config "$OFF" "$@"
    else
        run "$o" "$RC" --model "$m" --prompts-dir "$P" $TP --linear-backend "$b" "$@"
    fi
}
warm() {  # A1: warm-up with a longer timeout and one retry; returns 0 if ok
    t=$RUN_TIMEOUT; RUN_TIMEOUT=$WARM_TIMEOUT
    rc "$@" --mode warmup
    done_ok "$1" || { log "retry warm-up ${1#$RES/}"; rc "$@" --mode warmup; }
    RUN_TIMEOUT=$t
    done_ok "$1"
}
acc() {  # $1 out dir, $2 mode, $3 on|off
    t=$RUN_TIMEOUT; [ "$2" = warmup ] && RUN_TIMEOUT=$WARM_TIMEOUT
    if [ "$3" = off ]; then
        run "$1" "$A2" --mode "$2" --no-score --model "$DENSE" --prompts-dir "$P" $ACC --linear-backend deep_gemm --compilation-config "$OFF"
    else
        run "$1" "$A2" --mode "$2" --no-score --model "$DENSE" --prompts-dir "$P" $ACC --linear-backend deep_gemm
    fi
    RUN_TIMEOUT=$t
}
for s in on off; do
    acc "$RES/acc/warm_$s" warmup "$s"
    acc "$RES/acc/$s" full "$s"
done
for s in on off; do
    rc "$RES/prof/$s" "$DENSE" deep_gemm "$s" --mode throughput $PROF
done
# A1 (R-59): early V0 gate; if the fusion does not fire with on, or fires with off, stop here (saves the rest).
python3 "$E46/analyze_e46.py" --res "$RES" --gate > "$RES/gate.txt" 2>&1; g=$?
log "V0 gate: $(cat "$RES/gate.txt") (exit $g)"
[ "$g" = 0 ] || { log "STOP: V0 gate failed"; exit 1; }

pair() {  # $1 pair name, $2 model, $3 backend; X = on, Y = off
    d=$RES/paired/$1
    warm "$d/warm_on" "$2" "$3" on && warm "$d/warm_off" "$2" "$3" off || { log "skip $1: warm-up not ok"; return; }
    i=0
    while [ "$i" -lt "$ROUNDS" ]; do
        i=$((i + 1))
        capped "$ROUND_RESERVE" && { log "time cap: stop $1 before round $i"; break; }
        if [ $((i % 2)) = 1 ]; then
            rc "$d/X_$i" "$2" "$3" on --mode throughput
            rc "$d/Y_$i" "$2" "$3" off --mode throughput
        else
            rc "$d/Y_$i" "$2" "$3" off --mode throughput
            rc "$d/X_$i" "$2" "$3" on --mode throughput
        fi
        [ "$i" = 1 ] && ! done_ok "$d/X_1" && ! done_ok "$d/Y_1" && { log "skip $1: round 1 failed both"; break; }
    done
}
bench() {  # A1 (R-59): a reproducer with vLLM's own CLI and the default multiprocess engine; 3 alternating rounds
    d=$RES/bench; mkdir -p "$d"
    for i in 1 2 3; do
        for s in on off; do
            [ $((i % 2)) = 0 ] && { [ "$s" = on ] && s=off || s=on; }
            f=$d/${s}_$i.json; [ -f "$f" ] && continue
            capped 300 && { log "time cap: skip bench"; return; }
            if [ "$s" = off ]; then
                env -u VLLM_ENABLE_V1_MULTIPROCESSING timeout -k 30 900 vllm bench latency --model "$DENSE" --linear-backend deep_gemm -cc "$OFF" --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10 --output-json "$f" > "$d/${s}_$i.log" 2>&1
            else
                env -u VLLM_ENABLE_V1_MULTIPROCESSING timeout -k 30 900 vllm bench latency --model "$DENSE" --linear-backend deep_gemm --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10 --output-json "$f" > "$d/${s}_$i.log" 2>&1
            fi
            log "bench/${s}_$i: exit $? after $(elapsed) s"
        done
    done
}
pair P1 "$DENSE" deep_gemm
rc "$RES/prof/default" "$DENSE" auto on --mode throughput $PROF      # A1: who is affected by default
[ "$BENCH" = 1 ] && bench
pair P2 "$MOE" deep_gemm
pair P3 "$DENSE" cutlass
cat > "$RES/repro.md" <<REPRO
# E-46 reproducer (written by session_e46.sh)
Image: vllm/vllm-openai@sha256:2e735ffeef26457f8a3ff44a62ae182870b10af37adc06ecbb08baca152365c6
$(cat "$RES/versions.txt")
GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)

Fusions on (default pass config):
  vllm bench latency --model $DENSE --linear-backend deep_gemm --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
Fusions off:
  vllm bench latency --model $DENSE --linear-backend deep_gemm -cc '$OFF' --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
REPRO
kill "$TEL" 2>/dev/null
python3 "$E46/analyze_e46.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
