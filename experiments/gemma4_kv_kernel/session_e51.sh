#!/bin/sh
# E-51 session (README Sections 4 and 9): version and hardware record, telemetry, copy bandwidth, the kernel
# microbenchmark (no engine, no model download), copy bandwidth again, then the frozen analysis. Same safety as
# E-50: session watchdog, a timeout and AUTO_STOP. The Pod start command adds the boot watchdog
# (experiments/runpod/pod_cmd.py). STUB=1 runs the same flow on a Mac with synthetic results; STUB_ARGS passes stub
# flags (for example "--stub-fail").
set -u
CODE=${CODE:-/workspace/code}
E51=$CODE/gemma4_kv_kernel
RES=${RES:-/workspace/results/e51}
STUB=${STUB:-0}
STUB_ARGS=${STUB_ARGS:-}
KB_ARGS=${KB_ARGS:-}
KB_TIMEOUT=${KB_TIMEOUT:-1500}          # A1 (R7): with the 11 min pull, the Pod stays inside the $3 cap
POD_LIMIT=${POD_LIMIT:-1980}            # session watchdog at 33 min, before the boot watchdog (2,040 s)
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-180}
AUTO_STOP=${AUTO_STOP:-0}
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
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

ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
if [ "$STUB" = 1 ]; then
    { echo "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active"
      i=0; while [ $i -lt 150 ]; do echo "2026/11/01 10:00:00.000, 600, 1980, 60, 99, 0x0"; i=$((i + 1)); done; } > "$ENV/telemetry.csv"
    echo '{"copy_gb_per_s_median": 3045.0, "status": "ok"}' > "$ENV/microbench_start.json"
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
    { python3 -c "import vllm, torch, triton; print('vllm', vllm.__version__); print('torch', torch.__version__); print('triton', triton.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"
      python3 -m pip list 2>/dev/null | grep -i -E '^(triton|nvidia-cutlass-dsl|quack-kernels|flashinfer)' || true; } > "$RES/versions.txt" 2>&1
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"

out=$RES/kbench
if [ -f "$out/run.json" ] && grep -q '"status": "ok"' "$out/run.json"; then
    log "kbench: ok before, skip"
else
    [ -e "$out" ] && mv "$out" "$out.failed.$(date -u +%Y%m%dT%H%M%SZ)"   # keep partial raw data
    mkdir -p "$out"
    if [ "$STUB" = 1 ]; then
        python3 "$E51/kbench_e51.py" --stub --out "$out" $STUB_ARGS $KB_ARGS > "$out/log.txt" 2>&1; rc=$?
    else
        timeout -k 30 "$KB_TIMEOUT" python3 "$E51/kbench_e51.py" --out "$out" $KB_ARGS > "$out/log.txt" 2>&1; rc=$?
    fi
    log "kbench: exit $rc after $(elapsed) s"
fi
[ "$STUB" = 1 ] || python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_end.json" > /dev/null 2>&1
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E51/analyze_e51.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
