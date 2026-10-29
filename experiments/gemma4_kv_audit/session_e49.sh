#!/bin/sh
# E-49 session (README Sections 4 to 9): version and hardware record, telemetry, copy bandwidth, two model
# downloads, GSM8K prompts, then decode_probe.py for m31 (Gemma 4 31B FP8) and m26 (Gemma 4 26B-A4B BF16),
# then the frozen analysis. Same safety as E-46: session watchdog, a new-work cap, per-run timeout and
# AUTO_STOP. The Pod start command adds the boot watchdog (experiments/runpod/pod_cmd.py).
# STUB=1 runs the same flow on a Mac with synthetic probe outputs (no GPU, no vLLM).
set -u
CODE=${CODE:-/workspace/code}
E49=$CODE/gemma4_kv_audit
RES=${RES:-/workspace/results/e49}
STUB=${STUB:-0}
M31=${M31:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
R31=${R31:-d4ab4f579dd3516f97d8a6a4c98d0653480bad15}
M26=${M26:-google/gemma-4-26B-A4B-it}
R26=${R26:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}
C31=${C31:-2048:4,8192:4,32768:4,65536:4}
C26=${C26:-2048:8,8192:8,32768:8,65536:8}
W=${W:-1024:1,1024:8,1024:32}
P31=${P31:-2048:4,32768:4,1024:1,1024:8,1024:32}
P26=${P26:-2048:8,32768:8,1024:1,1024:8,1024:32}
PROBE_ARGS=${PROBE_ARGS:-}
PROBE_TIMEOUT=${PROBE_TIMEOUT:-1500}
NEW_RUN_CAP=${NEW_RUN_CAP:-3300}        # no new work after 55 min
POD_LIMIT=${POD_LIMIT:-3900}            # session watchdog at 65 min
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-300}
AUTO_STOP=${AUTO_STOP:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
    export FLASHINFER_WORKSPACE_BASE=$VLLM_CACHE_ROOT/flashinfer
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

ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
if [ "$STUB" = 1 ]; then
    { echo "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active"
      i=0; while [ $i -lt 150 ]; do echo "2026/10/30 10:00:00.000, 600, 1980, 60, 99, 0x0"; i=$((i + 1)); done; } > "$ENV/telemetry.csv"
    echo '{"copy_gb_per_s_median": 3000.0, "status": "ok"}' > "$ENV/microbench_start.json"
    mkdir -p "$RES/gsm8k"; echo '[]' > "$RES/gsm8k/gsm8k.json"
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -m pip install --quiet datasets || pip install --quiet datasets || uv pip install --system datasets; } > "$ENV/pip.txt" 2>&1
    python3 -c "
import sys
from huggingface_hub import snapshot_download
for m, r in zip(sys.argv[1::2], sys.argv[2::2]):
    print(m, r, snapshot_download(m, revision=r), flush=True)
" "$M31" "$R31" "$M26" "$R26" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed"
    P="$RES/gsm8k"
    [ -f "$P/gsm8k.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$M31" --out "$P" > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
    python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
    { python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
    python3 -m vllm.collect_env > "$RES/collect_env.txt" 2>&1 || true
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"

probe() {  # $1 key, $2 model, $3 revision, $4 slope cells, $5 profile cells; skip ok runs; log every exit code
    out=$RES/$1
    done_ok "$out" && { log "$1: ok before, skip"; return; }
    capped && { log "time cap: skip $1"; return; }
    rm -rf "$out"; mkdir -p "$out"
    if [ "$STUB" = 1 ]; then
        python3 "$E49/decode_probe.py" --stub --model "$2" --out "$out" --cells "$4" --waste-cells "$W" --profile-cells "$5" $PROBE_ARGS > "$out/probe_stdout.txt" 2>&1; rc=$?
    else
        timeout -k 30 "$PROBE_TIMEOUT" python3 "$E49/decode_probe.py" --model "$2" --revision "$3" --out "$out" \
            --gsm8k "$RES/gsm8k/gsm8k.json" --cells "$4" --waste-cells "$W" --profile-cells "$5" $PROBE_ARGS > "$out/log.txt" 2>&1; rc=$?
    fi
    log "$1: exit $rc after $(elapsed) s"
    sleep 5
}
probe m31 "$M31" "$R31" "$C31" "$P31"
probe m26 "$M26" "$R26" "$C26" "$P26"
[ "$STUB" = 1 ] || python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_end.json" > /dev/null 2>&1
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E49/analyze_e49.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
