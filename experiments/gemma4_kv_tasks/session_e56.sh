#!/bin/sh
# E-56 session (README Sections 4 and 9): the version and hardware record, telemetry, the SHA-256 of the code that runs
# (L-25), the pinned BF16 checkpoint download, the main phase (the warm-up, the pre-flight, BF16, the health gate, DET,
# BF16C, BF16R, then STOCK8 while the time allows) and the frozen decision. The same safety as E-55: a session watchdog,
# a timeout for the main phase and AUTO_STOP (the run script sets AUTO_STOP=1 on the Pod). The Pod start command adds
# the boot watchdog (experiments/runpod/pod_cmd.py --cap BOOT_CAP). The session starts about 3 min after the container,
# so POD_LIMIT 3,900 s fires at about 4,080 s after boot, at the boot watchdog. Exit 3 of run_e56.py is a pre-flight or
# health stop. STUB=1 runs the same flow on a Mac (no GPU, no vLLM, no download); STUB_ARGS passes stub flags. REH=1 is
# the S5 rehearsal on a small Gemma 4 model (M31 and R31 name it).
set -u
CODE=${CODE:-/workspace/code}
E56=$CODE/gemma4_kv_tasks
E55=$CODE/gemma4_kv_model
RES=${RES:-/workspace/results/e56}
STUB=${STUB:-0}
STUB_ARGS=${STUB_ARGS:-}
REH=${REH:-0}
M31=${M31:-google/gemma-4-31B-it}
R31=${R31:-842da3794eaa0b77d5f08bae87a17459d91ff475}
DL_TIMEOUT=${DL_TIMEOUT:-900}           # 62.6 GB: E-53 read 33 GB at about 120 MB/s, E-54 faster
POD_LIMIT=${POD_LIMIT:-3900}            # session watchdog at 65 min
BOOT_CAP=${BOOT_CAP:-4080}              # the boot watchdog that the run script gives pod_cmd.py (information here)
AN_RESERVE=${AN_RESERVE:-120}           # seconds kept for the analysis after the main phase
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-300} # the copy window for RES
AUTO_STOP=${AUTO_STOP:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0  # the engine core and the worker in this process (E-53 README Section 4)
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
mkdir -p "$RES"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ); DEADLINE=$((START + POD_LIMIT))
elapsed() { echo $(( $(date +%s) - START )); }
left() { echo $(( DEADLINE - $(date +%s) )); }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }
run_to() {  # $1 seconds, then the command: with GNU timeout where it exists (the Pod), else without (a Mac stub)
    s=$1; shift
    if command -v timeout > /dev/null 2>&1; then timeout -k 10 "$s" "$@"; else "$@"; fi
}
sha_list() {  # the SHA-256 of every file under $1, with paths relative to $1 (RUNPOD.md Section 6 format)
    if command -v sha256sum > /dev/null 2>&1; then
        find "$1" -type f -exec sha256sum {} + | sed "s|$1/||" | sort -k 2
    else
        find "$1" -type f -exec shasum -a 256 {} + | sed "s|$1/||" | sort -k 2
    fi
}
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
    [ "$AUTO_STOP" = 1 ] && { log "AUTO_STOP in $AUTO_STOP_DELAY s; copy RES now"; sleep "$AUTO_STOP_DELAY"; }
    stop_pod exit
}
trap finish EXIT; trap 'exit 130' INT TERM
[ "$AUTO_STOP" = 1 ] && { ( sleep "$POD_LIMIT"; log "watchdog"; stop_pod watchdog ) & WD=$!; }

# The code that runs (L-25): the E-56 package, and the three E-55 files that it imports.
{ sha_list "$E56"; sha_list "$E55" | awk '$2 == "splice_e55.py" || $2 == "run_e55.py" || $2 == "analyze_e55.py"'; } \
    > "$RES/code_sha.txt"
[ -f "$E56/code_manifest.json" ] && cp "$E56/code_manifest.json" "$RES/code_manifest.json"
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
SNAP=""; RUN_FLAGS=""; [ "$REH" = 1 ] && RUN_FLAGS="--rehearsal"
if [ "$STUB" = 1 ]; then
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
    run_to "$DL_TIMEOUT" python3 -c "
import sys
from huggingface_hub import snapshot_download
for m, r in zip(sys.argv[1::2], sys.argv[2::2]):
    print(m, r, snapshot_download(m, revision=r), flush=True)
" "$M31" "$R31" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed (exit $?)"
    SNAP=$(awk -v m="$M31" '$1 == m {print $3}' "$ENV/models.txt" | tail -1)
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"
log "disk: $(df -h "$RES" 2>/dev/null | tail -1)"

FLAGS="--data $E56/data_e56.json --out $RES --model-id $M31 --revision $R31 --start $START"
FLAGS="$FLAGS --deadline $((DEADLINE - AN_RESERVE)) --reserve 60 $RUN_FLAGS"
if [ -f "$RES/run_main.json" ] && grep -q '"status": "ok"' "$RES/run_main.json"; then
    log "main: ok before, skip"
else
    [ -e "$RES/run_main.json" ] && mv "$RES/run_main.json" "$RES/run_main.failed.$(date -u +%Y%m%dT%H%M%SZ).json"
    if [ "$STUB" = 1 ]; then
        python3 "$E56/run_e56.py" $FLAGS --stub $STUB_ARGS > "$RES/main_log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to $(( $(left) - AN_RESERVE )) python3 "$E56/run_e56.py" $FLAGS --snapshot "$SNAP" \
            > "$RES/main_log.txt" 2>&1; rc=$?
    else
        rc=2; echo "no snapshot folder after the download" > "$RES/main_log.txt"
    fi
    [ "$rc" = 3 ] && log "PREFLIGHT OR HEALTH STOP: a check failed before the compared arms; see run_main.json"
    log "main: exit $rc after $(elapsed) s; $(tail -1 "$RES/main_log.txt" | cut -c1-300)"
fi
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E56/analyze_e56.py" --res "$RES" --data "$E56/data_e56.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(tail -1 "$RES/decision.txt" | cut -c1-900)"
log "results: $(du -sk "$RES" | cut -f1) KB in $RES; disk: $(df -h "$RES" 2>/dev/null | tail -1)"
log "done after $(elapsed) s"
