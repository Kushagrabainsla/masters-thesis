#!/bin/sh
# E-55 session (README Sections 4 and 9): the version and hardware record, telemetry, the SHA-256 of the code that runs
# (L-25, R-76 C7), the pinned checkpoint download, the main phase (engine 1: the decision arms, then the reported steps
# while the time allows), the noise phase (engine 2: BF16_B) if the time allows, and the frozen decision. The same
# safety as E-54: a session watchdog, a timeout for each step and AUTO_STOP. The Pod start command adds the boot
# watchdog (experiments/runpod/pod_cmd.py --cap 2820). The session starts about 2 min after the boot watchdog, so
# POD_LIMIT 2,580 s fires at about 2,700 s after boot, before the boot watchdog. The results are small (about 60 MB).
# STUB=1 runs the same flow on a Mac with synthetic log-probabilities (no GPU, no vLLM, no download); STUB_ARGS passes
# stub flags to run_e55.py. REH=1 is the S5 rehearsal on a small Gemma 4 model (M31 and R31 name it).
set -u
CODE=${CODE:-/workspace/code}
E55=$CODE/gemma4_kv_model
SCALES=${SCALES:-$CODE/gemma4_kv_scaletie/scales_e54.json}
RES=${RES:-/workspace/results/e55}
STUB=${STUB:-0}
STUB_ARGS=${STUB_ARGS:-}
REH=${REH:-0}
M31=${M31:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
R31=${R31:-d4ab4f579dd3516f97d8a6a4c98d0653480bad15}
DL_TIMEOUT=${DL_TIMEOUT:-420}           # E-54: the setup with this checkpoint took 82 s
POD_LIMIT=${POD_LIMIT:-2580}            # session watchdog at 43 min: about 2,700 s after boot, before 2,820 s
NOISE_NEED=${NOISE_NEED:-420}           # the noise phase starts only with this many seconds left (engine and 1 arm)
AN_RESERVE=${AN_RESERVE:-120}           # seconds kept for the analysis after the last phase
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

# The code that runs (L-25): the hashes of the uploaded package and of the scales file, and the design hashes that
# the main session computed from the design commit (code_manifest.json).
{ sha_list "$E55"; s=$(sha_list "$(dirname "$SCALES")" | awk '$2 == "scales_e54.json"'); echo "$s"; } > "$RES/code_sha.txt"
[ -f "$E55/code_manifest.json" ] && cp "$E55/code_manifest.json" "$RES/code_manifest.json"
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

common() {  # the flags of both phases; the main phase keeps 60 s, so its reported steps come before the noise phase
    echo "--data $E55/data_e55.json --scales $SCALES --out $RES --model-id $M31 --revision $R31 --start $START --deadline $((DEADLINE - AN_RESERVE)) --reserve 60 $RUN_FLAGS"
}
if [ -f "$RES/run_main.json" ] && grep -q '"status": "ok"' "$RES/run_main.json"; then
    log "main: ok before, skip"
else
    [ -e "$RES/run_main.json" ] && mv "$RES/run_main.json" "$RES/run_main.failed.$(date -u +%Y%m%dT%H%M%SZ).json"
    if [ "$STUB" = 1 ]; then
        python3 "$E55/run_e55.py" --phase main $(common) --stub $STUB_ARGS > "$RES/main_log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to $(( $(left) - AN_RESERVE )) python3 "$E55/run_e55.py" --phase main $(common) --snapshot "$SNAP" \
            > "$RES/main_log.txt" 2>&1; rc=$?
    else
        rc=2; echo "no snapshot folder after the download" > "$RES/main_log.txt"
    fi
    log "main: exit $rc after $(elapsed) s; $(tail -1 "$RES/main_log.txt" | cut -c1-300)"
fi
if [ "$STUB" != 1 ]; then  # wait up to 60 s until the GPU has less than 2 GiB in use (as E-53)
    i=0
    while [ $i -lt 12 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 2048 ] && break
        sleep 5; i=$((i + 1))
    done
    log "GPU memory in use before the noise phase: ${used:-?} MiB"
fi
if [ "$(left)" -gt $((NOISE_NEED + AN_RESERVE)) ]; then
    if [ "$STUB" = 1 ]; then
        python3 "$E55/run_e55.py" --phase noise $(common) --stub $STUB_ARGS > "$RES/noise_log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to $(( $(left) - AN_RESERVE )) python3 "$E55/run_e55.py" --phase noise $(common) --snapshot "$SNAP" \
            > "$RES/noise_log.txt" 2>&1; rc=$?
    else
        rc=2
    fi
    log "noise: exit $rc after $(elapsed) s; $(tail -1 "$RES/noise_log.txt" 2>/dev/null | cut -c1-300)"
else
    log "noise: cut ($(left) s left, $((NOISE_NEED + AN_RESERVE)) s needed)"
fi
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E55/analyze_e55.py" --res "$RES" --data "$E55/data_e55.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(tail -1 "$RES/decision.txt" | cut -c1-700)"
log "results: $(du -sk "$RES" | cut -f1) KB in $RES; disk: $(df -h "$RES" 2>/dev/null | tail -1)"
log "done after $(elapsed) s"
