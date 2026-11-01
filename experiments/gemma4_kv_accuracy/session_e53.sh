#!/bin/sh
# E-53 session (README Sections 4 and 9; amendment A1 after R-73): version and hardware record, telemetry, the pinned
# checkpoint download (as E-50), the capture (vLLM in-process, one engine start, 4 prompts), the analysis on the GPU,
# the frozen decision, then (A1, R-73 F2) the checksum list of the raw captures, so the main session copies CAPDIR to
# the Mac in the AUTO_STOP window. The same safety as E-52: session watchdog, a timeout for each step and AUTO_STOP.
# The Pod start command adds the boot watchdog (experiments/runpod/pod_cmd.py, A1: CAP 1,740 s). (A1, R-73 B1) The
# session starts about 140 s after the boot watchdog (E-52), so POD_LIMIT 1,440 s fires at about 1,580 s after boot,
# before the boot watchdog. The raw captures (about 3.1 GB) go to CAPDIR, outside RES. STUB=1 runs the same flow on a
# Mac with synthetic captures (no GPU, no vLLM, no download); STUB_ARGS passes stub flags to capture_e53.py (for
# example "--stub-fail"). CAP_ARGS passes the shape flags of the S5 rehearsal (README Section 9).
set -u
CODE=${CODE:-/workspace/code}
E53=$CODE/gemma4_kv_accuracy
RES=${RES:-/workspace/results/e53}
CAPDIR=${CAPDIR:-/workspace/capture_e53}
STUB=${STUB:-0}
STUB_ARGS=${STUB_ARGS:-}
CAP_ARGS=${CAP_ARGS:-}
M31=${M31:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
R31=${R31:-d4ab4f579dd3516f97d8a6a4c98d0653480bad15}
DL_TIMEOUT=${DL_TIMEOUT:-420}           # (A1) E-50: 130 s for this checkpoint (33.3 GB)
CAP_TIMEOUT=${CAP_TIMEOUT:-600}         # (A1) the engine start (E-50: 87 s plus compile) and 4 prefills
AN_TIMEOUT=${AN_TIMEOUT:-240}           # (A1) 3.1 GB read and 40 x 11 attention passes in float32
POD_LIMIT=${POD_LIMIT:-1440}            # (A1) session watchdog at 24 min: about 1,580 s after boot, before 1,740 s
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-300} # (A1, F2) the copy window for RES and CAPDIR (3.1 GB at 30 MB/s: 100 s)
AUTO_STOP=${AUTO_STOP:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0  # the engine core and the worker in this process (README Section 4)
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
mkdir -p "$RES" "$CAPDIR"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ)
elapsed() { echo $(( $(date +%s) - START )); }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }
run_to() {  # $1 seconds, then the command: with GNU timeout where it exists (the Pod), else without (a Mac stub)
    s=$1; shift
    if command -v timeout > /dev/null 2>&1; then timeout -k 10 "$s" "$@"; else "$@"; fi
}
sha_list() {  # (A1, F2) the SHA-256 of every file under $1, with paths relative to $1 (RUNPOD.md Section 6 format)
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
    [ "$AUTO_STOP" = 1 ] && { log "AUTO_STOP in $AUTO_STOP_DELAY s; copy RES and CAPDIR now"; sleep "$AUTO_STOP_DELAY"; }
    stop_pod exit
}
trap finish EXIT; trap 'exit 130' INT TERM
[ "$AUTO_STOP" = 1 ] && { ( sleep "$POD_LIMIT"; log "watchdog"; stop_pod watchdog ) & WD=$!; }

ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
SNAP=""
if [ "$STUB" = 1 ]; then
    { echo "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active"
      i=0; while [ $i -lt 30 ]; do echo "2026/11/02 10:00:00.000, 300, 1980, 50, 60, 0x0"; i=$((i + 1)); done; } > "$ENV/telemetry.csv"
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
    # The pinned download, as E-50 (session_e50.sh): the id and the full revision.
    run_to "$DL_TIMEOUT" python3 -c "
import sys
from huggingface_hub import snapshot_download
for m, r in zip(sys.argv[1::2], sys.argv[2::2]):
    print(m, r, snapshot_download(m, revision=r), flush=True)
" "$M31" "$R31" > "$ENV/models.txt" 2>&1 || log "WARNING: model download failed (exit $?)"
    SNAP=$(awk -v m="$M31" '$1 == m {print $3}' "$ENV/models.txt" | tail -1)
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"

if [ -f "$RES/capture_run.json" ] && grep -q '"status": "ok"' "$RES/capture_run.json"; then
    log "capture: ok before, skip"
else
    [ -e "$RES/capture_run.json" ] && mv "$RES/capture_run.json" "$RES/capture_run.failed.$(date -u +%Y%m%dT%H%M%SZ).json"
    if [ "$STUB" = 1 ]; then
        python3 "$E53/capture_e53.py" --stub --out "$RES" --capture "$CAPDIR" $STUB_ARGS > "$RES/capture_log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to "$CAP_TIMEOUT" python3 "$E53/capture_e53.py" --snapshot "$SNAP" --model-id "$M31" --revision "$R31" \
            --out "$RES" --capture "$CAPDIR" $CAP_ARGS > "$RES/capture_log.txt" 2>&1; rc=$?
    else
        rc=2; echo "no snapshot folder after the download" > "$RES/capture_log.txt"
    fi
    log "capture: exit $rc after $(elapsed) s; $(tail -1 "$RES/capture_log.txt" | cut -c1-300)"
fi
if [ "$STUB" != 1 ]; then  # wait up to 60 s until the GPU has less than 2 GiB in use (as E-50)
    i=0
    while [ $i -lt 12 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 2048 ] && break
        sleep 5; i=$((i + 1))
    done
    log "GPU memory in use before the analysis: ${used:-?} MiB"
fi
DEV=auto; [ "$STUB" = 1 ] && DEV=cpu
run_to "$AN_TIMEOUT" python3 "$E53/analyze_e53.py" --res "$RES" --capture "$CAPDIR" --device "$DEV" \
    --out "$RES/summary.json" > "$RES/analysis_log.txt" 2>&1; rc=$?
log "analysis: exit $rc after $(elapsed) s"
# The frozen rule on what metrics.json holds (it is the same decision if the step above ended well).
python3 "$E53/analyze_e53.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
log "decision: $(tail -1 "$RES/decision.txt")"
# (A1, R-73 F2) keep the raw captures: the checksum list in RES; the main session copies CAPDIR and compares it
sha_list "$CAPDIR" > "$RES/capture_sha.txt" 2>/dev/null
log "captures: $(wc -l < "$RES/capture_sha.txt" | tr -d ' ') files, $(du -sk "$CAPDIR" | cut -f1) KB in $CAPDIR; copy them with RES"
log "done after $(elapsed) s"
