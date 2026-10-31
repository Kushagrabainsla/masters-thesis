#!/bin/sh
# E-50 session (README Sections 4 and 9): version and hardware record, telemetry, copy bandwidth, two model
# downloads, GSM8K prompts, then decode_probe.py for three arms in this order: m31f8 (Gemma 4 31B FP8 with an FP8 KV
# cache, the deciding arm), m26 (Gemma 4 26B-A4B BF16, the E-49 cells with the fixed token budget) and m31bf (31B
# with the BF16 KV cache), then the frozen analysis. Same safety as E-49: session watchdog, a new-work cap, a
# per-run timeout and AUTO_STOP. The Pod start command adds the boot watchdog (experiments/runpod/pod_cmd.py).
# STUB=1 runs the same flow on a Mac with synthetic probe outputs (no GPU, no vLLM). STUB_FAIL="m31f8" makes that
# stub probe fail. ONLY selects arms (the rehearsal).
set -u
CODE=${CODE:-/workspace/code}
E49=$CODE/gemma4_kv_audit
E50=$CODE/gemma4_kv_headroom
RES=${RES:-/workspace/results/e50}
STUB=${STUB:-0}
ONLY=${ONLY:-m31f8 m26 m31bf}
M31=${M31:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
R31=${R31:-d4ab4f579dd3516f97d8a6a4c98d0653480bad15}
M26=${M26:-google/gemma-4-26B-A4B-it}
R26=${R26:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}
S31=${S31:-2048:4,8192:4,32768:4,65536:4,65536:4:1,32768:4:1}
A31=${A31:-2048:4,8192:4,32768:4,65536:4}
G32=2048:a32768,8192:a32768,32768:a32768
G64=2048:a65536,8192:a65536,32768:a65536,65536:a65536
PF8=${PF8:-$G32,32768:a32768:1,$G64,65536:a65536:1}
PBF=${PBF:-$G32,$G64}
PP=${PP:-2048:4,32768:4,65536:4,2048:a32768,32768:a32768,2048:a65536,65536:a65536}
C26=${C26:-2048:8,8192:8,32768:8,65536:8,65536:8:1,32768:8:1}
P26=${P26:-2048:8,32768:8,65536:8}
PROBE_ARGS=${PROBE_ARGS:-}
PROBE_TIMEOUT=${PROBE_TIMEOUT:-1500}
T31F8=${T31F8:-2100}                    # A1 (R-67 R2): the FP8 arm is a first use, so its speed is unknown
NEW_RUN_CAP=${NEW_RUN_CAP:-3000}        # no new work after 50 min
POD_LIMIT=${POD_LIMIT:-3600}            # session watchdog at 60 min
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-180}
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
done_ok() { [ -f "$1/run.json" ] && grep -Eq '"status": "(ok|ok_pairs)"' "$1/run.json"; }
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
      i=0; while [ $i -lt 150 ]; do echo "2026/10/31 10:00:00.000, 600, 1980, 60, 99, 0x0"; i=$((i + 1)); done; } > "$ENV/telemetry.csv"
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
    # The GSM8K chat template comes from the pinned m31 snapshot (R-64 O8).
    TOK=$(awk -v m="$M31" '$1 == m {print $3}' "$ENV/models.txt" | tail -1); [ -d "$TOK" ] || TOK=$M31
    P="$RES/gsm8k"
    [ -f "$P/gsm8k.json" ] || python3 "$CODE/safedispatch_testA2/gsm8k.py" --tokenizer "$TOK" --out "$P" > "$ENV/gsm8k.log" 2>&1 || { log "gsm8k failed"; exit 1; }
    python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
    { python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)"
      python3 -c "import vllm.vllm_flash_attn as fa; print('vllm_flash_attn', getattr(fa, '__version__', 'unknown'))"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1   # A1 (R-67 O7)
    python3 -m vllm.collect_env > "$RES/collect_env.txt" 2>&1 || true
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"

gpu_free() {  # wait up to 180 s until the GPU has less than 2 GiB in use (R-64 R8)
    [ "$STUB" = 1 ] && return 0
    i=0; used=""
    while [ $i -lt 36 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 2048 ] && { log "GPU free ($used MiB in use)"; return 0; }
        sleep 5; i=$((i + 1))
    done
    log "WARNING: GPU still has ${used:-?} MiB in use"
}
# $1 arm, $2 model, $3 revision, $4 slope cells, $5 pair cells, $6 profile cells, $7 KV cache dtype, $8 s needed,
# $9 timeout in s
probe() {
    out=$RES/$1
    case " $ONLY " in *" $1 "*) ;; *) log "$1: not selected"; return;; esac
    done_ok "$out" && { log "$1: ok before, skip"; return; }
    capped "$8" && { log "time cap: skip $1"; return; }
    [ -e "$out" ] && mv "$out" "$out.failed.$(date -u +%Y%m%dT%H%M%SZ)"   # keep partial raw data (R-64 R8)
    mkdir -p "$out"
    if [ "$STUB" = 1 ]; then
        fail=""; case " ${STUB_FAIL:-} " in *" $1 "*) fail=--stub-fail;; esac
        python3 "$E49/decode_probe.py" --stub $fail --model "$2" --out "$out" --cells "$4" --pair-cells "$5" \
            --profile-cells "$6" --kv-cache-dtype "$7" --waste-cells "" --max-model-len 67584 $PROBE_ARGS \
            > "$out/probe_stdout.txt" 2>&1; rc=$?
    else
        timeout -k 30 "$9" python3 "$E49/decode_probe.py" --model "$2" --revision "$3" --out "$out" \
            --gsm8k "$RES/gsm8k/gsm8k.json" --cells "$4" --pair-cells "$5" --profile-cells "$6" --kv-cache-dtype "$7" \
            --waste-cells "" --max-model-len 67584 $PROBE_ARGS > "$out/log.txt" 2>&1; rc=$?
    fi
    log "$1: exit $rc after $(elapsed) s"
    gpu_free
}
probe m31f8 "$M31" "$R31" "$S31" "$PF8" "$PP" fp8 0 "$T31F8"          # the deciding arm first
probe m26 "$M26" "$R26" "$C26" "" "$P26" auto 600 "$PROBE_TIMEOUT"    # the second arm needs 10 min before the cap
probe m31bf "$M31" "$R31" "$A31" "$PBF" "$PP" auto 600 "$PROBE_TIMEOUT"
[ "$STUB" = 1 ] || python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_end.json" > /dev/null 2>&1
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E50/analyze_e50.py" --res "$RES" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(elapsed) s"
