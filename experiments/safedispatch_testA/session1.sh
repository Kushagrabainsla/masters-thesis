#!/bin/sh
# E-36 Session 1 on one rented H100 (amendment A1 after R-34, A2 for Runpod after R-39). Run inside the
# official vllm/vllm-openai:v0.32.0 image, with this directory at $CODE and a writable results root.
# Phase A is the frozen design (README Sections 9 and 10): the decision comes only from Phase A.
# Phase B (A2) is exploratory and runs only after Phase A, until EXTRA_STOP_AFTER. The analysis always runs.
#
# A2: a run with an ok run.json is skipped, so the script resumes after a restart with the same RES.
# The elapsed time is kept in $RES/elapsed_s, so the caps count across restarts. On Runpod, put CODE,
# RES and the caches on the volume disk (/workspace), which survives a Pod stop.
set -u

STOP_AFTER=${STOP_AFTER:-11400}            # Phase A: 3 h 10 min (frozen)
EXTRA_STOP_AFTER=${EXTRA_STOP_AFTER:-21600} # Phase B: no new run after 6 h
POD_LIMIT=${POD_LIMIT:-24000}              # AUTO_STOP watchdog: stop the Pod after 6 h 40 min
RUN_TIMEOUT=${RUN_TIMEOUT:-1500}           # 25 min per run
CODE=${CODE:-/code}
RESULTS_ROOT=${RESULTS_ROOT:-/results}
RES=${RES:-$RESULTS_ROOT/session1}
PHASE_B=${PHASE_B:-1}
AUTO_STOP=${AUTO_STOP:-0}                  # 1: stop this Runpod Pod at the end (README Section 11)
TELEMETRY_MS=${TELEMETRY_MS:-1000}
DENSE=${DENSE:-Qwen/Qwen3-8B-FP8}
MOE=${MOE:-Qwen/Qwen3-30B-A3B-FP8}
# Candidates (R-34): no aliases of the reference, no backends that cannot run block FP8 on Hopper.
DENSE_CANDS=${DENSE_CANDS:-"deep_gemm cutlass triton torch marlin"}   # marlin = W8A16 positive control
MOE_CANDS=${MOE_CANDS:-"deep_gemm flashinfer_cutlass hpc marlin triton"}  # triton = expected alias (negative control)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# A2 (R-39 item 2): an SSH shell on Runpod does not get the Pod environment. Read only these names
# from PID 1, and never write their values to a file.
if [ -r /proc/1/environ ]; then
    for v in RUNPOD_POD_ID RUNPOD_API_KEY HF_TOKEN; do
        eval "cur=\${$v:-}"
        [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1)
        [ -n "$val" ] && export "$v=$val"
    done
fi

mkdir -p "$RES"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BEFORE=$(cat "$RES/elapsed_s" 2>/dev/null || echo 0)

elapsed() { echo $(( BEFORE + $(date +%s) - START )); }
save_elapsed() { elapsed > "$RES/elapsed_s"; }
over_time() { [ "$(elapsed)" -ge "$STOP_AFTER" ]; }
over_extra() { [ "$(elapsed)" -ge "$EXTRA_STOP_AFTER" ]; }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }

done_ok() { [ -f "$1/run.json" ] && grep -q '"status": "ok"' "$1/run.json"; }

stop_pod() {  # $1 reason; stop (not terminate), so the volume disk keeps the results
    [ "$AUTO_STOP" = 1 ] || return 0
    if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
        log "AUTO_STOP: RUNPOD_POD_ID or RUNPOD_API_KEY missing; the Pod keeps billing"
        return 0
    fi
    for try in 1 2 3; do
        code=$(python3 -c "
import os, urllib.request
req = urllib.request.Request('https://rest.runpod.io/v1/pods/%s/stop' % os.environ['RUNPOD_POD_ID'], method='POST',
    headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']})
try:
    print(urllib.request.urlopen(req, timeout=30).status)
except Exception as e:
    print(getattr(e, 'code', type(e).__name__))
" 2>&1)
        log "AUTO_STOP ($1): try $try, HTTP $code"
        case "$code" in 2??) return 0;; esac
        sleep 20
    done
}

TELEMETRY=""
WATCHDOG=""
finish() {
    [ -n "$TELEMETRY" ] && kill "$TELEMETRY" 2>/dev/null
    [ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null
    save_elapsed
    stop_pod "exit"
}
trap finish EXIT
trap 'exit 130' INT TERM
if [ "$AUTO_STOP" = 1 ]; then
    left=$(( POD_LIMIT - BEFORE )); [ "$left" -lt 600 ] && left=600
    ( sleep "$left"; log "AUTO_STOP watchdog fired"; stop_pod "watchdog" ) &
    WATCHDOG=$!
fi

run() {  # $1 out dir, then run_cell.py arguments; never fails the script; skips a finished run
    out=$1; shift
    if done_ok "$out"; then log "$(basename "$(dirname "$out")")/$(basename "$out"): ok before, skip"; return; fi
    rm -rf "$out"; mkdir -p "$out"
    timeout -k 30 "$RUN_TIMEOUT" python3 "$CODE/run_cell.py" --out "$out" "$@" > "$out/log.txt" 2>&1
    rc=$?  # read first: a command substitution in the log line would replace $? (A1 bug, fixed in A2)
    log "$(basename "$(dirname "$out")")/$(basename "$out"): exit $rc after $(elapsed) s"
    save_elapsed
    sleep 10  # let the GPU memory of the finished process be released
}

# Step 0: environment manifest, GPU check, telemetry, model prefetch and prompts. One folder per start.
ENV="$RES/env-$STAMP"
mkdir -p "$ENV"
nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=uuid --format=csv,noheader > "$ENV/gpu_uuid.txt" 2>&1
# A2 (R-39 item 9): a restarted Pod can come back on another GPU. Do not mix GPUs in one session.
if [ -f "$RES/gpu_uuid.txt" ] && ! cmp -s "$RES/gpu_uuid.txt" "$ENV/gpu_uuid.txt" && [ "${ALLOW_GPU_CHANGE:-0}" != 1 ]; then
    log "STOP: GPU changed since the first start ($(cat "$RES/gpu_uuid.txt") -> $(cat "$ENV/gpu_uuid.txt")). Set ALLOW_GPU_CHANGE=1 to continue."
    exit 2
fi
[ -f "$RES/gpu_uuid.txt" ] || cp "$ENV/gpu_uuid.txt" "$RES/gpu_uuid.txt"
[ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1

# A2: GPU telemetry for the whole session, in UTC. Old drivers call the last field clocks_throttle_reasons.
start_telemetry() {  # $1 name of the reasons field
    TZ=UTC nvidia-smi --format=csv,nounits -lms "$TELEMETRY_MS" \
        --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,"$1".active \
        > "$ENV/telemetry.csv" 2>&1 &
    TELEMETRY=$!
}
start_telemetry clocks_event_reasons
sleep 3
if ! head -1 "$ENV/telemetry.csv" | grep -q '^timestamp'; then
    kill "$TELEMETRY" 2>/dev/null
    mv "$ENV/telemetry.csv" "$ENV/telemetry_first_try.txt"
    start_telemetry clocks_throttle_reasons
    sleep 3
    head -1 "$ENV/telemetry.csv" | grep -q '^timestamp' || log "WARNING: telemetry did not start; see $ENV/telemetry.csv"
fi

pip install --quiet datasets > "$ENV/pip.txt" 2>&1
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)" > "$ENV/versions.txt" 2>&1
# A2: download both models in the background while the prompts are built; record the revisions.
python3 -c "
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    print(m, snapshot_download(m), flush=True)
" "$DENSE" "$MOE" > "$ENV/models.txt" 2>&1 &
PREFETCH=$!
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
if [ ! -f "$RES/prompts/prompts.json" ]; then
    python3 "$CODE/prompts.py" --source hub --tokenizer "$DENSE" --out "$RES/prompts" > "$ENV/prompts.log" 2>&1 \
        || { log "prompts failed"; exit 1; }
fi
wait "$PREFETCH" || log "WARNING: model prefetch failed; vLLM downloads at engine start"
log "setup done after $(elapsed) s"

PROFILE="--profile-kernels"

cell() {  # $1 cell dir, $2 model, $3 flag, $4 candidates (Phase A, frozen)
    dir=$1; model=$2; flag=$3; cands=$4
    over_time && { log "time cap: skip $dir"; return; }
    run "$dir/R_a" --mode both --model "$model" --prompts-dir "$RES/prompts" $PROFILE
    grep -qE "Selected [^ ]+ for |Fp8 MoE backend out of" "$dir/R_a/log.txt" \
        || log "WARNING: no kernel-selection line in $dir/R_a/log.txt; alias detection cannot work"
    run "$dir/R_b" --mode both --model "$model" --prompts-dir "$RES/prompts"
    for v in $cands; do
        over_time && { log "time cap: stop candidates in $dir"; break; }
        run "$dir/cand_$v" --mode both --model "$model" --prompts-dir "$RES/prompts" $PROFILE "$flag" "$v"
    done
}

cell "$RES/cell_D" "$DENSE" --linear-backend "$DENSE_CANDS"
cell "$RES/cell_M" "$MOE" --moe-backend "$MOE_CANDS"

# Phase A analysis (the decision).
python3 "$CODE/compare.py" --cells "$RES/cell_D" "$RES/cell_M" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"

# Phase B (A2, exploratory, reported only; README Section 11). In order of value.
xcell() {  # $1 cell dir, $2 model, $3 flag, $4 candidates, $5 runs of R (e.g. "R_a R_b"), then run_cell.py arguments
    dir=$1; model=$2; flag=$3; cands=$4; refs=$5; shift 5
    for r in $refs; do
        over_extra && { log "extra cap: skip $dir/$r"; return; }
        run "$dir/$r" --model "$model" --prompts-dir "$RES/prompts" "$@"
    done
    for v in $cands; do
        over_extra && { log "extra cap: stop candidates in $dir"; break; }
        run "$dir/cand_$v" --model "$model" --prompts-dir "$RES/prompts" "$@" "$flag" "$v"
    done
}
most_flipping() {  # $1 cell name: the non-alias, non-control candidate with the most flips in Phase A
    python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))['cells'].get(sys.argv[2], {}).get('candidates', {})
ok = [(v.get('flipped', 0), k) for k, v in c.items() if v.get('status') == 'ok']
print(max(ok)[1] if ok and max(ok)[0] > 0 else '')
" "$RES/summary.json" "$1" 2>/dev/null
}
if [ "$PHASE_B" = 1 ]; then
    X="$RES/extra"
    # B0: the most-flipping candidate again (are its flips repeatable?). Outputs only.
    xcell "$X/b0_cell_D" "$DENSE" --linear-backend "$(most_flipping cell_D)" "" --mode outputs
    xcell "$X/b0_cell_M" "$MOE" --moe-backend "$(most_flipping cell_M)" "" --mode outputs
    # B1: the same flip test at max_num_seqs 8 (batch dependence of flips, README Section 7).
    xcell "$X/b1_cell_D" "$DENSE" --linear-backend "$DENSE_CANDS" "R_a R_b" --mode outputs --max-num-seqs 8
    xcell "$X/b1_cell_M" "$MOE" --moe-backend "$MOE_CANDS" "R_a R_b" --mode outputs --max-num-seqs 8
    python3 "$CODE/compare.py" --cells "$X/b1_cell_D" "$X/b1_cell_M" --out "$X/summary_b1.json" > "$X/b1.txt" 2>&1 || true
    # B3: the reference in eager mode (no CUDA graphs): flips against Phase A R_a, and speed.
    xcell "$X/b3_cell_D" "$DENSE" --linear-backend "" "R_a R_b" --mode both --enforce-eager
    xcell "$X/b3_cell_M" "$MOE" --moe-backend "" "R_a R_b" --mode both --enforce-eager
    # B2: a finer and wider decode sweep (crossovers between the frozen batch sizes, and above 64).
    xcell "$X/b2_cell_D" "$DENSE" --linear-backend "$DENSE_CANDS" "R_a R_b" --mode throughput --max-num-seqs 256 \
        --batch-sizes 1 2 4 8 16 32 64 128 256
    xcell "$X/b2_cell_M" "$MOE" --moe-backend "$MOE_CANDS" "R_a R_b" --mode throughput --max-num-seqs 256 \
        --batch-sizes 1 2 4 8 16 32 64 128 256
fi

[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_end.json" > /dev/null 2>&1
kill "$TELEMETRY" 2>/dev/null
python3 "$CODE/extras.py" --res "$RES" --out "$RES/extras.json" > "$RES/extras.txt" 2>&1 || true
log "done after $(elapsed) s; copy $RES off the machine before terminating it"
