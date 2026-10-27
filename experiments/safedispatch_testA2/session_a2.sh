#!/bin/sh
# E-38 Test A2 session on one H100 (README Sections 4 to 9). Run inside vllm/vllm-openai:v0.32.0 with this
# directory at $CODE/safedispatch_testA2, E-36 at $CODE/safedispatch_testA, the ledger at $CODE/ledger and
# the Runpod tools at $CODE/runpod. Resume: a run with an ok run.json is skipped. The E-36 lessons apply:
# CUDA bin on PATH, caches under the vLLM cache root, one warm-up per config, AUTO_STOP with a copy window.
set -u

NEW_RUN_CAP=${NEW_RUN_CAP:-10800}      # no new run after 3 h (A1 after R-43)
POD_LIMIT=${POD_LIMIT:-12600}          # AUTO_STOP watchdog at 3 h 30 min
SPEED_PLAN=${SPEED_PLAN:-"cell_M:marlin"} # A1: speed only where E-36 saw a possible gain (README 9)
ANCHOR_ARGS_M=${ANCHOR_ARGS_M:-"--max-num-seqs 64"} # the MoE BF16 anchor (about 61 GB) needs a smaller batch
RUN_TIMEOUT=${RUN_TIMEOUT:-1500}
CODE=${CODE:-/workspace/code}
A2=$CODE/safedispatch_testA2
RES=${RES:-${RESULTS_ROOT:-/workspace/results}/a2}
AUTO_STOP=${AUTO_STOP:-0}
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-600}
CELLS=${CELLS:-"D M"}
DENSE=${DENSE:-Qwen/Qwen3-8B-FP8};        DENSE_BF16=${DENSE_BF16:-Qwen/Qwen3-8B}
MOE=${MOE:-Qwen/Qwen3-30B-A3B-FP8};       MOE_BF16=${MOE_BF16:-Qwen/Qwen3-30B-A3B}
DENSE_CANDS=${DENSE_CANDS:-"deep_gemm cutlass triton marlin"}
MOE_CANDS=${MOE_CANDS:-"deep_gemm flashinfer_cutlass marlin triton"}
LIMIT=${LIMIT:-0}                      # dry run: first N problems
EXTRA_ARGS=${EXTRA_ARGS:-}             # dry run: smaller engine settings for run_a2.py
SPEED_ARGS=${SPEED_ARGS:-"--batch-sizes 1 4 8 16 32 64 128 256 --max-num-seqs 256"}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton}
    export TRTLLM_DG_CACHE_DIR=${TRTLLM_DG_CACHE_DIR:-$VLLM_CACHE_ROOT/trtllm_dg}
fi
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
if [ -r /proc/1/environ ]; then  # SSH shells do not get the Pod environment; read three names only
    for v in RUNPOD_POD_ID RUNPOD_API_KEY HF_TOKEN; do
        eval "cur=\${$v:-}"; [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1)
        [ -n "$val" ] && export "$v=$val"
    done
fi

mkdir -p "$RES"
START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BEFORE=$(cat "$RES/elapsed_s" 2>/dev/null || echo 0)
elapsed() { echo $(( BEFORE + $(date +%s) - START )); }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }
done_ok() { [ -f "$1/run.json" ] && grep -q '"status": "ok"' "$1/run.json"; }
capped() { [ "$(elapsed)" -ge "$1" ]; }

stop_pod() {
    [ "$AUTO_STOP" = 1 ] || return 0
    [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { log "AUTO_STOP: Pod variables missing"; return 0; }
    for try in 1 2 3; do
        code=$(python3 -c "
import json, os, urllib.request
q = {'query': 'mutation { podStop(input: {podId: \\\"%s\\\"}) { id desiredStatus } }' % os.environ['RUNPOD_POD_ID']}
req = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps(q).encode(), method='POST',
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']})
try:
    r = urllib.request.urlopen(req, timeout=30); b = json.load(r); print(r.status if not b.get('errors') else 'graphql-error')
except Exception as e:
    print(getattr(e, 'code', type(e).__name__))
" 2>&1)
        log "AUTO_STOP ($1): try $try, HTTP $code"
        case "$code" in 2??) return 0;; esac
        sleep 20
    done
}
TELEMETRY=""; WATCHDOG=""
finish() {
    [ -n "$TELEMETRY" ] && kill "$TELEMETRY" 2>/dev/null
    [ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null
    elapsed > "$RES/elapsed_s"
    if [ "$AUTO_STOP" = 1 ] && [ "$AUTO_STOP_DELAY" -gt 0 ]; then log "AUTO_STOP in $AUTO_STOP_DELAY s; copy the results now"; sleep "$AUTO_STOP_DELAY"; fi
    stop_pod exit
}
trap finish EXIT
trap 'exit 130' INT TERM
if [ "$AUTO_STOP" = 1 ]; then
    left=$(( POD_LIMIT - BEFORE )); [ "$left" -lt 600 ] && left=600
    ( sleep "$left"; log "AUTO_STOP watchdog fired"; stop_pod watchdog ) &
    WATCHDOG=$!
fi

run() {  # $1 out dir, $2 script, then its arguments; skips an ok run; never fails the session
    out=$1; script=$2; shift 2
    if done_ok "$out"; then log "$(basename "$(dirname "$out")")/$(basename "$out"): ok before, skip"; return; fi
    rm -rf "$out"; mkdir -p "$out"
    timeout -k 30 "$RUN_TIMEOUT" python3 "$script" --out "$out" "$@" > "$out/log.txt" 2>&1
    rc=$?
    log "$(basename "$(dirname "$out")")/$(basename "$out"): exit $rc after $(elapsed) s"
    elapsed > "$RES/elapsed_s"
    sleep 10
}

# Setup: environment record, GPU check, telemetry, model prefetch, GSM8K build.
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=uuid --format=csv,noheader > "$ENV/gpu_uuid.txt" 2>&1
if [ -f "$RES/gpu_uuid.txt" ] && ! cmp -s "$RES/gpu_uuid.txt" "$ENV/gpu_uuid.txt" && [ "${ALLOW_GPU_CHANGE:-0}" != 1 ]; then
    log "STOP: GPU changed since the first start"; exit 2
fi
[ -f "$RES/gpu_uuid.txt" ] || cp "$ENV/gpu_uuid.txt" "$RES/gpu_uuid.txt"
[ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 \
    --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
    > "$ENV/telemetry.csv" 2>&1 &
TELEMETRY=$!
pip install --quiet datasets > "$ENV/pip.txt" 2>&1
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)" > "$ENV/versions.txt" 2>&1
env | grep -E "^(VLLM_CACHE_ROOT|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR|TRTLLM_DG_CACHE_DIR|HF_HOME)=" > "$ENV/cache_dirs.txt"
python3 -c "
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    print(m, snapshot_download(m), flush=True)
" $DENSE $DENSE_BF16 $MOE $MOE_BF16 > "$ENV/models.txt" 2>&1 &
PREFETCH=$!
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_start.json" > /dev/null 2>&1
P="$RES/gsm8k"
[ -f "$P/gsm8k.json" ] || python3 "$A2/gsm8k.py" --tokenizer "$DENSE" --out "$P" ${GSM8K_ARGS:-} > "$ENV/gsm8k.log" 2>&1 \
    || { log "gsm8k build failed"; exit 1; }
wait "$PREFETCH" || log "WARNING: model prefetch failed"
log "setup done after $(elapsed) s"

LIM=""; [ "$LIMIT" -gt 0 ] && LIM="--limit $LIMIT"

a2() {  # $1 cell dir, $2 name, $3 model, then backend args; warm-up then the full run
    dir=$1; name=$2; model=$3; shift 3
    capped "$NEW_RUN_CAP" && { log "time cap: skip $dir/$name"; return; }
    ref=""; [ "$name" != R_a ] && ref="--reference-dir $dir/R_a"
    if [ "$name" != R_b ]; then
        run "$dir/warm_$name" "$A2/run_a2.py" --mode warmup --model "$model" --prompts-dir "$P" $EXTRA_ARGS "$@"
        done_ok "$dir/warm_$name" || log "WARNING: warm-up $dir/warm_$name failed; the next run starts cold"
    fi
    run "$dir/$name" "$A2/run_a2.py" --mode full --model "$model" --prompts-dir "$P" $ref $LIM $EXTRA_ARGS "$@"
    case "$name" in R_*|bf16) ;; *) grep -q "Directly load AOT" "$dir/$name/log.txt" 2>/dev/null \
        || log "WARNING: $dir/$name did not load the AOT compile cache";; esac
}

speed() {  # $1 cell dir, $2 model, $3 flag, $4 candidate: run_cell.py throughput in the order R, C, R, C
    dir=$1; model=$2; flag=$3; v=$4; S="$dir/speed/$v"
    RC="$CODE/safedispatch_testA/run_cell.py"
    capped "$NEW_RUN_CAP" && { log "time cap: skip speed $dir/$v"; return; }
    run "$S/warm_R" "$RC" --mode warmup --model "$model" --prompts-dir "$P" $SPEED_ARGS
    run "$S/warm_C" "$RC" --mode warmup --model "$model" --prompts-dir "$P" $SPEED_ARGS "$flag" "$v"
    for i in 1 2; do
        run "$S/R_$i" "$RC" --mode throughput --model "$model" --prompts-dir "$P" $SPEED_ARGS
        run "$S/C_$i" "$RC" --mode throughput --model "$model" --prompts-dir "$P" $SPEED_ARGS "$flag" "$v"
    done
    for r in R_1 C_1 R_2 C_2; do  # A1 (R-43): a cold compile must not enter a throughput run
        grep -q "Directly load AOT" "$S/$r/log.txt" 2>/dev/null || log "WARNING: $S/$r did not load the AOT compile cache"
    done
}

accuracy() {  # $1 cell dir, $2 model, $3 BF16 anchor, $4 anchor args, $5 backend flag, $6 candidates
    dir=$1; model=$2; anchor=$3; aargs=$4; flag=$5; cands=$6
    a2 "$dir" R_a "$model"
    a2 "$dir" R_b "$model"
    for v in $cands; do a2 "$dir" "cand_$v" "$model" "$flag" "$v"; done
    a2 "$dir" bf16 "$anchor" $aargs
    ok=$(python3 "$A2/analyze_a2.py" accept --cell "$dir" --out "$dir/accept.json" 2>>"$RES/session.log")
    log "$(basename "$dir") acceptable: ${ok:-none}"
}

planned() {  # $1 cell name: the candidates in SPEED_PLAN for that cell
    echo "$SPEED_PLAN" | tr ';' '\n' | sed -n "s/^$1://p" | tr ',' ' '
}

# A1 (R-43): the decisive data first. The MoE cell holds the only GO path (Marlin, E-36 B2).
case " $CELLS " in *" M "*)
    accuracy "$RES/cell_M" "$MOE" "$MOE_BF16" "$ANCHOR_ARGS_M" --moe-backend "$MOE_CANDS"
    for v in $(planned cell_M); do speed "$RES/cell_M" "$MOE" --moe-backend "$v"; done;;
esac
case " $CELLS " in *" D "*)
    accuracy "$RES/cell_D" "$DENSE" "$DENSE_BF16" "" --linear-backend "$DENSE_CANDS"
    for v in $(planned cell_D); do speed "$RES/cell_D" "$DENSE" --linear-backend "$v"; done;;
esac

cells=""; for c in $CELLS; do [ -d "$RES/cell_$c" ] && cells="$cells $RES/cell_$c"; done
python3 "$A2/analyze_a2.py" decide --cells $cells --speed-plan "$SPEED_PLAN" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
[ -f "$CODE/runpod/gpu_microbench.py" ] && python3 "$CODE/runpod/gpu_microbench.py" --out "$ENV/microbench_end.json" > /dev/null 2>&1
kill "$TELEMETRY" 2>/dev/null
log "done after $(elapsed) s; copy $RES off the machine before terminating it"
