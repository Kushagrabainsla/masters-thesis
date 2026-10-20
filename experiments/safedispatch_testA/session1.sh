#!/bin/sh
# E-36 Session 1 on one rented H100 (amendment A1 after R-34). Run inside the official
# vllm/vllm-openai:v0.32.0 container with --entrypoint sh, this directory mounted at /code and a
# writable results directory at /results. It stops starting new runs after STOP_AFTER seconds and
# always runs the analysis.
set -u

STOP_AFTER=${STOP_AFTER:-11400}   # 3 h 10 min
RUN_TIMEOUT=${RUN_TIMEOUT:-1500}  # 25 min per run
CODE=${CODE:-/code}
RES=${RES:-/results/session1-$(date -u +%Y%m%dT%H%M%SZ)}
DENSE=${DENSE:-Qwen/Qwen3-8B-FP8}
MOE=${MOE:-Qwen/Qwen3-30B-A3B-FP8}
# Candidates (R-34): no aliases of the reference, no backends that cannot run block FP8 on Hopper.
DENSE_CANDS=${DENSE_CANDS:-"deep_gemm cutlass triton torch marlin"}   # marlin = W8A16 positive control
MOE_CANDS=${MOE_CANDS:-"deep_gemm flashinfer_cutlass hpc marlin triton"}  # triton = expected alias (negative control)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
START=$(date +%s)
mkdir -p "$RES"

over_time() { [ $(( $(date +%s) - START )) -ge "$STOP_AFTER" ]; }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }

run() {  # $1 out dir, then run_cell.py arguments; never fails the script
    out=$1; shift
    mkdir -p "$out"
    timeout "$RUN_TIMEOUT" python3 "$CODE/run_cell.py" --out "$out" "$@" > "$out/log.txt" 2>&1
    log "$(basename "$(dirname "$out")")/$(basename "$out"): exit $?"
    sleep 10  # let the GPU memory of the finished process be released
}

# Step 0: environment manifest and prompts.
nvidia-smi -q > "$RES/nvidia-smi.txt" 2>&1
pip install --quiet datasets > "$RES/pip.txt" 2>&1
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)" > "$RES/versions.txt" 2>&1
python3 "$CODE/prompts.py" --source hub --tokenizer "$DENSE" --out "$RES/prompts" > "$RES/prompts.log" 2>&1 \
    || { log "prompts failed"; exit 1; }

cell() {  # $1 cell dir, $2 model, $3 flag, $4 candidates
    dir=$1; model=$2; flag=$3; cands=$4
    over_time && { log "time cap: skip $dir"; return; }
    run "$dir/R_a" --mode both --model "$model" --prompts-dir "$RES/prompts"
    grep -qE "Selected [^ ]+ for |Fp8 MoE backend out of" "$dir/R_a/log.txt" \
        || log "WARNING: no kernel-selection line in $dir/R_a/log.txt; alias detection cannot work"
    run "$dir/R_b" --mode both --model "$model" --prompts-dir "$RES/prompts"
    for v in $cands; do
        over_time && { log "time cap: stop candidates in $dir"; break; }
        run "$dir/cand_$v" --mode both --model "$model" --prompts-dir "$RES/prompts" "$flag" "$v"
    done
}

cell "$RES/cell_D" "$DENSE" --linear-backend "$DENSE_CANDS"
cell "$RES/cell_M" "$MOE" --moe-backend "$MOE_CANDS"

# Analysis (always runs).
python3 "$CODE/compare.py" --cells "$RES/cell_D" "$RES/cell_M" --out "$RES/summary.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(cat "$RES/decision.txt")"
log "done after $(( $(date +%s) - START )) s; copy $RES off the machine before terminating it"
