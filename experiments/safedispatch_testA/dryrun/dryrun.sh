#!/bin/sh
# E-36 Mac dry run: mechanics only, on the vLLM v0.32.0 CPU backend (linux/arm64 image built locally).
# Model Qwen/Qwen3-0.6B from the local Hugging Face cache; no network; 16 prompts, short outputs.
# The numbers are NOT results. Usage (from the host):
#   docker run --rm --network none --cpus 8 --memory 7g \
#     -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_CPU_KVCACHE_SPACE=2 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
#     -v <repo>/experiments/safedispatch_testA:/code:ro -v <out>:/results \
#     -v <repo>/experiments/safedispatch_oracle/prompts.json:/local/prompts.json:ro \
#     -v <repo>/experiments/safedispatch_testA/dryrun/cpu_triton_utils_v0.32.0_pr61992.py:/opt/venv/lib/python3.12/site-packages/vllm/utils/cpu_triton_utils.py:ro \
#     --entrypoint sh vllm-cpu:v0.32.0 /code/dryrun/dryrun.sh
set -u
CODE=/code
RES=/results/dryrun-$(date -u +%Y%m%dT%H%M%SZ)
MODEL=Qwen/Qwen3-0.6B
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$RES/cell_D"
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__)" > "$RES/versions.txt" 2>&1
python3 "$CODE/prompts.py" --source local --local-prompts /local/prompts.json --tokenizer "$MODEL" --n-each 8 --n-prefill 4 --prefill-tokens 300 \
    --out "$RES/prompts" > "$RES/prompts.log" 2>&1 || { echo "prompts failed"; cat "$RES/prompts.log"; exit 1; }

common="--model $MODEL --prompts-dir $RES/prompts --max-model-len 2048 --max-num-seqs 8 --max-tokens 32 \
--throughput-max-tokens 16 --batch-sizes 1 4 --repeats 2 --gpu-memory-utilization 0.5"
for run in R_a R_b; do
    mkdir -p "$RES/cell_D/$run"
    python3 "$CODE/run_cell.py" --mode both --out "$RES/cell_D/$run" $common > "$RES/cell_D/$run/log.txt" 2>&1
    echo "$run exit $?"
done
# A candidate flag on CPU: the BF16 model has no FP8 layers, so this checks that the flag is accepted
# and that the fallback warning (or its absence) is captured in kernel_log.
mkdir -p "$RES/cell_D/cand_torch"
python3 "$CODE/run_cell.py" --mode both --out "$RES/cell_D/cand_torch" $common --linear-backend torch \
    > "$RES/cell_D/cand_torch/log.txt" 2>&1
echo "cand_torch exit $?"
python3 "$CODE/compare.py" --cells "$RES/cell_D" --out "$RES/summary.json"
echo "results in $RES"
