#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
experiment_dir="$repo_root/experiments/versioned_session_commit"
image_name="i122-e31:local"
run_id=$(date -u +%Y%m%dT%H%M%SZ)
result_dir="$experiment_dir/results/$run_id"
cpu_set=${E31_CPU_SET:-0-3}
cpu_limit=${E31_CPU_LIMIT:-4}
memory_limit=${E31_MEMORY_LIMIT:-4g}

mkdir -p "$result_dir"

docker build \
  --file "$experiment_dir/Dockerfile" \
  --tag "$image_name" \
  "$repo_root"

image_id=$(docker image inspect --format '{{.Id}}' "$image_name")

run_container() {
  docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=256m \
    --cpuset-cpus "$cpu_set" \
    --cpus "$cpu_limit" \
    --memory "$memory_limit" \
    --pids-limit 256 \
    --env "E31_IMAGE_ID=$image_id" \
    --env "E31_CPU_LIMIT=$cpu_limit" \
    --env "E31_MEMORY_LIMIT=$memory_limit" \
    --volume "$result_dir:/results" \
    "$image_name" "$@"
}

run_container -m unittest -v test_model.py
run_container run_e31.py \
  --output-dir /results \
  --model-check-depth 6 \
  --repeats 5 \
  --turns 20
run_container decide_e31.py --result-dir /results

printf '%s\n' "$result_dir"
