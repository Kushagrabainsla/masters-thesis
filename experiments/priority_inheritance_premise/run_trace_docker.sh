#!/bin/sh
# E-35 amendment A1: run the frozen trace check in the pinned image with no network.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model_cfg=${1:?usage: run_trace_docker.sh <dir with config.json and generation_config.json>}
image="e35-vllm-sched:3897f95"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_dir="$script_dir/results/trace-$timestamp"
mkdir -p "$result_dir"

docker run --rm \
    --network none \
    --cpus 4 \
    --memory 4g \
    -e VLLM_TARGET_DEVICE=cpu \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "$model_cfg:/model:ro" \
    -v "$script_dir:/code:ro" \
    "$image" /code/trace_check.py >"$result_dir/trace_check.json" 2>"$result_dir/stderr.log"

python3 - "$result_dir" "$image" "$(docker image inspect "$image" --format '{{.Id}}')" "$script_dir" <<'PY'
import hashlib, json, sys
from datetime import UTC, datetime
from pathlib import Path
out, image, image_id, code = sys.argv[1:]
code = Path(code)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = {
    "schema": "e35-trace-manifest-v1",
    "completed_at": datetime.now(UTC).isoformat(),
    "container_image": image,
    "container_image_id": image_id,
    "vllm_sha": "3897f95cf17619137536d2a7d325d9c2f39dc38f",
    "network": "none",
    "sha256": {name: sha(code / name) for name in ("README.md", "trace_check.py", "harness.py", "Dockerfile")},
    "result_sha256": sha(Path(out) / "trace_check.json"),
}
(Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
printf '%s\n' "$result_dir"
