#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
results_root="$script_dir/results"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_dir="$results_root/source-audit-$timestamp"
image="python:3.12-slim"

mkdir -p "$results_root"
mkdir "$result_dir"

image_id=$(docker image inspect "$image" --format '{{.Id}}')
docker run --rm \
    --network bridge \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=256m \
    --cpus 4 \
    --memory 4g \
    --pids-limit 256 \
    --user "$(id -u):$(id -g)" \
    --volume "$script_dir/source_audit.py:/audit.py:ro" \
    "$image" \
    python /audit.py >"$result_dir/audit.json"

audit_sha256=$(shasum -a 256 "$result_dir/audit.json" | awk '{print $1}')
script_sha256=$(shasum -a 256 "$script_dir/source_audit.py" | awk '{print $1}')

python3 - "$result_dir/manifest.json" "$image" "$image_id" "$audit_sha256" "$script_sha256" <<'PY'
import json
import sys
from datetime import UTC, datetime

path, image, image_id, audit_sha256, script_sha256 = sys.argv[1:]
manifest = {
    "schema": "i122-source-audit-manifest-v1",
    "completed_at": datetime.now(UTC).isoformat(),
    "container_image": image,
    "container_image_id": image_id,
    "container_cpus": 4,
    "container_memory": "4g",
    "audit_sha256": audit_sha256,
    "source_audit_script_sha256": script_sha256,
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(manifest, output, indent=2, sort_keys=True)
    output.write("\n")
PY

printf '%s\n' "$result_dir"
