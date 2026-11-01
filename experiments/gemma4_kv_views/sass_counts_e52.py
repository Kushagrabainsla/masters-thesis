"""E-52 SASS counts (A1, R-71 F2; diagnostic only): for each saved cubin, the registers, spills and shared memory
from `cuobjdump --dump-resource-usage`, and the counts of the SASS opcodes that H4 and the staging question name
(conversions F2F and F2FP, HMMA, the async copy LDGSTS, LDS, STS, BAR, LDG). It answers which SASS instruction
`cvt.bf16.f16` becomes on sm_90. Without `cuobjdump` it writes that and stops.

Usage: sass_counts_e52.py --out <json> <cubin dir> [<cubin dir> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

OPS = ("F2F", "F2FP", "HMMA", "HGMMA", "LDGSTS", "LDS", "STS", "BAR", "LDG", "PRMT", "LOP3", "HADD2", "HMUL2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("dirs", nargs="+", type=Path)
    a = ap.parse_args()
    tool = shutil.which("cuobjdump") or ("/usr/local/cuda/bin/cuobjdump"
                                         if Path("/usr/local/cuda/bin/cuobjdump").exists() else None)
    if not tool:
        a.out.write_text(json.dumps({"error": "cuobjdump not found"}))
        return
    res = {}
    for d in a.dirs:
        for f in sorted(d.glob("*.cubin")):
            entry = {}
            try:
                use = subprocess.run([tool, "--dump-resource-usage", str(f)], capture_output=True, text=True,
                                     timeout=60).stdout
                for k in ("REG", "STACK", "SHARED", "LOCAL"):
                    m = re.search(rf"{k}:(\d+)", use)
                    entry[k.lower()] = int(m.group(1)) if m else None
                sass = subprocess.run([tool, "-sass", str(f)], capture_output=True, text=True, timeout=120).stdout
                ops = [m.group(1) for m in re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9]*)", sass)]
                entry["ops"] = {op: sum(1 for o in ops if o == op) for op in OPS}
                entry["f2f_variants"] = sorted(set(re.findall(r"\bF2FP?\.[A-Z0-9_.]+", sass)))[:12]
            except Exception as exc:  # reported only
                entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            res[f"{d.name}/{f.stem}"] = entry
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
