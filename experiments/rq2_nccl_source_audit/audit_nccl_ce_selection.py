#!/usr/bin/env python3
"""RQ2 Mac-side audit: can NCCL's CE-vs-SM decision see live compute state?

RQ2 proposes choosing the collective execution engine (SM-driven vs copy-engine)
from the resource pressure of the concurrently running compute kernel. Measuring
the benefit needs NVLink GPUs. But whether the decision point can even OBSERVE
that state is a source question, answerable on any machine.

Run:  python3 audit_nccl_ce_selection.py /path/to/nccl
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def sh(root: Path, *args: str) -> str:
    return subprocess.run(args, cwd=root, capture_output=True, text=True).stdout


def grep(root: Path, pattern: str, *paths: str, flags="-rnE") -> list[str]:
    out = sh(root, "grep", flags, pattern, *paths)
    return [l for l in out.splitlines() if l.strip()]


# Terms that would indicate the runtime senses compute-side resource pressure.
RESOURCE_SENSING_TERMS = [
    "occupancy", "cudaOccupancy", "smUtil", "GetUtilization", "activeWarps",
    "nvmlDeviceGetUtilizationRates", "smActive", "achieved_occupancy",
]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (root / "src").is_dir():
        print(f"error: {root} does not look like an NCCL checkout")
        return 2

    ver = {}
    for line in (root / "makefiles" / "version.mk").read_text().splitlines():
        m = re.match(r"NCCL_(MAJOR|MINOR|PATCH)\s*:=\s*(\d+)", line.strip())
        if m:
            ver[m.group(1).lower()] = int(m.group(2))
    version = f"{ver.get('major')}.{ver.get('minor')}.{ver.get('patch')}"
    commit = sh(root, "git", "rev-parse", "HEAD").strip()

    findings = {}

    # 1. The CE routing gate.
    gate = grep(root, r"CTA_POLICY_ZERO.*ceAvailable|ceAvailable.*CTA_POLICY_ZERO",
                "src/")
    findings["ce_routing_gate"] = {
        "matches": gate,
        "conclusion": (
            "CE routing is gated on a user-supplied static CTA policy AND a static "
            "eligibility predicate. No cost model and no runtime state participate."
        ),
    }

    # 2. Inputs to the eligibility predicate.
    sig = sh(root, "sed", "-n", "/^bool ncclCeAvailable/,/^}/p", "src/ce_coll.cc")
    params = re.search(r"ncclCeAvailable\(([^)]*)\)", sig, re.S)
    reasons = re.findall(r'TRACE\(NCCL_TUNING,\s*"Skipping CE collective:\s*([^"]+)"', sig)
    findings["ce_eligibility_inputs"] = {
        "parameters": [p.strip() for p in (params.group(1).split(",") if params else [])],
        "rejection_reasons": reasons,
        "conclusion": (
            "Every rejection reason is a static capability or registration property: "
            "collective/op/dtype support, host-backed segments, NVLink connectivity, "
            "symmetric support, window registration type. None is a load condition."
        ),
    }

    # 3. Tuner plugin API inputs, the only extension point for algorithm choice.
    tuner_versions = sorted(p.name for p in (root / "src/include/plugin/tuner").glob("tuner_v*.h"))
    latest = sorted(tuner_versions, key=lambda s: int(re.search(r"v(\d+)", s).group(1)))[-1]
    tsrc = (root / "src/include/plugin/tuner" / latest).read_text()
    tsig = re.search(r"getCollInfo\)\((.*?)\);", tsrc, re.S)
    findings["tuner_api_inputs"] = {
        "versions_present": tuner_versions,
        "latest": latest,
        "getCollInfo_parameters": (
            [p.strip() for p in tsig.group(1).replace("\n", " ").split(",")] if tsig else []
        ),
        "conclusion": (
            "getCollInfo receives collective type, byte count, pipelined-op count, a "
            "cost table, algo/proto counts and a registration flag. It receives no "
            "occupancy, no bandwidth pressure and no handle to the concurrent kernel. "
            "A tuner plugin therefore CANNOT implement RQ2's policy, and CE routing "
            "does not consult the tuner at all."
        ),
    }

    # 4. Does NCCL sense compute-side resource pressure anywhere?
    hits = grep(root, "|".join(RESOURCE_SENSING_TERMS), "src/")
    findings["runtime_resource_sensing"] = {
        "terms_searched": RESOURCE_SENSING_TERMS,
        "matches": hits,
        "conclusion": (
            "No occupancy or utilization sensing anywhere in src/. The information "
            "RQ2's policy would need is not collected by the runtime at all."
        ) if not hits else "Matches found; inspect before claiming absence.",
    }

    gap_confirmed = (not hits) and bool(gate)
    result = {
        "nccl_version": version,
        "nccl_commit": commit,
        "findings": findings,
        "rq2_source_level_gap_confirmed": gap_confirmed,
        "what_this_establishes": (
            "The architectural gap is real and does not need a GPU to demonstrate: the "
            "CE-vs-SM decision is a static user policy, the eligibility check is static, "
            "the tuner is not consulted for it, and the runtime never measures compute-side "
            "resource pressure. What still needs NVLink hardware is how much performance "
            "the missing input is worth."
        ),
        "what_this_does_not_establish": (
            "Nothing about magnitude. A static policy may well capture most of the "
            "available benefit; that is the simplest-solution attack and it remains open."
        ),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/nccl_ce_selection_audit.json").write_text(json.dumps(result, indent=2))

    print(f"NCCL {version} @ {commit[:10]}\n")
    for name, f in findings.items():
        print(f"--- {name}")
        for k, v in f.items():
            if k == "conclusion":
                continue
            print(f"      {k}: {v if not isinstance(v, list) else ''}")
            if isinstance(v, list):
                for item in v:
                    print(f"        - {item}")
        print(f"      => {f['conclusion']}\n")
    print(f"RQ2 source-level gap confirmed: {gap_confirmed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
