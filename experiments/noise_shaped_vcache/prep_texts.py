"""E-27 text preparation. Writes raw texts of three corpora to a JSON file (no tokenization).

Run with an interpreter that has pyarrow (the agent corpus reads Parquet), for example
~/Downloads/agent-llm-traces/.venv/bin/python. All sources are local files; nothing is downloaded.
- prose: this repo's Markdown (RESEARCH_JOURNAL.md and dossier/*.md)
- code: Python source of torch._inductor and torch._dynamo in the experiment virtual environment
- agent: SWE-bench agent traces from the Exgentic dataset shards in ~/Downloads/agent-llm-traces
"""

import argparse
import glob
import json
import os
import random


def prose(repo):
    files = [os.path.join(repo, "RESEARCH_JOURNAL.md")] + sorted(glob.glob(os.path.join(repo, "dossier", "*.md")))
    return ["\n\n".join(open(f, encoding="utf-8").read() for f in files)]


def code(torch_dir):
    # Compiler code has fewer docstrings and less boilerplate than torch.nn (amendment A1).
    files = sorted(glob.glob(os.path.join(torch_dir, "_inductor", "*.py")))
    files += sorted(glob.glob(os.path.join(torch_dir, "_dynamo", "*.py")))
    return ["\n\n".join(open(f, encoding="utf-8").read() for f in files)]


def agent(data_dir, n, seed):
    import pyarrow.parquet as pq
    texts = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        for row in pq.read_table(f, columns=["benchmark", "spans"]).to_pylist():
            if row["benchmark"] != "swebench":
                continue
            spans = row["spans"]
            best = max(spans, key=lambda s: len((s["attributes"] or {}).get("gen_ai.input.messages") or ""))
            msgs = json.loads(best["attributes"]["gen_ai.input.messages"])
            parts = []
            for m in msgs:
                for p in m.get("parts", []):
                    for key in ("content", "result", "arguments"):
                        val = p.get(key)
                        if isinstance(val, (dict, list)):
                            val = json.dumps(val)
                        if isinstance(val, str):
                            parts.append(f"{m.get('role')}: {val}")
            text = "\n".join(parts)
            if len(text) > 40000:
                texts.append(text)
    random.Random(seed).shuffle(texts)
    return texts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--torch-dir", required=True)
    ap.add_argument("--agent-dir", default=os.path.expanduser("~/Downloads/agent-llm-traces/data"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"prose": prose(args.repo), "code": code(args.torch_dir), "agent": agent(args.agent_dir, 12, 27)}
    json.dump(out, open(args.out, "w"))
    print({k: (len(v), sum(len(t) for t in v)) for k, v in out.items()})


if __name__ == "__main__":
    main()
