"""E-25: does a small, query-agnostic structurer keep the text a coding agent later edits?

Inputs are E-24 samples (a Read tool result plus the Edit old_string later copied from it).
They contain the user's code, so they live in a scratch directory, and only aggregate numbers
are written to --out. Per-sample outputs go to --detail-out, which must also be scratch.

Predeclared conditions and criteria: RESEARCH_JOURNAL.md, Section 6.1, E-25.
"""

import argparse
import json
import math
import random
import re
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
N_SAMPLES = 30
KEEP_FRACTION = 0.25
READ_PREFIX = re.compile(r"^\s*\d+\t")

ABSTRACTIVE = (
    "You are a context compressor for an AI coding agent. Rewrite the tool output below as "
    "terse, structured facts that the agent will rely on later. Keep only what matters. "
    "Output only the compressed result.\n\nTool: Read\nOutput:\n<<<\n{text}\n>>>"
)
EXTRACTIVE = (
    "Below is a file shown with line numbers. Choose the lines an AI coding agent is most "
    "likely to need later. Answer with line numbers only, as a comma-separated list, at most "
    "{k} numbers.\n\n<<<\n{text}\n>>>"
)


def numbered_lines(read_result):
    """(line number, content) pairs from a Read result, in file order."""
    out = []
    for i, line in enumerate(read_result.split("\n")):
        m = re.match(r"^\s*(\d+)\t", line)
        out.append((int(m.group(1)) if m else i + 1, READ_PREFIX.sub("", line)))
    return out


def survival(old_string, kept_text):
    lines = [l.strip() for l in old_string.split("\n") if l.strip()]
    kept = kept_text
    return {
        "exact": old_string in kept,
        "line_recall": sum(l in kept for l in lines) / max(1, len(lines)),
    }


def generate(model, tok, prompt, max_new):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    if model.device.type == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    new = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), len(new), ids["input_ids"].shape[1], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail-out", required=True)
    args = ap.parse_args()

    samples = json.load(open(args.samples))[:N_SAMPLES]
    tok = AutoTokenizer.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

    rows = []
    for idx, s in enumerate(samples):
        raw, old = s["read_result"], s["old_string"]
        lines = numbered_lines(raw)
        k = max(1, math.ceil(KEEP_FRACTION * len(lines)))
        plain = "\n".join(c for _, c in lines)
        n_in = len(tok(raw)["input_ids"])

        # A: abstractive structurer, the design proposed for I-41.
        a_text, a_new, a_prompt, a_dt = generate(model, tok, ABSTRACTIVE.format(text=raw), n_in // 4 + 64)

        # B: extractive structurer, the model names the lines to keep.
        b_ans, b_new, b_prompt, b_dt = generate(model, tok, EXTRACTIVE.format(text=raw, k=k), 8 * k + 16)
        wanted = []
        for n in re.findall(r"\d+", b_ans):
            if int(n) not in wanted:
                wanted.append(int(n))
        wanted = set(wanted[:k])
        b_text = "\n".join(c for n, c in lines if n in wanted)

        # C: first and last lines. D: random lines.
        head = math.ceil(k / 2)
        c_idx = set(range(head)) | set(range(len(lines) - (k - head), len(lines)))
        c_text = "\n".join(c for i, (_, c) in enumerate(lines) if i in c_idx)
        d_idx = set(random.Random(idx).sample(range(len(lines)), k))
        d_text = "\n".join(c for i, (_, c) in enumerate(lines) if i in d_idx)

        row = {"lag": s["lag"], "lines": len(lines), "input_tokens": n_in, "old_lines": old.count("\n") + 1}
        for name, text in (("A", a_text), ("B", b_text), ("C", c_text), ("D", d_text)):
            row[name] = {**survival(old, text), "char_ratio": len(text) / max(1, len(plain))}
        row["A"].update({"new_tokens": a_new, "seconds": a_dt, "prompt_tokens": a_prompt})
        row["B"].update({"new_tokens": b_new, "seconds": b_dt, "prompt_tokens": b_prompt,
                         "lines_returned": len(wanted)})
        rows.append({**row, "A_text": a_text, "B_answer": b_ans})
        print(idx, {n: (row[n]["exact"], round(row[n]["line_recall"], 2), round(row[n]["char_ratio"], 2))
                    for n in "ABCD"}, round(a_dt, 1), round(b_dt, 1), flush=True)

    summary = {"model": MODEL, "device": device, "n": len(rows)}
    for name in "ABCD":
        summary[name] = {
            "exact_survival": sum(r[name]["exact"] for r in rows) / len(rows),
            "mean_line_recall": statistics.mean(r[name]["line_recall"] for r in rows),
            "median_char_ratio": statistics.median(r[name]["char_ratio"] for r in rows),
        }
    for name in "AB":
        secs = [r[name]["seconds"] for r in rows]
        summary[name]["median_seconds"] = statistics.median(secs)
        summary[name]["p90_seconds"] = statistics.quantiles(secs, n=10)[-1]
        summary[name]["tokens_per_second"] = (
            sum(r[name]["new_tokens"] for r in rows) / sum(secs))
    summary["median_input_tokens"] = statistics.median(r["input_tokens"] for r in rows)
    summary["A_hit_max_new_tokens_share"] = sum(
        r["A"]["new_tokens"] >= r["input_tokens"] // 4 + 64 for r in rows) / len(rows)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.detail_out, "w") as fh:
        json.dump(rows, fh)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
