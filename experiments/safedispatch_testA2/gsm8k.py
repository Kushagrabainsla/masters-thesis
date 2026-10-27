"""Build the E-38 GSM8K prompt set (all 1,319 test problems) and parse answers.

Output (<out>/):
  gsm8k.json      [{"id", "index", "prompt", "gold"}], chat template applied, thinking off
  prompts.json    the same prompts in the E-36 run_cell.py format (id, prompt), for warm-ups and speed
  prefill.json    the 64 longest prompts as text, for run_cell.py's prefill test
  manifest.json   dataset revision, tokenizer, SHA-256 of each file

Usage: gsm8k.py --tokenizer <model> --out <dir> [--source hub | --local <jsonl with question, answer>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED = re.compile(r"\\boxed\{")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def gold_of(answer: str) -> str:
    """The gold answer is the number after '####' in GSM8K, without commas."""
    return normalize(answer.split("####")[-1])


def normalize(text: str) -> str:
    text = text.replace(",", "").replace("$", "").strip()
    m = NUMBER.findall(text)
    if not m:
        return ""
    value = m[-1]
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value.lstrip("+")


def extract(completion: str) -> str:
    """The value of the last \\boxed{...} (braces balanced); empty if there is none."""
    starts = [m.end() for m in BOXED.finditer(completion)]
    if not starts:
        return ""
    i, depth, out = starts[-1], 1, []
    while i < len(completion) and depth:
        ch = completion[i]
        depth += ch == "{"
        depth -= ch == "}"
        if depth:
            out.append(ch)
        i += 1
    return normalize("".join(out))


def chat(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", choices=("hub", "local"), default="hub")
    parser.add_argument("--local", type=Path, help="jsonl with question and answer (dry run)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.source == "hub":
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info("openai/gsm8k").sha
        rows = load_dataset("openai/gsm8k", "main", split="test", revision=revision)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    else:
        revision = str(args.local)
        rows = [json.loads(line) for line in args.local.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    items = [{"id": f"gsm8k-{i}", "index": i, "prompt": chat(tokenizer, r["question"]), "gold": gold_of(r["answer"])}
             for i, r in enumerate(rows)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gsm8k.json").write_text(json.dumps(items))
    (args.out / "prompts.json").write_text(json.dumps([{"id": x["id"], "prompt": x["prompt"]} for x in items]))
    longest = sorted(items, key=lambda x: -len(x["prompt"]))[:64]
    (args.out / "prefill.json").write_text(json.dumps([x["prompt"] for x in longest]))
    manifest = {"source": args.source, "revision": revision, "tokenizer": args.tokenizer, "n": len(items),
                "instruction": INSTRUCTION,
                "sha256": {f: sha(args.out / f) for f in ("gsm8k.json", "prompts.json", "prefill.json")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"n": len(items), "revision": revision}))


if __name__ == "__main__":
    main()
