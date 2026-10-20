"""Build the frozen E-36 prompt set.

--source hub   (GPU box) 128 GSM8K test questions + 128 UltraChat test_sft first user turns, chosen
               by a fixed seed, with the dataset revisions recorded.
--source local (Mac dry run) the E-23 pilot prompts from this repository.

Writes <out>/prompts.json (greedy set, chat template applied), <out>/prefill.json (long prompts for
the prefill throughput test) and <out>/prompts_manifest.json (hashes and revisions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20261020
HERE = Path(__file__).resolve().parent


def chat(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def from_hub(n_each: int, tokenizer, max_tokens: int):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revisions = {
        "openai/gsm8k": api.dataset_info("openai/gsm8k").sha,
        "HuggingFaceH4/ultrachat_200k": api.dataset_info("HuggingFaceH4/ultrachat_200k").sha,
    }
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=revisions["openai/gsm8k"])
    uc = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="test_sft", revision=revisions["HuggingFaceH4/ultrachat_200k"]
    )
    rng = random.Random(SEED)

    def pick(name, size, text_of):
        """Take indices in seeded order and keep those whose chat prompt fits max_tokens."""
        chosen = []
        for i in rng.sample(range(size), size):
            text = text_of(i)
            if len(tokenizer(chat(tokenizer, text))["input_ids"]) <= max_tokens:
                chosen.append((name, i, text))
            if len(chosen) == n_each:
                return chosen
        raise SystemExit(f"{name}: only {len(chosen)} prompts fit {max_tokens} tokens")

    items = pick("gsm8k", len(gsm), lambda i: gsm[i]["question"])
    items += pick("ultrachat", len(uc), lambda i: uc[i]["messages"][0]["content"])
    return items, revisions


def from_local(n: int, path: Path):
    texts = json.loads(path.read_text())
    return [("e23_local", i, t) for i, t in enumerate(texts[:n])], {"local": str(path)}


def long_prompts(tokenizer, texts: list[str], n: int, target_tokens: int) -> list[str]:
    rng = random.Random(SEED + 1)
    out = []
    for _ in range(n):
        parts, total = [], 0
        while total < target_tokens:
            t = texts[rng.randrange(len(texts))]
            parts.append(t)
            total += len(tokenizer(t)["input_ids"])
        out.append(chat(tokenizer, "Summarize the following text.\n\n" + "\n\n".join(parts)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("hub", "local"), required=True)
    parser.add_argument("--tokenizer", required=True, help="model id or path whose chat template is used")
    parser.add_argument("--n-each", type=int, default=128)
    parser.add_argument("--n-prefill", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=1000)
    parser.add_argument("--local-prompts", default=str(HERE.parent / "safedispatch_oracle" / "prompts.json"))
    parser.add_argument("--max-prompt-tokens", type=int, default=3072,
                        help="drop prompts longer than this (max_model_len 4096 minus 256 new tokens, with margin)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.source == "hub":
        items, revisions = from_hub(args.n_each, tokenizer, args.max_prompt_tokens)
    else:
        items, revisions = from_local(2 * args.n_each, Path(args.local_prompts))
    prompts = [
        {"id": f"{src}-{idx}", "source": src, "index": idx, "prompt": chat(tokenizer, text)}
        for src, idx, text in items
    ]
    prefill = long_prompts(tokenizer, [t for _, _, t in items], args.n_prefill, args.prefill_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1))
    (out / "prefill.json").write_text(json.dumps(prefill, indent=1))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    manifest = {
        "seed": SEED,
        "max_prompt_tokens": args.max_prompt_tokens,
        "source": args.source,
        "tokenizer": args.tokenizer,
        "revisions": revisions,
        "n_prompts": len(prompts),
        "n_prefill": len(prefill),
        "sha256": {"prompts.json": sha(out / "prompts.json"), "prefill.json": sha(out / "prefill.json")},
    }
    (out / "prompts_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
