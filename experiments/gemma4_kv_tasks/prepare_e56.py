"""E-56 data (README Section 4.3): the task items as token ids of the pinned Gemma 4 31B tokenizer and the chat template
of the BF16 checkpoint (thinking off, the template default), frozen in `data_e56.json` before any E-56 run. Runs on the
Mac (no GPU).

- gsm8k: 400 questions of the GSM8K test split (openai/gsm8k, MIT, the pinned revision), seed 56. A chat prompt with the
  generation prompt, the gold number, and a scoring sequence: the prompt, the reference solution (the calculator notes
  `<<...>>` removed) with its final line `#### <gold>`, and the end-of-turn token. `answer_start` is the first scored
  position (the first token of the reference).
- needle: 32 prompts of about 7,700 tokens: 4 PG-19 test books (the E-55 local copies, seed 57), 8 depths each, one
  passphrase sentence and a question at the end.
- health: BOS and the first 2,047 tokens of the E-55 documents B1, P1 and C1 (raw text; reported; R-78). B1 is also the
  probe that each arm scores alone first (the V5 and V7 evidence needs one prompt in the call).

Usage: prepare_e56.py [--out data_e56.json]   (downloads go to data/, which git ignores)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
E55 = HERE.parent / "gemma4_kv_model"
MODEL_ID = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
GSM_URL = ("https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/"
           "main/test-00000-of-00001.parquet")
GSM_N, GSM_SEED = 400, 56
GSM_ASK = ("Solve the math problem. Show brief steps, then write the final answer as a number after '####'.\n\n"
           "Problem: ")
NEEDLE_BOOKS, NEEDLE_DEPTHS, NEEDLE_SEED = 4, (0.05, 0.17, 0.29, 0.41, 0.53, 0.65, 0.77, 0.89), 57
HAY_SKIP, HAY_TOKENS = 1_000, 7_600          # skip the book header; the haystack length in tokens
NEEDLE_ASK = "Read the document. Then answer the question at the end.\n\n<document>\n"
NEEDLE_Q = ("\n</document>\n\nQuestion: What is the secret passphrase for the archive? Answer with the passphrase "
            "only.")
WORDS = ("amber", "basalt", "cobalt", "dune", "ember", "fjord", "granite", "harbor", "indigo", "juniper", "kestrel",
         "lagoon", "marble", "nectar", "onyx", "pebble", "quartz", "raven", "saffron", "tundra", "umber", "velvet",
         "willow", "xenon", "yarrow", "zephyr", "acorn", "bramble", "cinder", "delta", "falcon", "garnet")
HEALTH_DOCS, HEALTH_TOKENS = ("B1", "P1", "C1"), 2_048
TOK_CHECK = ("Hello world.", "def f(x):\n    return x * 2\n", "Grüße, naïve café: 3 × 4 = 12.")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def fetch(url: str, path: Path) -> bytes:
    """The file at url, kept in path (a second call reads the copy)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "msthesis-prepare/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def tokenizer(tok_dir: Path):
    """The pinned tokenizer and chat template (transformers, as the Pod image), from local copies of its files."""
    from transformers import AutoTokenizer
    for name in TOK_FILES:
        fetch(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", tok_dir / name)
    return AutoTokenizer.from_pretrained(str(tok_dir))


def chat_ids(tok, msg: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)   # transformers 5 returns a BatchEncoding


def gold_number(answer: str) -> str:
    """The GSM8K gold answer: the text after '####', without commas."""
    return answer.split("####")[-1].strip().replace(",", "")


def reference(answer: str) -> str:
    """The reference solution for the scoring sequence: the steps without the calculator notes, then '#### <gold>'."""
    steps = re.sub(r"<<[^>]*>>", "", answer.split("####")[0]).strip()
    return f"{steps}\n#### {gold_number(answer)}"


def build(data_dir: Path) -> dict:
    import pyarrow.parquet as pq
    tok = tokenizer(data_dir / "tokenizer")
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eot = tok.convert_tokens_to_ids("<turn|>")
    if not isinstance(eot, int) or eot < 0 or eot == tok.unk_token_id:
        sys.exit("no end-of-turn token <turn|> in the tokenizer")

    raw = fetch(GSM_URL, data_dir / "gsm8k_test.parquet")
    table = pq.read_table(data_dir / "gsm8k_test.parquet").to_pylist()
    picks = random.Random(GSM_SEED).sample(range(len(table)), GSM_N)
    gsm = []
    for n, i in enumerate(picks):
        q, a = table[i]["question"], table[i]["answer"]
        ids = chat_ids(tok, GSM_ASK + q)
        ref = encode(reference(a))
        score_ids = ids + ref + [eot]
        gsm.append({"id": f"G{n + 1:03d}", "row": i, "gold": gold_number(a), "ids": ids, "ids_sha256": ids_sha(ids),
                    "score_ids": score_ids, "answer_start": len(ids), "score_sha256": ids_sha(score_ids)})

    books = sorted(p.name for p in (E55 / "data" / "pg19").glob("*.txt"))
    random.Random(NEEDLE_SEED).shuffle(books)
    rng = random.Random(NEEDLE_SEED + 1)
    words = list(WORDS)
    rng.shuffle(words)
    needles, used = [], []
    for name in books:
        if len(used) == NEEDLE_BOOKS:
            break
        text = (E55 / "data" / "pg19" / name).read_text(encoding="utf-8", errors="replace")
        toks = encode(text)
        if len(toks) < HAY_SKIP + HAY_TOKENS:
            continue
        hay = tok.decode(toks[HAY_SKIP:HAY_SKIP + HAY_TOKENS])
        used.append(name)
        for depth in NEEDLE_DEPTHS:
            k = len(needles)
            phrase = f"{words[k]}-{rng.randrange(1000, 10000)}"
            cut = hay.find("\n", int(len(hay) * depth))
            cut = int(len(hay) * depth) if cut < 0 else cut
            doc = hay[:cut] + f"\nThe secret passphrase for the archive is {phrase}.\n" + hay[cut:]
            ids = chat_ids(tok, NEEDLE_ASK + doc + NEEDLE_Q)
            needles.append({"id": f"N{k + 1:02d}", "book": name, "depth": depth, "gold": phrase, "ids": ids,
                            "ids_sha256": ids_sha(ids)})
    if len(used) < NEEDLE_BOOKS:
        sys.exit(f"needle books: only {len(used)} with enough tokens")

    e55 = json.loads((E55 / "data_e55.json").read_text())
    health = []
    for d in e55["docs"]:
        if d["id"] in HEALTH_DOCS:
            ids = d["ids"][:HEALTH_TOKENS]
            health.append({"id": f"H{d['id']}", "source": d["id"], "domain": d["domain"], "ids": ids,
                           "ids_sha256": ids_sha(ids)})
    health.sort(key=lambda h: HEALTH_DOCS.index(h["source"]))

    import transformers
    return {"experiment": "E-56", "model_id": MODEL_ID, "revision": REVISION,
            "transformers": transformers.__version__, "eot_id": eot, "bos_id": tok.bos_token_id,
            "tokenizer_sha256": sha((data_dir / "tokenizer" / "tokenizer.json").read_bytes()),
            "template_sha256": sha((data_dir / "tokenizer" / "chat_template.jinja").read_bytes()),
            "gsm8k_parquet_sha256": sha(raw), "gsm_ask": GSM_ASK, "needle_ask": NEEDLE_ASK, "needle_q": NEEDLE_Q,
            "seeds": {"gsm8k": GSM_SEED, "needle": NEEDLE_SEED}, "needle_books": used,
            "tok_check": [{"text": t, "ids": encode(t)} for t in TOK_CHECK],
            "gsm8k": gsm, "needle": needles, "health": health}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "data_e56.json")
    args = ap.parse_args()
    data = build(HERE / "data")
    args.out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    lens = [len(x["ids"]) for x in data["needle"]]
    print(f"{args.out}: {len(data['gsm8k'])} gsm8k, {len(data['needle'])} needle ({min(lens)} to {max(lens)} tokens), "
          f"{len(data['health'])} health; sha256 {sha(args.out.read_bytes())}")


if __name__ == "__main__":
    main()
