"""E-55 data (README Section 4.3): the 12 scored documents and the 16 generation prompts, as token ids of the pinned
Gemma 4 31B tokenizer, frozen in `data_e55.json` before any E-55 run. Runs on the Mac (no GPU).

Domains (4 documents each; the seeds fix the order):
- books: PG-19 test books (the official list of deepmind/pg19; the texts from its public bucket). BOS, then the first
  65,535 tokens of a book: 65,536 tokens. The first 4 books in the seed-55 order with enough tokens.
- prose: our own Markdown files in `dossier/` (written 2026-09 to 2026-11, after the Gemma 4 release), read from git at
  one commit. BOS, then the first 6,000 tokens of each file in the seed-56 order, each followed by a blank line, cut to
  32,768 tokens.
- code: our own Python files under `experiments/` (the same commit; not this package), as prose with seed 57.
No source is used twice. The generation prompts: 8 more books and 8 more dossier files; a user turn "Read the text
below. Then summarize it in five sentences." with the first 3,000 tokens of the source, rendered by the tokenizer's
chat template with the generation prompt.

Usage: prepare_e55.py [--commit <git ref>] [--out data_e55.json]   (downloads go to data/, which git ignores)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_ID = "RedHatAI/gemma-4-31B-it-FP8-dynamic"
REVISION = "d4ab4f579dd3516f97d8a6a4c98d0653480bad15"      # E-50 to E-54
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
PG19_LIST = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/test_files.txt"
PG19_BUCKET = "https://storage.googleapis.com/deepmind-gutenberg/"
SEEDS = {"books": 55, "prose": 56, "code": 57}
PLAN = {"books": (4, 65_536), "prose": (4, 32_768), "code": (4, 32_768)}   # documents and tokens per domain
FILE_TOKENS = 6_000                  # at most this many tokens of one file (prose and code), so a document has 6 files
GEN_N, GEN_TOKENS = 8, 3_000         # generation prompts per source family, and the excerpt length in tokens
GEN_ASK = "Read the text below. Then summarize it in five sentences.\n\n"
TOK_CHECK = ("Hello world.", "def f(x):\n    return x * 2\n", "Grüße, naïve café: 3 × 4 = 12.")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def fetch(url: str, path: Path) -> bytes:
    """The file at url, kept in path (a second call reads the copy)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def git_text(commit: str, rel: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, text=True,
                          check=True).stdout


def git_files(commit: str, prefix: str, suffix: str, exclude: str | None = None) -> list[str]:
    """The files under prefix (at any depth) that end with suffix, at the commit."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", commit], capture_output=True,
                         text=True, check=True).stdout.split()
    return sorted(f for f in out if f.startswith(prefix) and f.endswith(suffix)
                  and not (exclude and f.startswith(exclude)))


def tokenizer(tok_dir: Path):
    """The pinned tokenizer (transformers, as the Pod image), from local copies of its files."""
    from transformers import AutoTokenizer
    for name in TOK_FILES:
        fetch(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", tok_dir / name)
    return AutoTokenizer.from_pretrained(str(tok_dir))


def special_ids(tok) -> set[int]:
    return set(tok.all_special_ids)


def concat_doc(sources: list[tuple[str, str]], encode, bos: int, length: int, specials: set[int],
               used: set[str]) -> tuple[list[int], list[dict]]:
    """BOS, then up to FILE_TOKENS tokens of each source (a source with a special id is skipped), each followed by
    the ids of a blank line, cut to length. Returns the ids and the sources it used."""
    sep = encode("\n\n")
    ids, taken = [bos], []
    for name, text in sources:
        if len(ids) >= length:
            break
        if name in used:
            continue
        part = encode(text)[:FILE_TOKENS]
        if not part or specials & set(part):
            continue
        used.add(name)
        taken.append({"name": name, "sha256": sha(text.encode()), "tokens": len(part)})
        ids += part + sep
    return ids[:length], taken


def build(commit: str, data_dir: Path) -> dict:
    tok = tokenizer(data_dir / "tokenizer")
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    bos, specials = tok.bos_token_id, special_ids(tok)
    used: set[str] = set()
    docs, gen = [], []

    # books
    names = fetch(PG19_LIST, data_dir / "pg19_test_files.txt").decode().split()
    random.Random(SEEDS["books"]).shuffle(names)
    n_books, length = PLAN["books"]
    rest_books = []
    for name in names:
        text = fetch(PG19_BUCKET + name, data_dir / "pg19" / Path(name).name).decode("utf-8")
        if len([d for d in docs if d["domain"] == "books"]) < n_books:
            ids = encode(text)
            if len(ids) >= length - 1 and not specials & set(ids[:length - 1]):
                k = len([d for d in docs if d["domain"] == "books"]) + 1
                doc = [bos] + ids[:length - 1]
                used.add(name)
                docs.append({"id": f"B{k}", "domain": "books", "length": length, "ids_sha256": ids_sha(doc),
                             "sources": [{"name": f"pg19 {name}", "sha256": sha(text.encode()), "tokens": length - 1}],
                             "ids": doc})
                continue
        rest_books.append((name, text))
        if len([d for d in docs if d["domain"] == "books"]) == n_books and len(rest_books) >= GEN_N:
            break

    # prose and code (our own files, read from git at one commit)
    families = {"prose": git_files(commit, "dossier/", ".md"),
                "code": git_files(commit, "experiments/", ".py", exclude="experiments/gemma4_kv_model/")}
    rest_prose = []
    for domain, files in families.items():
        random.Random(SEEDS[domain]).shuffle(files)
        sources = [(f, git_text(commit, f)) for f in files]
        n_docs, length = PLAN[domain]
        for k in range(1, n_docs + 1):
            ids, taken = concat_doc(sources, encode, bos, length, specials, used)
            if len(ids) != length:
                sys.exit(f"{domain}: not enough files for document {k}")
            docs.append({"id": f"{domain[0].upper()}{k}", "domain": domain, "length": length,
                         "ids_sha256": ids_sha(ids), "sources": taken, "ids": ids})
        if domain == "prose":
            rest_prose = [(n, t) for n, t in sources if n not in used]

    # generation prompts: 8 more books, then 8 more dossier files
    for family, pool in (("books", rest_books), ("prose", rest_prose)):
        picked = 0
        for name, text in pool:
            if picked == GEN_N:
                break
            part = encode(text)[:GEN_TOKENS]
            if len(part) < GEN_TOKENS or specials & set(part) or name in used:
                continue
            used.add(name)
            msg = GEN_ASK + tok.decode(part)
            ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                          tokenize=True)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)   # transformers 5 returns a BatchEncoding
            picked += 1
            gen.append({"id": f"G{len(gen) + 1}", "family": family, "source": name, "sha256": sha(text.encode()),
                        "ids_sha256": ids_sha(ids), "ids": ids})
        if picked < GEN_N:
            sys.exit(f"generation prompts: only {picked} {family} sources")

    import transformers
    return {"experiment": "E-55", "model_id": MODEL_ID, "revision": REVISION, "source_commit": commit,
            "transformers": transformers.__version__, "bos_id": bos,
            "tokenizer_sha256": sha((data_dir / "tokenizer" / "tokenizer.json").read_bytes()),
            "seeds": SEEDS, "plan": {k: list(v) for k, v in PLAN.items()}, "file_tokens": FILE_TOKENS,
            "gen_ask": GEN_ASK, "gen_tokens": GEN_TOKENS,
            "tok_check": [{"text": t, "ids": encode(t)} for t in TOK_CHECK], "docs": docs, "gen": gen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", type=Path, default=HERE / "data_e55.json")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", args.commit], capture_output=True, text=True,
                            check=True).stdout.strip()
    data = build(commit, HERE / "data")
    args.out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out.read_bytes()), "source_commit": commit,
                      "docs": [(d["id"], d["length"], len(d["sources"])) for d in data["docs"]],
                      "gen": [(g["id"], len(g["ids"])) for g in data["gen"]]}))


if __name__ == "__main__":
    main()
