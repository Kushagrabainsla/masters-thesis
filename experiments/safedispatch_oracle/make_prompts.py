"""Freeze the pilot prompts to prompts.json.

Prompts are prose paragraphs from stable snapshot documents in this repo (not the journal,
which keeps changing), chosen by a fixed seed. The output file is hashed into every manifest,
so the pilot stays reproducible even if the source documents are later edited.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCES = [
    REPO / "dossier" / "RESEARCH_GAP_DOSSIER_2026-09-16.md",
    REPO / "dossier" / "ADVERSARIAL_REVIEW_2026-09-23.md",
    REPO / "ideas" / "01-state-reuse-soundness.md",
    REPO / "ideas" / "02-collective-engine-arbitration.md",
]
SEED = 20260924
N_PROMPTS = 48


def paragraphs():
    out = []
    for f in SOURCES:
        text = re.sub(r"```.*?```", "", f.read_text(), flags=re.S)
        for p in re.split(r"\n\s*\n", text):
            p = " ".join(p.split())
            if 300 <= len(p) <= 900 and re.match(r"^[A-Z]", p) and "|" not in p and not p.startswith(("#", "-", "*")):
                out.append(p)
    return out


if __name__ == "__main__":
    paras = paragraphs()
    rng = random.Random(SEED)
    rng.shuffle(paras)
    chosen = paras[:N_PROMPTS]
    (ROOT / "prompts.json").write_text(json.dumps(chosen, indent=1))
    print(f"{len(paras)} candidate paragraphs, wrote {len(chosen)} to prompts.json")
