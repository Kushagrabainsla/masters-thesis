"""E-27 driver: noise-shaped V-cache quantization screen on Qwen3-0.6B (I-58).

Phases (each saves its results with a manifest of code and data hashes; a later phase refuses to
resume when the manifest differs):
  step1   calibration and evaluation measurement runs (full-precision forward, no propagation):
          gates, smoothness diagnostic G, per-window layer-local error energy (M1)
  ppl     next-token NLL with the emulated cache in all layers (M2)
  passkey passkey retrieval with a natural-text haystack and decoys (M3, report only)
Decisions are made by e27_decide.py. Plan, amendment A1 and criteria: RESEARCH_JOURNAL.md, E-27.
"""

import argparse
import hashlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

import vcache as vc

MODEL = "Qwen/Qwen3-0.6B"
WIN = 4096
SEED = 27
N_CALIB, N_EVAL = 4, 6          # windows per corpus
MAX_OVERLAP = 0.05              # 16-gram overlap allowed between selected windows
GATE_RATIO = 0.9                # shape a KV head only if shaped/plain error energy < this

CONDS = {
    "C1u": dict(bits=2, rotated=False),
    "C1": dict(bits=2, rotated=True),
    "C1d": dict(bits=2, rotated=True, dither=True),
    "C2": dict(bits=2, rotated=True, shaping=True),
    "C3": dict(bits=2, rotated=True, shaping=True, gate="C3"),
    "C4": dict(bits=2, rotated=True, shaping=True, order="keys", gate="C4"),
    "C4r": dict(bits=2, rotated=True, shaping=True, order="random", gate="C3"),
    "C5": dict(bits=3, rotated=True),
    "C1z": dict(bits=2, rotated=True, sink=0, recent=0, self_quantized=True),
    "C3z": dict(bits=2, rotated=True, shaping=True, gate="C3", sink=0, recent=0, self_quantized=True),
    "C5z": dict(bits=3, rotated=True, sink=0, recent=0, self_quantized=True),
}
MEASURE_EVAL = ["C1u", "C1", "C1d", "C2", "C3", "C4", "C4r", "C5"]
CALIB_CONDS = {
    "C1": CONDS["C1"],
    "C2": CONDS["C2"],
    "C4all": dict(bits=2, rotated=True, shaping=True, order="keys"),
}
PPL_CONDS = ["C0", "C1", "C3", "C5", "C2", "C1u", "C1d", "C4", "C4r", "C1z", "C3z", "C5z"]
PASSKEY_CONDS = ["C0", "C1", "C3", "C5"]

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def manifest(texts_path, wins):
    import tokenizers
    import transformers
    from huggingface_hub import snapshot_download
    snap = os.path.basename(snapshot_download(MODEL, local_files_only=True))
    win_ids = json.dumps([[w["corpus"], w["ids"]] for part in ("calib", "eval", "haystack") for w in wins[part]])
    return {
        "vcache.py": sha256(os.path.join(HERE, "vcache.py")),
        "e27_run.py": sha256(os.path.join(HERE, "e27_run.py")),
        "texts": sha256(texts_path),
        "windows": hashlib.sha256(win_ids.encode()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "model_snapshot": snap,
    }


def dump(obj, path, **kw):
    """Atomic JSON write: a kill during the write cannot leave a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def log(msg, fh=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def load_model():
    AttentionInterface.register("vcache", vc.attention)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="vcache")
    model.eval()
    return tok, model


def shingles(ids, n=16):
    return {tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1)}


def windows(tok, texts_path):
    """Fixed-seed windows, N_CALIB + N_EVAL per corpus, with near-duplicates removed. Also returns
    unused prose and code windows as passkey haystacks."""
    texts = json.load(open(texts_path))
    rng = random.Random(SEED)
    out = {"calib": [], "eval": [], "haystack": []}
    taken = set()
    for corpus in ("prose", "code", "agent"):
        cands = []
        if corpus in ("prose", "code"):
            ids = tok(texts[corpus][0], add_special_tokens=False)["input_ids"]
            cands = [(i, ids[i:i + WIN]) for i in range(0, len(ids) - WIN, WIN)]
            rng.shuffle(cands)
        else:
            for k, t in enumerate(texts[corpus]):
                ids = tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) < WIN:
                    continue
                off = rng.randrange(0, len(ids) - WIN + 1)
                cands.append((k * 10 ** 7 + off, ids[off:off + WIN]))
        chosen, rest = [], []
        for off, c in cands:
            sh = shingles(c)
            if len(chosen) < N_CALIB + N_EVAL and len(sh & taken) / len(sh) < MAX_OVERLAP:
                chosen.append((off, c))
                taken |= sh
            else:
                rest.append(c)
        assert len(chosen) == N_CALIB + N_EVAL, (corpus, len(chosen))
        for i, (off, c) in enumerate(chosen):
            out["calib" if i < N_CALIB else "eval"].append({"corpus": corpus, "offset": off, "ids": c})
        if corpus in ("prose", "code"):
            out["haystack"] += [{"corpus": corpus, "ids": c} for c in rest[:15]]
    return out


def hidden(model, ids):
    with torch.no_grad():
        return model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]


def rec_to_json(rec):
    return {k: (v.tolist() if torch.is_tensor(v) else {n: x.tolist() for n, x in v.items()})
            for k, v in rec.items()}


def measure(model, wins, conds, fh):
    cfg = model.config
    recs = []
    vc.STATE.mode, vc.STATE.measure_conds = "measure", conds
    for i, w in enumerate(wins):
        vc.STATE.record = vc.new_record(cfg.num_hidden_layers, cfg.num_attention_heads, list(conds))
        t0 = time.time()
        hidden(model, w["ids"])
        recs.append(vc.STATE.record)
        log(f"  measure window {i} ({w['corpus']}) {time.time() - t0:.0f}s", fh)
    vc.STATE.mode = "fp"
    return recs


def kv_sum(x, cfg):
    rep = cfg.num_attention_heads // cfg.num_key_value_heads
    return x.view(cfg.num_hidden_layers, cfg.num_key_value_heads, rep).sum(-1)


def gates_from(recs, cfg):
    e = lambda name: kv_sum(sum(r["err_energy"][name] for r in recs), cfg)
    plain = e("C1")
    return {"C3": e("C2") < GATE_RATIO * plain, "C4": e("C4all") < GATE_RATIO * plain}


def step1(model, wins, out_dir, man, fh):
    cfg = model.config
    log("step1: calibration measurement", fh)
    rc = measure(model, wins["calib"], CALIB_CONDS, fh)
    vc.STATE.gates = gates_from(rc, cfg)
    # Split-half agreement: first two vs last two calibration windows of each corpus.
    corp = [w["corpus"] for w in wins["calib"]]
    half_a = [r for i, r in enumerate(rc) if corp[:i].count(corp[i]) < 2]
    half_b = [r for i, r in enumerate(rc) if corp[:i].count(corp[i]) >= 2]
    ga, gb = gates_from(half_a, cfg), gates_from(half_b, cfg)
    agree = {k: float((ga[k] == gb[k]).float().mean()) for k in ga}
    log(f"step1: gates {[(k, float(v.float().mean())) for k, v in vc.STATE.gates.items()]} "
        f"split-half agreement {agree}", fh)
    log("step1: evaluation measurement", fh)
    re = measure(model, wins["eval"], {n: CONDS[n] for n in MEASURE_EVAL}, fh)

    # Smoothness diagnostic (amendment A1: not a kill rule), per query head and per KV head.
    g_num = sum(r["g_num"] for r in rc + re)
    g_den = sum(r["g_den"] for r in rc + re)
    diag = {}
    for level, num, den in (("q_head", g_num, g_den), ("kv_head", kv_sum(g_num, cfg), kv_sum(g_den, cfg))):
        G = (num / den.clamp_min(1e-300)).sqrt()
        diag[level] = {
            "S_0.5": float(den[G < 0.5].sum() / den.sum()),
            "predicted_ideal_gating_energy_ratio": float((den * torch.clamp(G ** 2, max=1.0)).sum() / den.sum()),
            "G": G.tolist(),
        }
    res = {
        "manifest": man,
        "gates": {k: v.tolist() for k, v in vc.STATE.gates.items()},
        "gate_fraction": {k: float(v.float().mean()) for k, v in vc.STATE.gates.items()},
        "gate_split_half_agreement": agree,
        "smoothness_diagnostic": diag,
        "eval_windows": [{"corpus": w["corpus"], "offset": w["offset"]} for w in wins["eval"]],
        "calib_windows": [{"corpus": w["corpus"], "offset": w["offset"]} for w in wins["calib"]],
        "eval_records": [rec_to_json(r) for r in re],
    }
    dump(res, os.path.join(out_dir, "e27_step1.json"))
    tot = {n: sum(float(r["err_energy"][n].sum()) for r in re) for n in MEASURE_EVAL}
    log("step1: total M1 error energy relative to C1: " +
        ", ".join(f"{n}={tot[n] / tot['C1']:.3f}" for n in MEASURE_EVAL), fh)
    return res


def check_manifest(path, man, gates_json):
    if not os.path.exists(path):
        return {}
    res = json.load(open(path))
    if res.get("manifest") != man or res.get("gates") != gates_json:
        raise SystemExit(f"{path}: manifest or gates differ from the current run. Move the file away to rerun.")
    return res


def ppl(model, wins, out_dir, man, gates_json, fh):
    path = os.path.join(out_dir, "e27_ppl.json")
    res = check_manifest(path, man, gates_json) or {"manifest": man, "gates": gates_json, "nll": {}}
    for name in PPL_CONDS:
        if name in res["nll"]:
            continue
        if name == "C0":
            vc.STATE.mode = "fp"
        else:
            vc.STATE.mode, vc.STATE.cond = "quant", CONDS[name]
        nll = []
        for w in wins["eval"]:
            t0 = time.time()
            h = hidden(model, w["ids"])
            tgt = torch.tensor(w["ids"][512:])
            total = 0.0
            with torch.no_grad():
                for s in range(511, WIN - 1, 512):
                    e = min(s + 512, WIN - 1)
                    logits = model.lm_head(h[s:e])
                    total += float(F.cross_entropy(logits, tgt[s - 511:e - 511], reduction="sum"))
            nll.append(total / len(tgt))
            log(f"  ppl {name} {w['corpus']} nll={nll[-1]:.4f} {time.time() - t0:.0f}s", fh)
        res["nll"][name] = nll
        res["corpus"] = [w["corpus"] for w in wins["eval"]]
        dump(res, path, indent=1)
    vc.STATE.mode = "fp"
    return res


HEAD = ("There is an important piece of information hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about the important information there.\n")


def passkey_prompts(tok, haystacks, n=30, hay_tokens=3150):
    """Natural-text haystack (unused prose and code windows), the pass key once, two decoys."""
    rng = random.Random(SEED + 1)
    prompts = []
    for i in range(n):
        hay = haystacks[i % len(haystacks)]["ids"][:hay_tokens]
        lines = tok.decode(hay).split("\n")
        key, d1, d2 = (str(rng.randrange(10000, 100000)) for _ in range(3))
        assert len({key, d1, d2}) == 3
        depth = rng.uniform(0.1, 0.9)
        inserts = [(depth, f"The pass key is {key}. Remember it. {key} is the pass key."),
                   (rng.uniform(0.05, 0.95), f"The backup code is {d1}."),
                   (rng.uniform(0.05, 0.95), f"The room number is {d2}.")]
        for frac, line in sorted(inserts, key=lambda x: -x[0]):
            lines.insert(int(frac * len(lines)), line)
        text = HEAD + "\n".join(lines) + "\nWhat is the pass key? The pass key is"
        prompts.append({"prompt": tok(text, add_special_tokens=False)["input_ids"],
                        "answer": tok(" " + key, add_special_tokens=False)["input_ids"],
                        "depth": depth})
    return prompts


def passkey(model, tok, wins, out_dir, man, gates_json, fh, n=30):
    path = os.path.join(out_dir, "e27_passkey.json")
    res = check_manifest(path, man, gates_json) or {"manifest": man, "gates": gates_json, "correct": {}}
    prompts = passkey_prompts(tok, wins["haystack"], n)
    res["n_tokens"] = [len(p["prompt"]) for p in prompts]
    for name in PASSKEY_CONDS:
        if name in res["correct"]:
            continue
        if name == "C0":
            vc.STATE.mode = "fp"
        else:
            vc.STATE.mode, vc.STATE.cond = "quant", CONDS[name]
        ok = []
        for pr in prompts:
            h = hidden(model, pr["prompt"] + pr["answer"])
            L = len(pr["prompt"])
            with torch.no_grad():
                pred = model.lm_head(h[L - 1:L - 1 + len(pr["answer"])]).argmax(-1).tolist()
            ok.append(pred == pr["answer"])
        res["correct"][name] = ok
        log(f"  passkey {name} accuracy={sum(ok) / len(ok):.3f}", fh)
        dump(res, path, indent=1)
    vc.STATE.mode = "fp"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["step1", "ppl", "passkey", "all"], default="all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(SEED)
    fh = open(os.path.join(args.out, "e27_log.txt"), "a")
    tok, model = load_model()
    wins = windows(tok, args.texts)
    man = manifest(args.texts, wins)
    log(f"manifest {man}", fh)
    log(f"windows: calib={len(wins['calib'])} eval={len(wins['eval'])} haystack={len(wins['haystack'])}", fh)
    s1_path = os.path.join(args.out, "e27_step1.json")
    if os.path.exists(s1_path):
        s1 = json.load(open(s1_path))
        if s1["manifest"] != man:
            raise SystemExit("e27_step1.json was made with different code or data. Move it away to rerun.")
        log("step1: reusing e27_step1.json (manifest matches)", fh)
    elif args.phase in ("step1", "all"):
        s1 = step1(model, wins, args.out, man, fh)
    else:
        raise SystemExit("run step1 first")
    vc.STATE.gates = {k: torch.tensor(v) for k, v in s1["gates"].items()}
    if args.phase in ("ppl", "all"):
        ppl(model, wins, args.out, man, s1["gates"], fh)
    if args.phase in ("passkey", "all"):
        passkey(model, tok, wins, args.out, man, s1["gates"], fh)


if __name__ == "__main__":
    main()
