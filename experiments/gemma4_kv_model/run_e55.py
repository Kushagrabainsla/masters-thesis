"""E-55 run (README Sections 4 and 9): the model-level test of the compact K=V cache in the Gemma 4 31B global layers.
One vLLM engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, as E-53 and E-54), hooks on the 10 global-layer
Attention modules (`splice_e55.Splice`), then for each arm the teacher-forced log-probability of every token of the 12
documents (prompt_logprobs 1: the token's log-probability, its rank and the top-1 token) and, for 4 arms, greedy
generation on 16 chat prompts.

Phases: `main` (engine 1, 4,096 batched tokens): BF16, the determinism rerun (DET: 2 documents with the hooks
removed), BF16C, E5M2, STOCK8, TIED8, then the reported steps JOINT8, GEN and TIED8_UP while the time allows.
`noise` (engine 2, 2,048 batched tokens): BF16_B, the BF16 reference under another chunk size (reported).
A reported step runs only if its estimate (the mean time of the arms so far, times its share of tokens) ends at least
`--reserve` seconds before `--deadline`. Each arm writes `arrays/<arm>.npz`; `run_<phase>.json` holds the records.
--stub replaces the engine with synthetic log-probabilities (the Mac tests; no GPU, no vLLM).

Usage: run_e55.py --phase main|noise --data data_e55.json --scales scales_e54.json --out <res> --snapshot <dir>
                  --model-id <id> --revision <sha> --start <epoch> --deadline <epoch> [--rehearsal] [--stub ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import splice_e55 as sp  # noqa: E402

DECISION_ARMS = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8")
REPORTED = ("JOINT8", "GEN", "TIED8_UP")
GEN_ARMS = ("BF16", "BF16C", "STOCK8", "TIED8")
DET_DOCS = ("P1", "C1")                      # the determinism rerun (V3)
GLOBAL_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
HEADS = (32, 4, 512)                         # query heads, KV heads, head size of the 31B global layers
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_model_len": 65_664,
          "max_num_batched_tokens": 4_096, "max_num_seqs": 8, "gpu_memory_utilization": 0.70,
          "enable_prefix_caching": False, "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}
NOISE_TOKENS = 2_048                         # engine 2: another chunk size (the engine noise floor)
GEN_MAX_TOKENS = 128


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def parse_prompt_logprobs(pl, ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The FlatLogprobs of one prompt (vLLM 3966924d5, logprobs.py 178 to 209: position 0 is empty, and each later
    position holds [the prompt token with its rank, the top-1 token with rank 1]) -> the token log-probabilities
    (float32 [L-1]), the top-1 ids (int32 [L-1]) and the token ranks (int32 [L-1]). Raises on any other shape."""
    st, en = np.asarray(pl.start_indices), np.asarray(pl.end_indices)
    n = len(ids)
    if len(st) != n or len(en) != n or en[0] != st[0] or not np.all(en[1:] - st[1:] == 2):
        raise ValueError(f"prompt logprobs: {len(st)} positions for {n} tokens, or not 2 entries per position")
    tok = np.asarray(pl.token_ids, dtype=np.int64)[st[1]:].reshape(-1, 2)
    lps = np.asarray(pl.logprobs, dtype=np.float64)[st[1]:].reshape(-1, 2)
    rk = np.asarray([-1 if r is None else r for r in pl.ranks[st[1]:]], dtype=np.int64).reshape(-1, 2)
    if not np.array_equal(tok[:, 0], np.asarray(ids[1:], dtype=np.int64)):
        raise ValueError("prompt logprobs: the first entry is not the prompt token")
    if not np.all(rk[:, 1] == 1):
        raise ValueError("prompt logprobs: the second entry is not the top-1 token")
    return lps[:, 0].astype(np.float32), tok[:, 1].astype(np.int32), rk[:, 0].astype(np.int32)


class Stub:
    """Synthetic log-probabilities for the Mac tests: a fixed base per document, an arm effect (nats per token) and
    flips of the top-1 token at a per-arm rate. Effects and rates can be planted with --stub-effect ARM=x."""

    EFFECT = {"BF16": 0.0, "BF16_B": 0.0, "BF16C": 0.0, "E5M2": 0.004, "STOCK8": 0.001, "TIED8": 0.0006,
              "JOINT8": 0.0005, "TIED8_UP": 0.0015}
    FLIP = {"BF16": 0.0, "BF16_B": 0.002, "BF16C": 0.002, "E5M2": 0.03, "STOCK8": 0.01, "TIED8": 0.008,
            "JOINT8": 0.007, "TIED8_UP": 0.012}

    def __init__(self, effects: dict[str, float], seconds: float):
        self.effect = dict(self.EFFECT, **effects)
        self.seconds = seconds

    def score(self, arm: str, doc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = doc["length"] - 1
        base_rng = np.random.default_rng(int(doc["ids_sha256"][:8], 16))
        base = -np.abs(base_rng.normal(1.5, 1.0, n)).astype(np.float64)
        top1 = np.asarray(doc["ids"][1:], dtype=np.int64) + (base_rng.random(n) < 0.4)
        rng = np.random.default_rng(int(hashlib.sha256((arm + doc["id"]).encode()).hexdigest()[:8], 16))
        lp = base - self.effect.get(arm, 0.0) + (0.0 if arm in ("BF16", "DET") else rng.normal(0, 1e-4, n))
        flip = rng.random(n) < self.FLIP.get(arm, 0.0)
        top1 = np.where(flip, top1 + 7, top1)
        rank = np.where(top1 == np.asarray(doc["ids"][1:]), 1, 2)
        time.sleep(self.seconds)
        return lp.astype(np.float32), top1.astype(np.int32), rank.astype(np.int32)

    def generate(self, arm: str, gen: list[dict]) -> list[list[int]]:
        out = []
        for g in gen:
            toks = [(g["ids"][-1] * 31 + i) % 1000 for i in range(GEN_MAX_TOKENS)]
            if arm != "BF16" and int(hashlib.sha256((arm + g["id"]).encode()).hexdigest()[:2], 16) < 64:
                toks[GEN_MAX_TOKENS // 2:] = [t + 1 for t in toks[GEN_MAX_TOKENS // 2:]]
            out.append(toks)
        return out


def find_global(model) -> list[tuple[int, object, object]]:
    """The Attention modules of the global layers (the parent is a Gemma4Attention with is_sliding False and no KV
    sharing; as E-53 `register`), with their parents."""
    mods = dict(model.named_modules())
    found = []
    for name, m in mods.items():
        hit = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
        if not hit or type(m).__name__ != "Attention":
            continue
        parent = mods.get(name.rsplit(".", 1)[0])
        if parent is None or type(parent).__name__ != "Gemma4Attention":
            continue
        if getattr(parent, "is_sliding", True) or getattr(parent, "is_kv_shared_layer", False):
            continue
        found.append((int(hit.group(1)), m, parent))
    return sorted(found, key=lambda x: x[0])


def register(model, splice: sp.Splice, w_file: dict[int, float] | None) -> dict:
    """Hook the global layers. Returns the hooked layers, their shapes, the k_norm weight value on the unrotated dims
    (the w of the compact row) and the problems of V3."""
    import torch
    rec, problems = [], []
    for layer, m, parent in find_global(model):
        head = m.head_size
        _, unrot = sp.rot_dims(head)
        kn = getattr(parent, "k_norm", None)
        wt = getattr(kn, "weight", None)
        if not isinstance(wt, torch.Tensor):
            problems.append(f"layer {layer}: no k_norm weight")
            continue
        wu = wt.detach().float().cpu()[unrot]
        w = float(wu[0])
        if int(torch.unique(wu).numel()) != 1:
            problems.append(f"layer {layer}: k_norm has more than one value on the unrotated dims")
        if w_file is not None and layer in w_file and w != w_file[layer]:
            problems.append(f"layer {layer}: k_norm w {w} is not the scales file w {w_file[layer]}")
        splice.w[layer] = w
        splice.heads[layer] = (m.num_heads, m.num_kv_heads, head)
        splice.attach(layer, m)
        rec.append({"layer": layer, "num_heads": m.num_heads, "num_kv_heads": m.num_kv_heads, "head_size": head,
                    "w": w, "impl": type(getattr(m, "impl", None)).__name__,
                    "impl_scale": float(getattr(getattr(m, "impl", None), "scale", float("nan")))})
    return {"hooked": rec, "problems": problems}


def engine_record(llm) -> dict:
    rec = {}
    for name, get in (("vllm", lambda: __import__("vllm").__version__),
                      ("max_model_len", lambda: llm.llm_engine.model_config.max_model_len),
                      ("kv_cache_dtype", lambda: llm.llm_engine.vllm_config.cache_config.cache_dtype),
                      ("prefix_caching", lambda: llm.llm_engine.vllm_config.cache_config.enable_prefix_caching),
                      ("max_num_batched_tokens",
                       lambda: llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens),
                      ("compilation_mode", lambda: str(llm.llm_engine.vllm_config.compilation_config.mode)),
                      ("cudagraph_mode", lambda: str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)),
                      ("multiprocessing_env", lambda: os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))):
        try:
            rec[name] = get()
        except Exception as exc:  # recorded; V2 reads what exists
            rec[name] = f"unavailable: {type(exc).__name__}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("main", "noise"), required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data_e55.json")
    ap.add_argument("--scales", type=Path, default=HERE.parent / "gemma4_kv_scaletie" / "scales_e54.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--model-id", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    ap.add_argument("--revision", default="d4ab4f579dd3516f97d8a6a4c98d0653480bad15")
    ap.add_argument("--start", type=float, default=time.time())
    ap.add_argument("--deadline", type=float, default=time.time() + 2_400)
    ap.add_argument("--reserve", type=float, default=420.0, help="seconds kept free before the deadline")
    ap.add_argument("--rehearsal", action="store_true", help="a small Gemma 4 model: no w check, short documents")
    ap.add_argument("--rehearsal-tokens", type=int, default=16_384)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-effect", action="append", default=[], help="ARM=nats, for the stub")
    ap.add_argument("--stub-seconds", type=float, default=0.0)
    args = ap.parse_args()

    out, arrays = args.out, args.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    docs, gen = data["docs"], data["gen"]
    if args.rehearsal:
        docs = [dict(d, ids=d["ids"][:args.rehearsal_tokens], length=min(d["length"], args.rehearsal_tokens),
                     ids_sha256=ids_sha(d["ids"][:args.rehearsal_tokens])) for d in docs]
    scales_doc = json.loads(args.scales.read_text())
    scales = {int(k): v for k, v in scales_doc["layers"].items()}
    w_file = {k: v["w"] for k, v in scales.items()}
    rec = {"phase": args.phase, "started_at": now(), "status": "running", "stub": args.stub,
           "rehearsal": args.rehearsal, "data_sha256": sha_file(args.data), "scales_sha256": sha_file(args.scales),
           "model_id": args.model_id, "revision": args.revision, "docs": [d["id"] for d in docs],
           "doc_lengths": {d["id"]: d["length"] for d in docs}, "arms": {}, "cut": [], "errors": []}
    run_path = out / f"run_{args.phase}.json"
    save = lambda: run_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")  # noqa: E731
    total_tokens = sum(d["length"] for d in docs)
    arm_seconds: list[float] = []

    def fits(name: str, share: float) -> bool:
        est = (sum(arm_seconds) / len(arm_seconds) if arm_seconds else 0.0) * share
        ok = time.time() + est <= args.deadline - args.reserve
        if not ok:
            rec["cut"].append({"step": name, "estimate_s": round(est, 1), "at": now(),
                               "elapsed_s": round(time.time() - args.start, 1)})
            save()
        return ok

    try:
        if args.stub:
            stub = Stub({k: float(v) for k, v in (e.split("=") for e in args.stub_effect)}, args.stub_seconds)
            splice = sp.Splice({}, scales, {})
            rec["hooks"] = {"hooked": [{"layer": layer, "num_heads": HEADS[0], "num_kv_heads": HEADS[1],
                                        "head_size": HEADS[2], "w": w_file.get(layer), "impl": "stub",
                                        "impl_scale": 1.0} for layer in GLOBAL_LAYERS], "problems": []}
            rec["engine"] = {"vllm": "STUB", "max_model_len": ENGINE["max_model_len"], "kv_cache_dtype": "auto",
                             "prefix_caching": False, "compilation_mode": "CompilationMode.NONE",
                             "cudagraph_mode": "CUDAGraphMode.NONE",
                             "max_num_batched_tokens": ENGINE["max_num_batched_tokens"] if args.phase == "main"
                             else NOISE_TOKENS}
            rec["tok_check"] = {"ok": True, "stub": True}
            rec["gpu"] = "STUB"

            def score(arm, doc):
                for layer in GLOBAL_LAYERS:
                    splice.counts[(arm, doc["id"], layer)] = doc["length"]
                splice.checks.setdefault(splice.arm, [])
                splice.gaps.setdefault(splice.arm, {layer: 0.002 for layer in GLOBAL_LAYERS})
                return stub.score(arm, doc)

            def generate(arm):
                return stub.generate(arm, gen)
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            import torch
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            torch.backends.cuda.matmul.allow_tf32 = False
            eng = dict(ENGINE)
            if args.phase == "noise":
                eng["max_num_batched_tokens"] = NOISE_TOKENS
            t0 = time.time()
            llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                      tokenizer=args.snapshot or args.model_id, **eng)
            rec["engine_seconds"] = round(time.time() - t0, 1)
            rec["engine_kwargs"] = eng
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["engine"] = engine_record(llm)
            tok = llm.get_tokenizer()
            got = [tok.encode(c["text"], add_special_tokens=False) for c in data["tok_check"]]
            rec["tok_check"] = {"ok": got == [c["ids"] for c in data["tok_check"]], "got": got}
            splice = sp.Splice({}, None if args.rehearsal else scales, {})
            rec["hooks"] = llm.apply_model(lambda model: register(model, splice,
                                                                  None if args.rehearsal else w_file))[0]
            save()
            if rec["hooks"]["problems"] and not args.rehearsal:
                raise RuntimeError("V3 hook problems: " + "; ".join(rec["hooks"]["problems"][:5]))
            sp_nll = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False,
                                    flat_logprobs=True)
            sp_gen = SamplingParams(max_tokens=GEN_MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)

            def score(arm, doc):
                o = llm.generate([TokensPrompt(prompt_token_ids=doc["ids"])], sp_nll, use_tqdm=False)[0]
                return parse_prompt_logprobs(o.prompt_logprobs, doc["ids"])

            def generate(arm):
                outs = llm.generate([TokensPrompt(prompt_token_ids=g["ids"]) for g in gen], sp_gen, use_tqdm=False)
                return [list(o.outputs[0].token_ids) for o in outs]

            warm = docs[0]["ids"][:512]
            llm.generate([TokensPrompt(prompt_token_ids=warm)], sp_nll, use_tqdm=False)   # warm-up, not scored

        def run_arm(arm: str, subset: list[dict] | None = None, name: str | None = None) -> None:
            name = name or arm
            splice.set_arm(arm if arm in sp.ARMS else "BF16")
            t0 = time.time()
            res, finite = {}, True
            for d in subset or docs:
                splice.new_doc(d["id"] if name != "DET" else f"DET:{d['id']}")
                lp, top1, rank = score(name if args.stub else arm, d)
                finite &= bool(np.isfinite(lp).all())
                res[f"{d['id']}_lp"], res[f"{d['id']}_top1"], res[f"{d['id']}_rank"] = lp, top1, rank
            np.savez_compressed(arrays / f"{name}.npz", **res)
            secs = time.time() - t0
            rec["arms"][name] = {"seconds": round(secs, 1), "docs": [d["id"] for d in subset or docs],
                                 "finite": finite, "finished_at": now()}
            if subset is None:
                arm_seconds.append(secs)
            save()

        if args.phase == "noise":
            run_arm("BF16", name="BF16_B")
        else:
            run_arm("BF16")
            det = [d for d in docs if d["id"] in DET_DOCS]
            if args.stub:
                run_arm("BF16", subset=det, name="DET")
            else:
                saved = splice.detach()
                run_arm("BF16", subset=det, name="DET")
                splice.reattach(saved)
            for arm in DECISION_ARMS[1:]:
                run_arm(arm)
            if fits("JOINT8", 1.0):
                run_arm("JOINT8")
            gen_share = len(GEN_ARMS) * sum(len(g["ids"]) + GEN_MAX_TOKENS * 8 for g in gen) / total_tokens
            if fits("GEN", gen_share):
                t0 = time.time()
                outs = {}
                for arm in GEN_ARMS:
                    splice.set_arm(arm)
                    splice.new_doc("GEN")
                    outs[arm] = generate(arm)
                (out / "gen_outputs.json").write_text(json.dumps({"ids": [g["id"] for g in gen], "outputs": outs}))
                rec["arms"]["GEN"] = {"seconds": round(time.time() - t0, 1), "arms": list(GEN_ARMS),
                                      "finished_at": now()}
                save()
            if fits("TIED8_UP", 1.0):
                run_arm("TIED8_UP")
        splice.set_arm("BF16")
        rec["checks"] = splice.checks
        rec["counts"] = {f"{a}|{d}|{layer}": n for (a, d, layer), n in splice.counts.items()}
        rec["hook_errors"] = splice.errors
        rec["gaps"] = {a: {str(k): v for k, v in g.items()} for a, g in splice.gaps.items()}
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "error"
        rec["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["finished_at"] = now()
    save()
    print(json.dumps({"phase": args.phase, "status": rec["status"], "arms": list(rec["arms"]), "cut": rec["cut"],
                      "errors": rec["errors"][:2]}))
    sys.exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
