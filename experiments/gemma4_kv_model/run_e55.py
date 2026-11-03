"""E-55 run (README Sections 4 and 9, amendment A1): the model-level test of the compact K=V cache in the Gemma 4 31B
global layers. One vLLM engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, as E-53 and E-54), hooks on the 10
global-layer Attention modules (`splice_e55.Splice`), then for each arm the teacher-forced log-probability of every
token of the 12 documents (prompt_logprobs 1: the token's log-probability, its rank and the top-1 token) and, for 4
arms, greedy generation on 16 chat prompts.

Order: the warm-up (512 tokens, not scored); the pre-flight (every arm and DET on the first 4,096 tokens of the first
document: one chunk; the run stops at once with exit 3 if a check fails, R-77 F3); the decision arms BF16, DET (P1 and
C1 with the hooks removed), BF16C, E5M2, STOCK8 and TIED8; then the reported steps BF16R, JOINT8, GEN and TIED8_UP while
the time allows. The evidence of the hooks starts again after the warm-up and after the pre-flight (R-77 B1). A reported
step runs only if 1.5 times its estimate (the mean time of the arms so far, times its share of tokens) ends at least
`--reserve` seconds before `--deadline`; a failed reported step does not stop the run (R-77 F2). Each arm writes
`arrays/<arm>.npz`; `run_main.json` holds the record and the hook evidence, written at each step (R-77 F2). SIGTERM
(the session timeout) saves the record before the exit.

--stub replaces the engine with synthetic log-probabilities and 10 small Attention modules that carry the real hooks
and see every token in the engine order (the Mac tests; no GPU, no vLLM). --rehearsal (S5, a small Gemma 4 model)
cuts the documents and uses synthetic scales (R-77 B2).

Usage: run_e55.py --data data_e55.json --scales scales_e54.json --out <res> --snapshot <dir> --model-id <id>
                  --revision <sha> --start <epoch> --deadline <epoch> [--rehearsal] [--stub ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_e55 as an  # noqa: E402
import splice_e55 as sp  # noqa: E402

DECISION_ARMS = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8")
REPORTED = ("BF16R", "JOINT8", "GEN", "TIED8_UP")   # in this order while the time allows; cut from the end
GEN_ARMS = ("BF16", "BF16C", "STOCK8", "TIED8")
DET_DOCS = ("P1", "C1")                      # the determinism rerun (V3)
GLOBAL_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
HEADS = (32, 4, 512)                         # query heads, KV heads, head size of the 31B global layers
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_model_len": 65_664,
          "max_num_batched_tokens": 4_096, "max_num_seqs": 8, "gpu_memory_utilization": 0.70,
          "enable_prefix_caching": False, "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}
GEN_MAX_TOKENS = 128
WARM_TOKENS, PRE_TOKENS = 512, 4_096         # the warm-up prompt; the pre-flight prompt (one chunk)
STUB_HEADS, STUB_FIRST = (4, 2, 16), 256     # the stub Attention modules; the first chunk of each stub prompt


class PreflightError(RuntimeError):
    """The pre-flight found a problem: the run stops before the paid arms (exit 3)."""


class Terminated(BaseException):
    """SIGTERM from the session timeout: the record is saved before the exit."""


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


def synthetic_scales(w: dict[int, float]) -> dict[int, dict]:
    """The rehearsal scales (R-77 B2): values other than 1 test the descale route on a model without the E-54 scales."""
    return {layer: {"w": x, "tied": {"vs": 0.125, "ks": sp.f32(x * 0.125)}, "calstd_joint": {"ks": 0.05, "vs": 0.125}}
            for layer, x in w.items()}


class Stub:
    """Synthetic log-probabilities for the Mac tests: a fixed base per document, an arm effect (nats per token) and
    flips of the top-1 token at a per-arm rate. Effects and rates can be planted with --stub-effect ARM=x."""

    EFFECT = {"BF16": 0.0, "BF16C": 0.0, "BF16R": 0.0, "E5M2": 0.004, "STOCK8": 0.001, "TIED8": 0.0006,
              "JOINT8": 0.0005, "TIED8_UP": 0.0015}
    FLIP = {"BF16": 0.0, "BF16C": 0.002, "BF16R": 0.002, "E5M2": 0.03, "STOCK8": 0.01, "TIED8": 0.008,
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


class StubImpl:
    def __init__(self):
        self.scale = 1.0


class StubAttention(torch.nn.Module):
    """The stub engine's stand-in of a vLLM Attention module: [T, H x D] in, [T, Hq x D] out in BF16, impl.scale on
    the scores. A call of at most STUB_FIRST tokens (the first chunk of a prompt, the one that V5 and V7 read) gets
    causal GQA attention inside the call for its last GAP_ROWS rows; other rows are zeros. plant 'ignore_scale': a
    kernel that keeps the scale 1.0; plant 'ignore_hook': a kernel that uses the inputs from before the splice hook."""

    def __init__(self, plant: str = ""):
        super().__init__()
        self.impl, self.plant, self.orig = StubImpl(), plant, None
        self.num_heads, self.num_kv_heads, self.head_size = STUB_HEADS
        self.register_forward_pre_hook(self.keep)       # registered before the splice hooks: it sees the inputs

    def keep(self, module, args):
        self.orig = args

    def forward(self, query, key, value):
        if self.plant == "ignore_hook":
            query, key, value = self.orig[:3]
        hq, hk, d = STUB_HEADS
        t = key.shape[0]
        out = torch.zeros(t, hq * d, dtype=query.dtype)
        if t <= STUB_FIRST:
            n = min(t, sp.GAP_ROWS)
            scale = 1.0 if self.plant == "ignore_scale" else self.impl.scale
            o = sp.attend32(query.view(t, hq, d)[-n:], key.view(t, hk, d), value.view(t, hk, d),
                            torch.arange(t - n, t), scale)
            out[-n:] = o.reshape(n, -1).to(query.dtype)
        return out


class StubEngine:
    """The --stub engine: one StubAttention module per global layer. A prompt runs as a first chunk of STUB_FIRST
    tokens and one chunk for the rest, each through the layers in order, as the engine calls them. The first chunk has
    random q, k and v with K_unrot = bf16(w V_unrot), as the model has."""

    def __init__(self, w: dict[int, float], plant: str):
        self.w = w
        self.modules = {layer: StubAttention(plant) for layer in GLOBAL_LAYERS}

    def qkv(self, layer: int, t: int, seed: int):
        hq, hk, d = STUB_HEADS
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(t, hk, d, generator=g).to(torch.bfloat16)
        k = (v.float() * self.w[layer]).to(torch.bfloat16)
        rot, _ = sp.rot_dims(d)
        k[..., rot] = (torch.randn(t, hk, len(rot), generator=g) * 0.5).to(torch.bfloat16)
        q = (torch.randn(t, hq, d, generator=g) * 8).to(torch.bfloat16)
        return q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)

    def prompt(self, length: int, seed: int) -> None:
        hq, hk, d = STUB_HEADS
        first, rest = min(length, STUB_FIRST), max(0, length - STUB_FIRST)
        for layer in GLOBAL_LAYERS:
            self.modules[layer](*self.qkv(layer, first, seed * 100 + layer))
        if rest:
            q = torch.zeros(rest, hq * d, dtype=torch.bfloat16)
            kv = torch.zeros(rest, hk * d, dtype=torch.bfloat16)
            for layer in GLOBAL_LAYERS:
                self.modules[layer](q, kv, kv)


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
                      ("use_v2_model_runner", lambda: llm.llm_engine.vllm_config.use_v2_model_runner),   # R-77 F6
                      ("multiprocessing_env", lambda: os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))):
        try:
            rec[name] = get()
        except Exception as exc:  # recorded; V2 reads what exists
            rec[name] = f"unavailable: {type(exc).__name__}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data_e55.json")
    ap.add_argument("--scales", type=Path, default=HERE.parent / "gemma4_kv_scaletie" / "scales_e54.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--model-id", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    ap.add_argument("--revision", default=an.REVISION)
    ap.add_argument("--start", type=float, default=time.time())
    ap.add_argument("--deadline", type=float, default=time.time() + 2_400)
    ap.add_argument("--reserve", type=float, default=60.0, help="seconds kept free before the deadline")
    ap.add_argument("--rehearsal", action="store_true", help="a small Gemma 4 model: synthetic scales, short documents")
    ap.add_argument("--rehearsal-tokens", type=int, default=16_384)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-effect", action="append", default=[], help="ARM=nats, for the stub")
    ap.add_argument("--stub-seconds", type=float, default=0.0)
    ap.add_argument("--stub-plant", default="", help="ignore_scale, ignore_hook or gen_error, for the stub")
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
    rec = {"phase": "main", "started_at": now(), "status": "running", "stub": args.stub,
           "rehearsal": args.rehearsal, "data_sha256": sha_file(args.data), "scales_sha256": sha_file(args.scales),
           "model_id": args.model_id, "revision_arg": args.revision,
           "snapshot_revision": Path(args.snapshot).name if args.snapshot else None,   # R-77 F9: the folder name
           "docs": [d["id"] for d in docs], "doc_lengths": {d["id"]: d["length"] for d in docs}, "arms": {},
           "cut": [], "step_errors": [], "errors": [], "decision_arms_done": False}
    run_path = out / "run_main.json"
    splice = None

    def save() -> None:
        if splice is not None:
            rec.update(splice.evidence())
        run_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")

    def on_term(signum, frame):
        raise Terminated("SIGTERM (the session timeout)")
    signal.signal(signal.SIGTERM, on_term)
    total_tokens = sum(d["length"] for d in docs)
    arm_seconds: list[float] = []

    def fits(name: str, share: float) -> bool:
        est = (sum(arm_seconds) / len(arm_seconds) if arm_seconds else 0.0) * share
        ok = time.time() + 1.5 * est <= args.deadline - args.reserve
        if not ok:
            rec["cut"].append({"step": name, "estimate_s": round(est, 1), "at": now(),
                               "elapsed_s": round(time.time() - args.start, 1)})
            save()
        return ok

    code = 1
    try:
        if args.stub:
            stub = Stub({k: float(v) for k, v in (e.split("=") for e in args.stub_effect)}, args.stub_seconds)
            engine = StubEngine(w_file, args.stub_plant)
            splice = sp.Splice({}, scales, {})
            hooked = []
            for layer, m in engine.modules.items():
                splice.w[layer], splice.heads[layer] = w_file[layer], STUB_HEADS
                splice.attach(layer, m)
                hooked.append({"layer": layer, "num_heads": STUB_HEADS[0], "num_kv_heads": STUB_HEADS[1],
                               "head_size": STUB_HEADS[2], "w": w_file[layer], "impl": "StubImpl", "impl_scale": 1.0})
            rec["hooks"] = {"hooked": hooked, "problems": []}
            rec["engine"] = {"vllm": "STUB", "max_model_len": ENGINE["max_model_len"], "kv_cache_dtype": "auto",
                             "prefix_caching": False, "compilation_mode": "CompilationMode.NONE",
                             "cudagraph_mode": "CUDAGraphMode.NONE", "use_v2_model_runner": True,
                             "max_num_batched_tokens": ENGINE["max_num_batched_tokens"]}
            rec["tok_check"] = {"ok": True, "stub": True}
            rec["gpu"] = "STUB"
            seed_of = lambda d: int(d["ids_sha256"][:6], 16)  # noqa: E731

            def score(name, doc):
                engine.prompt(doc["length"], seed_of(doc))
                return stub.score(name, doc)

            def generate(arm):
                if args.stub_plant == "gen_error":
                    raise RuntimeError("planted GEN failure")
                for g in gen:
                    engine.prompt(len(g["ids"]), seed_of(g))
                return stub.generate(arm, gen), ["length"] * len(gen)

            def warm():
                engine.prompt(WARM_TOKENS, 1)
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            torch.backends.cuda.matmul.allow_tf32 = False
            t0 = time.time()
            llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                      tokenizer=args.snapshot or args.model_id, **ENGINE)
            rec["engine_seconds"] = round(time.time() - t0, 1)
            signal.signal(signal.SIGTERM, on_term)  # again, in case the engine start replaced it
            rec["engine_kwargs"] = ENGINE
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["engine"] = engine_record(llm)
            tok = llm.get_tokenizer()
            got = [tok.encode(c["text"], add_special_tokens=False) for c in data["tok_check"]]
            rec["tok_check"] = {"ok": got == [c["ids"] for c in data["tok_check"]], "got": got}
            splice = sp.Splice({}, scales, {})
            rec["hooks"] = llm.apply_model(lambda model: register(model, splice,
                                                                  None if args.rehearsal else w_file))[0]
            save()
            if rec["hooks"]["problems"] and not args.rehearsal:
                raise RuntimeError("V3 hook problems: " + "; ".join(rec["hooks"]["problems"][:5]))
            sp_nll = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False,
                                    flat_logprobs=True)
            sp_gen = SamplingParams(max_tokens=GEN_MAX_TOKENS, temperature=0.0, detokenize=False)  # R-77 F10

            def score(name, doc):
                o = llm.generate([TokensPrompt(prompt_token_ids=doc["ids"])], sp_nll, use_tqdm=False)[0]
                return parse_prompt_logprobs(o.prompt_logprobs, doc["ids"])

            def generate(arm):
                outs = llm.generate([TokensPrompt(prompt_token_ids=g["ids"]) for g in gen], sp_gen, use_tqdm=False)
                return [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].finish_reason for o in outs]

            def warm():
                llm.generate([TokensPrompt(prompt_token_ids=docs[0]["ids"][:WARM_TOKENS])], sp_nll, use_tqdm=False)
        if args.rehearsal:                          # R-77 B2: no E-54 scales for this model
            splice.scales = synthetic_scales(splice.w)
            rec["synthetic_scales"] = {str(k): v for k, v in splice.scales.items()}

        def run_arm(arm: str, subset: list[dict] | None = None, name: str | None = None) -> None:
            name = name or arm
            splice.set_arm(arm if arm in sp.ARMS else "BF16")
            t0 = time.time()
            res, finite = {}, True
            for d in subset or docs:
                splice.new_doc(d["id"] if name != "DET" else f"DET:{d['id']}")
                lp, top1, rank = score(name, d)
                finite &= bool(np.isfinite(lp).all())
                res[f"{d['id']}_lp"], res[f"{d['id']}_top1"], res[f"{d['id']}_rank"] = lp, top1, rank
            np.savez_compressed(arrays / f"{name}.npz", **res)
            secs = time.time() - t0
            rec["arms"][name] = {"seconds": round(secs, 1), "docs": [d["id"] for d in subset or docs],
                                 "finite": finite, "finished_at": now()}
            if subset is None:
                arm_seconds.append(secs)
            save()

        def preflight() -> dict:
            """Every arm and DET on the first PRE_TOKENS tokens of the first document (one chunk), with the checks of
            V5, V7 and DET (README Section 6, A1). The evidence is reset before and after."""
            t0 = time.time()
            ids = docs[0]["ids"][:PRE_TOKENS]
            pre = dict(docs[0], id="PRE", ids=ids, length=len(ids), ids_sha256=ids_sha(ids))
            lps = {}
            for arm in sp.ARMS:
                splice.set_arm(arm)
                splice.new_doc("PRE")
                lps[arm] = score(arm, pre)[0]
            saved = splice.detach()
            splice.new_doc("PRE")
            lps["DET"] = score("DET", pre)[0]
            splice.reattach(saved)
            ev = splice.evidence()
            problems = an.preflight_problems(lps, ev, len(splice.modules), len(ids))
            return {"tokens": len(ids), "seconds": round(time.time() - t0, 1), "problems": problems,
                    "equal_to_bf16": {a: float(np.mean(x == lps["BF16"])) for a, x in lps.items()},
                    **{k: ev[k] for k in ("checks", "gaps", "gaps_id", "seps", "hook_errors")}}

        def run_gen() -> None:
            t0 = time.time()
            outs, finish = {}, {}
            for arm in GEN_ARMS:
                splice.set_arm(arm)
                splice.new_doc("GEN")
                outs[arm], finish[arm] = generate(arm)
            (out / "gen_outputs.json").write_text(json.dumps({"ids": [g["id"] for g in gen], "outputs": outs,
                                                              "finish": finish}))
            rec["arms"]["GEN"] = {"seconds": round(time.time() - t0, 1), "arms": list(GEN_ARMS),
                                  "finished_at": now()}
            save()

        def step(name: str, share: float, fn) -> None:
            if not fits(name, share):
                return
            try:
                fn()
            except Exception as exc:  # a reported step does not stop the run (R-77 F2)
                rec["step_errors"].append({"step": name, "error": f"{type(exc).__name__}: {exc}"[:500],
                                           "at": now()})
                save()

        warm()                                       # with the hooks on and the arm BF16; not scored
        splice.reset_evidence()                      # R-77 B1: the warm-up used the first check of each layer
        rec["preflight"] = preflight()
        splice.reset_evidence()
        save()
        if rec["preflight"]["problems"] and not args.rehearsal:
            raise PreflightError("; ".join(rec["preflight"]["problems"][:6]))
        run_arm("BF16")
        saved = splice.detach()
        run_arm("BF16", subset=[d for d in docs if d["id"] in DET_DOCS], name="DET")
        splice.reattach(saved)
        for arm in DECISION_ARMS[1:]:
            run_arm(arm)
        rec["decision_arms_done"] = True
        save()
        gen_share = len(GEN_ARMS) * sum(len(g["ids"]) + GEN_MAX_TOKENS * 8 for g in gen) / total_tokens
        step("BF16R", 1.0, lambda: run_arm("BF16R"))
        step("JOINT8", 1.0, lambda: run_arm("JOINT8"))
        step("GEN", gen_share, run_gen)
        step("TIED8_UP", 1.0, lambda: run_arm("TIED8_UP"))
        splice.set_arm("BF16")
        rec["status"] = "ok"
        code = 0
    except BaseException as exc:  # recorded and saved: an error, a pre-flight stop or SIGTERM
        rec["status"] = "error"
        rec["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        rec["traceback"] = traceback.format_exc()[-3000:]
        code = 3 if isinstance(exc, PreflightError) else 1
    rec["finished_at"] = now()
    save()
    print(json.dumps({"status": rec["status"], "arms": list(rec["arms"]), "cut": [c["step"] for c in rec["cut"]],
                      "step_errors": [e["step"] for e in rec["step_errors"]], "errors": rec["errors"][:2]}))
    sys.exit(code)


if __name__ == "__main__":
    main()
