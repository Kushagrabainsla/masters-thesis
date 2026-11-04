"""E-56 main phase (README Sections 4 and 5): the BF16 Gemma 4 31B (vLLM, eager, BF16 cache, no prefix caching) with
the E-55 splice hooks (`splice_e55.py`, unchanged) on the 10 global layers. Per arm, in this order: the probe (health
slice B1, scored alone: the V5 and V7 evidence needs one prompt in the call) and the other health slices, one at a
time; the GSM8K scoring sequences (the reference NLL, batched); the GSM8K answers and the needle answers (greedy,
batched). The run: the warm-up, the pre-flight (every arm and DET on the probe; exit 3 on a problem), BF16, the health
gate (V8 on the BF16 answers and scores; exit 3 on a problem, L-29), DET (BF16 again with the hooks removed, the whole
workload in the same order), BF16C, BF16R, then STOCK8 while the time allows.

--stub runs the same flow on the Mac: stub Attention modules carry the real hooks in the engine's call shapes (single
prompts; batched prefill of several sequences in chunks of the batched-token limit; then one token per running
sequence per decode call; L-26), and synthetic answers and scores stand in for the model. --rehearsal (S5, a small
Gemma 4 model): the gate records its problems and does not stop.

Usage: run_e56.py --data data_e56.json --out <res> --snapshot <dir> --model-id <id> --revision <sha> --start <epoch>
                  --deadline <epoch> [--rehearsal] [--stub ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E55 = HERE.parent / "gemma4_kv_model"
sys.path[:0] = [str(HERE), str(E55)]
import analyze_e56 as an  # noqa: E402
import run_e55 as r55  # noqa: E402
import splice_e55 as sp  # noqa: E402

ARMS = an.ARMS
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_model_len": 10_240,
          "max_num_batched_tokens": an.ENGINE_TOKENS, "max_num_seqs": 64, "gpu_memory_utilization": 0.92,
          "enable_prefix_caching": False, "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}
GSM_MAX_TOKENS, NEEDLE_MAX_TOKENS = 640, 128       # (A2, S5) 400 and 24 before A2; (A1) 320 before A1
WARM_TOKENS = 512


class HealthError(RuntimeError):
    """The BF16 baseline failed the health gate (V8): the run stops before the other arms (exit 3)."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class StubEngine56(r55.StubEngine):
    """The E-55 stub engine (single prompts) plus the batched call shapes of the vLLM scheduler."""

    def batch(self, prompt_lens: list[int], out_lens: list[int]) -> None:
        hq, hk, d = r55.STUB_HEADS
        total, step = sum(prompt_lens), ENGINE["max_num_batched_tokens"]
        calls = [min(step, total - s) for s in range(0, total, step)]
        calls += [sum(1 for o in out_lens if o - 1 > s) for s in range(max([o - 1 for o in out_lens], default=0))]
        for t in calls:
            q = torch.zeros(t, hq * d, dtype=torch.bfloat16)
            kv = torch.zeros(t, hk * d, dtype=torch.bfloat16)
            for layer in r55.GLOBAL_LAYERS:
                self.modules[layer](q, kv, kv)


class StubTasks:
    """Synthetic answers and scores for the Mac tests. Each item has a latent chance p of a right answer (0.97 for 80%
    of the items, 0.4 for the rest); BF16 draws its answer once. An arm resamples the answer of an item at its flip rate
    (a new draw with the same p: no net accuracy change), and turns a right answer wrong at its loss rate (planted).
    The reference log-probabilities get the arm effect (nats per token) and noise for every arm but BF16 and DET."""

    FLIP = {"BF16": 0.0, "DET": 0.0, "BF16C": 0.02, "BF16R": 0.02, "STOCK8": 0.06}
    EFFECT = {"BF16": 0.0, "DET": 0.0, "BF16C": 0.0, "BF16R": 0.0, "STOCK8": 0.003}

    def __init__(self, flips: dict, losses: dict, effects: dict, plant: str, seconds: float):
        self.flip, self.loss = dict(self.FLIP, **flips), losses
        self.effect = dict(self.EFFECT, **effects)
        self.plant, self.seconds = plant, seconds

    def _p(self, item: dict) -> float:
        p = 0.97 if seed_of("p", item["id"]) % 5 else 0.4
        return 0.2 if self.plant == "broken_baseline" else p

    def _correct(self, arm: str, item: dict) -> tuple[bool, bool]:
        """(the answer is right, the answer differs from the BF16 draw)."""
        rng = np.random.default_rng(seed_of("base", item["id"]))
        right = bool(rng.random() < self._p(item))
        a = "BF16" if arm == "DET" else arm
        r2 = np.random.default_rng(seed_of(a, item["id"]))
        changed = False
        if r2.random() < self.flip.get(a, 0.0):
            right, changed = bool(r2.random() < self._p(item)), True
        if right and r2.random() < self.loss.get(a, 0.0):
            right, changed = False, True
        return right, changed

    def answer(self, arm: str, item: dict, kind: str) -> dict:
        right, changed = self._correct(arm, item)
        n = 30 + seed_of("len", item["id"]) % 40 if kind == "gsm8k" else 6
        ids = [(seed_of("tok", item["id"]) + i) % 50_000 for i in range(n)]
        src = arm if changed else "BF16"           # an unchanged answer is the BF16 answer, wrong ones included
        if changed:
            ids[n // 2:] = [x + 1 for x in ids[n // 2:]]
        if kind == "gsm8k":
            gold = int(float(item["gold"]))
            text = f"Steps. #### {gold if right else gold + 1 + seed_of(src, item['id']) % 7}"
        else:
            text = item["gold"] if right else f"wrong-{1000 + seed_of(src, item['id']) % 9000}"
        return {"text": text, "ids": ids, "finish": "stop"}

    def score(self, arm: str, ids: list[int], start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-probabilities of tokens 1 .. L-1 of a sequence (the parser's shape)."""
        n = len(ids) - 1
        rng = np.random.default_rng(seed_of("score", ids[:8], len(ids)))
        base = -np.abs(rng.normal(0.4 if start else 1.5, 0.3 if start else 1.0, n))
        a = "BF16" if arm == "DET" else arm
        noise = 0.0 if a == "BF16" else np.random.default_rng(seed_of(a, ids[:8])).normal(0, 0.02, n)
        lp = base - self.effect.get(a, 0.0) + noise
        if self.plant == "det_noise" and arm == "DET":
            lp = lp + 1e-3
        top1 = np.asarray(ids[1:], dtype=np.int64)
        time.sleep(self.seconds)
        return lp.astype(np.float32), top1.astype(np.int32), np.ones(n, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data_e56.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
    ap.add_argument("--revision", default=an.REVISION)
    ap.add_argument("--start", type=float, default=time.time())
    ap.add_argument("--deadline", type=float, default=time.time() + 3_600)
    ap.add_argument("--reserve", type=float, default=60.0, help="seconds kept free before the deadline")
    ap.add_argument("--rehearsal", action="store_true", help="a small Gemma 4 model: the gate does not stop")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-flip", action="append", default=[], help="ARM=rate, for the stub")
    ap.add_argument("--stub-loss", action="append", default=[], help="ARM=rate, for the stub")
    ap.add_argument("--stub-effect", action="append", default=[], help="ARM=nats, for the stub")
    ap.add_argument("--stub-seconds", type=float, default=0.0)
    ap.add_argument("--stub-plant", default="",
                    help="ignore_scale, ignore_hook, broken_baseline, det_noise or step_error, for the stub")
    args = ap.parse_args()

    out, arrays = args.out, args.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    gsm, needle, health = data["gsm8k"], data["needle"], data["health"]
    probe = health[0]
    rec = {"phase": "main", "started_at": now(), "status": "running", "stub": args.stub, "rehearsal": args.rehearsal,
           "data_sha256": an.sha_file(args.data), "model_id": args.model_id, "revision_arg": args.revision,
           "snapshot_revision": Path(args.snapshot).name if args.snapshot else None, "arms": {}, "cut": [],
           "step_errors": [], "errors": [], "decision_arms_done": False}
    run_path = out / "run_main.json"
    splice = None

    def save() -> None:
        if splice is not None:
            rec.update(splice.evidence())
        run_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")

    def on_term(signum, frame):
        raise r55.Terminated("SIGTERM (the session timeout)")
    signal.signal(signal.SIGTERM, on_term)
    arm_seconds: list[float] = []

    def fits(name: str) -> bool:
        est = sum(arm_seconds) / len(arm_seconds) if arm_seconds else 0.0
        ok = time.time() + 1.5 * est <= args.deadline - args.reserve
        if not ok:
            rec["cut"].append({"step": name, "estimate_s": round(est, 1), "at": now(),
                               "elapsed_s": round(time.time() - args.start, 1)})
            save()
        return ok

    def parse(lists):
        flip = lambda xs: {k: float(v) for k, v in (e.split("=") for e in xs)}  # noqa: E731
        return [flip(x) for x in lists]

    code = 1
    try:
        if args.stub:
            torch.set_num_threads(1)                 # tiny tensors: thread start-up costs more than the work
            flips, losses, effects = parse([args.stub_flip, args.stub_loss, args.stub_effect])
            tasks = StubTasks(flips, losses, effects, args.stub_plant, args.stub_seconds)
            w = {layer: 0.0615234375 if layer != 29 else 0.0625 for layer in r55.GLOBAL_LAYERS}
            engine = StubEngine56(w, args.stub_plant if args.stub_plant in ("ignore_scale", "ignore_hook") else "")
            splice = sp.Splice({}, None, {})
            hooked = []
            for layer, m in engine.modules.items():
                splice.w[layer], splice.heads[layer] = w[layer], r55.STUB_HEADS
                splice.attach(layer, m)
                hooked.append({"layer": layer, "num_heads": r55.STUB_HEADS[0], "num_kv_heads": r55.STUB_HEADS[1],
                               "head_size": r55.STUB_HEADS[2], "w": w[layer], "impl": "StubImpl", "impl_scale": 1.0})
            rec["hooks"] = {"hooked": hooked, "problems": []}
            rec["engine"] = {"vllm": "STUB", "max_model_len": ENGINE["max_model_len"], "kv_cache_dtype": "auto",
                             "prefix_caching": False, "compilation_mode": "0", "cudagraph_mode": "NONE",
                             "use_v2_model_runner": True, "max_num_batched_tokens": ENGINE["max_num_batched_tokens"]}
            rec["engine_kwargs"] = ENGINE
            rec["tok_check"] = {"ok": True, "stub": True}
            rec["gpu"] = "STUB"

            def score_one(arm, ids):
                engine.prompt(len(ids), seed_of(ids[:8]) % 10_000)
                return tasks.score(arm, ids, 0)

            def score_batch(arm, seqs, starts):
                engine.batch([len(s) for s in seqs], [1] * len(seqs))
                return [tasks.score(arm, s, st) for s, st in zip(seqs, starts)]

            def gen_batch(arm, items, max_tokens, kind):
                if args.stub_plant == "step_error" and arm == "STOCK8":
                    raise RuntimeError("planted STOCK8 failure")
                res = {i["id"]: tasks.answer(arm, i, kind) for i in items}
                engine.batch([len(i["ids"]) for i in items], [len(r["ids"]) for r in res.values()])
                plp = {i["id"]: tasks.score(arm, i["ids"], 0) for i in items} if kind == "needle" else None
                if args.stub_plant == "det_ids" and arm == "DET" and kind == "gsm8k":
                    first = items[0]["id"]
                    res[first] = dict(res[first], ids=res[first]["ids"][:-1] + [res[first]["ids"][-1] + 1])
                return res, plp

            def warm():
                engine.prompt(WARM_TOKENS, 1)
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            torch.backends.cuda.matmul.allow_tf32 = False
            t0 = time.time()
            kwargs = dict(ENGINE)
            try:
                llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                          tokenizer=args.snapshot or args.model_id, **kwargs)
            except Exception as exc:  # (A1, R-79 F6) one retry: more KV room, or more room outside the budget
                msg = f"{type(exc).__name__}: {exc}"
                kwargs["gpu_memory_utilization"] = 0.88 if "OutOfMemory" in msg else 0.95
                rec["engine_retry"] = {"error": msg[:500], "gpu_memory_utilization": kwargs["gpu_memory_utilization"]}
                save()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                          tokenizer=args.snapshot or args.model_id, **kwargs)
            rec["engine_seconds"] = round(time.time() - t0, 1)
            signal.signal(signal.SIGTERM, on_term)  # again, in case the engine start replaced it
            rec["engine_kwargs"] = kwargs
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["engine"] = r55.engine_record(llm)
            tok = llm.get_tokenizer()
            got = [tok.encode(c["text"], add_special_tokens=False) for c in data["tok_check"]]
            rec["tok_check"] = {"ok": got == [c["ids"] for c in data["tok_check"]], "got": got}
            splice = sp.Splice({}, None, {})
            rec["hooks"] = llm.apply_model(lambda model: r55.register(model, splice, None))[0]
            save()
            if rec["hooks"]["problems"] and not args.rehearsal:
                raise RuntimeError("V3 hook problems: " + "; ".join(rec["hooks"]["problems"][:5]))
            sp_nll = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False,
                                    flat_logprobs=True)

            def score_one(arm, ids):
                o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp_nll, use_tqdm=False)[0]
                return r55.parse_prompt_logprobs(o.prompt_logprobs, ids)

            def score_batch(arm, seqs, starts):
                outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], sp_nll, use_tqdm=False)
                return [r55.parse_prompt_logprobs(o.prompt_logprobs, s) for o, s in zip(outs, seqs)]

            def gen_batch(arm, items, max_tokens, kind):
                extra = {"prompt_logprobs": 1, "flat_logprobs": True} if kind == "needle" else {}   # (A1, O1)
                params = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=0, detokenize=True,
                                        stop_token_ids=[data["eot_id"]], **extra)   # <turn|>; also in the config
                outs = llm.generate([TokensPrompt(prompt_token_ids=i["ids"]) for i in items], params, use_tqdm=False)
                res = {i["id"]: {"text": o.outputs[0].text, "ids": list(o.outputs[0].token_ids),
                                 "finish": o.outputs[0].finish_reason} for i, o in zip(items, outs)}
                plp = ({i["id"]: r55.parse_prompt_logprobs(o.prompt_logprobs, i["ids"]) for i, o in zip(items, outs)}
                       if kind == "needle" else None)
                return res, plp

            def warm():
                llm.generate([TokensPrompt(prompt_token_ids=probe["ids"][:WARM_TOKENS])], sp_nll, use_tqdm=False)

        def run_arm(arm: str, name: str | None = None) -> dict:
            """One arm's whole workload; DET is BF16 with the hooks removed by the caller. Returns the outputs."""
            name = name or arm
            splice.set_arm(arm)
            t0, res, finite, times = time.time(), {}, True, {}
            for h in health:
                splice.new_doc("PROBE" if h is probe else h["id"])
                lp, top1, rank = score_one(name, h["ids"])
                finite &= bool(np.isfinite(lp).all())
                res[f"{h['id']}_lp"], res[f"{h['id']}_top1"], res[f"{h['id']}_rank"] = lp, top1, rank
            times["health"] = round(time.time() - t0, 1)
            t1 = time.time()
            splice.new_doc("SCORE")
            scored = score_batch(name, [g["score_ids"] for g in gsm], [g["answer_start"] for g in gsm])
            for g, (lp, top1, rank) in zip(gsm, scored):
                a = g["answer_start"] - 1                 # lp[j] scores token j + 1
                finite &= bool(np.isfinite(lp[a:]).all())
                res[f"{g['id']}_lp"], res[f"{g['id']}_top1"], res[f"{g['id']}_rank"] = lp[a:], top1[a:], rank[a:]
            times["score"] = round(time.time() - t1, 1)
            outs = {}
            for task, items, mx in (("gsm8k", gsm, GSM_MAX_TOKENS), ("needle", needle, NEEDLE_MAX_TOKENS)):
                t1 = time.time()
                splice.new_doc(task.upper())
                outs[task], plp = gen_batch(name, items, mx, task)
                for i, (lp, top1, rank) in (plp or {}).items():   # (A1, O1) the long-context NLL from token 1,025
                    finite &= bool(np.isfinite(lp[an.LONG_FROM:]).all())
                    res[f"{i}_plp"], res[f"{i}_prank"] = lp[an.LONG_FROM:], rank[an.LONG_FROM:]
                times[task] = round(time.time() - t1, 1)
            np.savez(arrays / f"{name}.npz", **res)       # small arrays: no compression (time)
            (out / f"outputs_{name}.json").write_text(json.dumps(outs) + "\n")
            secs = time.time() - t0
            rec["arms"][name] = {"seconds": round(secs, 1), "times": times, "finite": finite, "finished_at": now()}
            arm_seconds.append(secs)
            save()
            return {"outs": outs, "arrays": res}

        def preflight() -> dict:
            """Every arm and DET on the probe alone (one chunk): V5, V7, the counts and DET (README Section 4.5)."""
            t0 = time.time()
            lps = {}
            for arm in ARMS:
                splice.set_arm(arm)
                splice.new_doc("PRE")
                lps[arm] = score_one(arm, probe["ids"])[0]
            saved = splice.detach()
            splice.new_doc("PRE")
            lps["DET"] = score_one("DET", probe["ids"])[0]
            if args.stub and args.stub_plant == "det_noise":
                lps["DET"] = lps["DET"] + 1e-3
            splice.reattach(saved)
            ev = splice.evidence()
            problems = an.preflight_problems(lps, ev, len(splice.modules), len(probe["ids"]))
            stock8 = an.preflight_problems(lps, ev, len(splice.modules), len(probe["ids"]), arms=("STOCK8",))
            return {"tokens": len(probe["ids"]), "seconds": round(time.time() - t0, 1), "problems": problems,
                    "stock8_problems": [p for p in stock8 if p not in problems],
                    "equal_to_bf16": {a: float(np.mean(x == lps["BF16"])) for a, x in lps.items()},
                    **{k: ev[k] for k in ("checks", "gaps", "gaps_id", "seps", "hook_errors")}}

        warm()                                       # with the hooks on and the arm BF16; not scored
        splice.reset_evidence()                      # E-55 R-77 B1: the warm-up used the first check of each layer
        rec["preflight"] = preflight()
        splice.reset_evidence()
        save()
        if rec["preflight"]["problems"] and not args.rehearsal:
            raise r55.PreflightError("; ".join(rec["preflight"]["problems"][:6]))
        base = run_arm("BF16")
        bad, m = an.health_problems(an.grade(data, base["outs"]), an.ref_nll(base["arrays"], data), data)
        rec["health"] = {"problems": bad, **m}
        save()
        if bad and not args.rehearsal:
            raise HealthError("; ".join(bad))
        saved = splice.detach()
        run_arm("BF16", name="DET")
        splice.reattach(saved)
        run_arm("BF16C")
        run_arm("BF16R")
        rec["decision_arms_done"] = True
        save()
        if rec["preflight"].get("stock8_problems"):
            rec["step_errors"].append({"step": "STOCK8", "error": "skipped: pre-flight problems", "at": now()})
            save()
        elif fits("STOCK8"):
            try:
                run_arm("STOCK8")
            except Exception as exc:  # a reported arm does not stop the run (E-55 R-77 F2)
                rec["step_errors"].append({"step": "STOCK8", "error": f"{type(exc).__name__}: {exc}"[:500],
                                           "at": now()})
                save()
        splice.set_arm("BF16")
        rec["status"] = "ok"
        code = 0
    except BaseException as exc:  # recorded and saved: an error, a pre-flight or health stop, or SIGTERM
        rec["status"] = "error"
        rec["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        rec["traceback"] = traceback.format_exc()[-3000:]
        code = 3 if isinstance(exc, (r55.PreflightError, HealthError)) else 1
    rec["finished_at"] = now()
    save()
    print(json.dumps({"status": rec["status"], "arms": list(rec["arms"]), "cut": [c["step"] for c in rec["cut"]],
                      "step_errors": [e["step"] for e in rec["step_errors"]], "errors": rec["errors"][:2]}))
    sys.exit(code)


if __name__ == "__main__":
    main()
