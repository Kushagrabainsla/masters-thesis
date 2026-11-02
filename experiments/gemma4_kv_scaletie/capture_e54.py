"""E-54 capture (README Sections 4 and 11): the E-53 capture (`capture_e53.py`: the hooks on the global attention
layers, the fail fast, the checks and the capture files) on the 7 fresh test prompts of the frozen plan: 4 plain texts
of 8,192 tokens, 2 chat-template prompts of 8,192 tokens with turn markers, and 1 plain text of 32,768 tokens (4
prefill chunks of 8,448 tokens or less). No prompt takes a file that E-53 used (`e53_files.json`; V2) or a file that an
earlier E-54 prompt used.

What is new against E-53 (the rest is `capture_e53.py`, README Section 4.3):
- `build_prompts`: the frozen plan of `analyze_e54.PLAN`, the E-53 exclusion, the chat prompts and the long prompt.
- The engine: max_model_len 33,024 (32,768 + 256) with 8,448 batched tokens per step, so the long prompt runs as 4
  prefill chunks (vLLM V1 always chunks a prefill); the hooks record each chunk (E-53 tested 2 chunks).
- The stub: a scalar k_norm weight per layer (as the 31B, R-74 C9) and a toy chat template.

Usage: capture_e54.py --snapshot <dir> --model-id <id> --revision <sha> --out <res> --capture <dir> [shape flags]
       capture_e54.py --stub --out <res> --capture <dir> [--stub-...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_accuracy"))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import capture_e53 as c53  # noqa: E402  (Store, register, global_layers, checks, save, engine record, stub layer)
from analyze_e54 import PLAN  # noqa: E402

ENGINE = dict(c53.ENGINE, max_model_len=33024)   # 32,768 + 256; 8,448 batched tokens per step (E-53)
CHAT_CHARS = 6000                                  # at most this many characters of one file in a chat turn
CHAT_ASKS = ("Here is a file. Explain what it does and how it works.\n\n",   # the ask of each chat prompt in turn
             "Read this file. Describe its purpose and its main parts.\n\n")  # (A1, R-75 F7: no shared prefix)
TOY_SPECIAL = {"<bos>": 2, "<start_of_turn>": 1, "<end_of_turn>": 0}   # the stub tokenizer's special strings
E53_FILES = HERE / "e53_files.json"


def toy_encode(text: str) -> list[int]:
    """The stub tokenizer: one id per byte (ids 3 to 258); the three special strings become ids 2, 1 and 0."""
    out: list[int] = []
    i = 0
    while i < len(text):
        for s, sid in TOY_SPECIAL.items():
            if text.startswith(s, i):
                out.append(sid)
                i += len(s)
                break
        else:
            j = min((p for p in (text.find(s, i) for s in TOY_SPECIAL) if p >= 0), default=len(text))
            out += [b + 3 for b in text[i:j].encode("utf-8")]
            i = j
    return out


def toy_chat(turns: list[dict]) -> str:
    """The stub chat template (the shape of a Gemma template: BOS, then marked turns)."""
    return "<bos>" + "".join(f"<start_of_turn>{t['role']}\n{t['content']}<end_of_turn>\n" for t in turns)


def load_exclude(path: Path = E53_FILES) -> dict:
    d = json.loads(path.read_text()) if path.exists() else {"files": []}
    return {"paths": {f["path"] for f in d["files"]}, "sha256": {f["sha256"] for f in d["files"]}}


def build_prompts(src: dict[str, list[Path]], encode, bos: int, base: Path, special: frozenset, render_chat,
                  exclude: dict, plan: dict = PLAN, seed: int | None = None, chat_chars: int = CHAT_CHARS) -> list[dict]:
    """The test prompts of the plan, in order. A plain prompt is built as in E-53: BOS, then the tokens of text files in
    a seeded order (at most c53.FILE_TOKENS per file), each followed by a blank line, cut to the planned length. A chat
    prompt alternates a user turn (its ask of CHAT_ASKS and at most chat_chars characters of a file) and an assistant
    turn (at most chat_chars characters of the next file), renders the turns with the chat template, encodes the text
    (the template strings become special ids) and cuts the ids to the planned length; BOS stays at position 0. Every
    prompt skips a file of `exclude` (by path or SHA-256), a file that an earlier prompt used, and (R-73 F4) a file
    whose ids hold a special id."""
    seed = plan["seed"] if seed is None else seed
    sep = list(encode("\n\n"))
    taken: set[str] = set()
    out = []
    for spec in plan["prompts"]:
        pid, kind, fam, tokens = spec["id"], spec["kind"], spec["family"], spec["tokens"]
        rng = random.Random(seed * 1000 + pid)
        order = list(src[fam])
        rng.shuffle(order)
        seen = set(order)
        rest = [f for f in src["mixed"] if f not in seen]
        rng.shuffle(rest)
        used, n_skip, n_excl = [], 0, 0
        ids: list[int] = [bos]
        turns: list[dict] = []
        ask = CHAT_ASKS[sum(1 for o in out if o["kind"] == "chat") % len(CHAT_ASKS)]
        for f in order + rest:
            if len(ids) >= tokens:
                break
            try:
                name = str(f.relative_to(base))
            except ValueError:
                name = str(f)
            if name in taken:
                continue
            try:
                raw = f.read_bytes()
                text = raw.decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            digest = c53.sha(raw)
            if name in exclude["paths"] or digest in exclude["sha256"]:
                n_excl += 1
                continue
            t = list(encode(text))
            if not t:
                continue
            if any(i in special for i in t):
                n_skip += 1
                continue
            taken.add(name)
            if kind == "plain":
                take = (t[: c53.FILE_TOKENS] + sep)[: tokens - len(ids)]
                ids += take
                used.append({"path": name, "sha256": digest, "tokens": len(take)})
            else:
                user = len(turns) % 2 == 0
                turns.append({"role": "user" if user else "assistant",
                              "content": (ask if user else "") + text[:chat_chars]})
                ids = list(encode(render_chat(turns)))
                if ids[:1] != [bos]:
                    ids = [bos] + ids
                used.append({"path": name, "sha256": digest, "chars": min(len(text), chat_chars)})
        ids = ids[:tokens]
        out.append({"id": pid, "kind": kind, "family": fam, "files": used, "n_tokens": len(ids),
                    "token_sha256": c53.ids_sha(ids), "bos_at_0": ids[:1] == [bos],
                    "special_after_0": sum(1 for i in ids[1:] if i in special), "skipped_special": n_skip,
                    "excluded": n_excl, "turns": len(turns), "token_ids": [int(i) for i in ids]})
    return out


def stub_plan(args) -> dict:
    """The stub plan: the kinds of the frozen plan with short lengths (the long prompt twice the others)."""
    return dict(PLAN, q_rows=args.stub_q_rows, heads=args.heads, kv_heads=args.kv_heads,
                global_layers=args.stub_layers,
                prompts=[dict(p, tokens=args.stub_tokens * (2 if p["tokens"] > 8192 else 1)) for p in PLAN["prompts"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="results folder (small files)")
    ap.add_argument("--capture", required=True, type=Path, help="capture folder (about 7.7 GB on the Pod)")
    ap.add_argument("--snapshot", type=Path, help="the pinned model snapshot folder (tokenizer)")
    ap.add_argument("--model-id", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    ap.add_argument("--revision", default="d4ab4f579dd3516f97d8a6a4c98d0653480bad15")
    ap.add_argument("--heads", type=int, default=PLAN["heads"])
    ap.add_argument("--kv-heads", type=int, default=PLAN["kv_heads"])
    ap.add_argument("--global-layers", type=int, default=PLAN["global_layers"])
    ap.add_argument("--rehearsal", action="store_true", help="the S5 rehearsal (a small model; not the frozen plan)")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-gpu", default="STUB NVIDIA H100 80GB HBM3")
    ap.add_argument("--stub-layers", type=int, default=2)
    ap.add_argument("--stub-tokens", type=int, default=640)
    ap.add_argument("--stub-q-rows", type=int, default=32)
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-text-root", type=Path, default=HERE.parent, help="text files for the stub prompts")
    ap.add_argument("--stub-exclude", type=Path, help="an exclusion file for the stub (default: e53_files.json)")
    ap.add_argument("--stub-w", type=float, default=0.0625, help="the scalar k_norm weight of the stub layers")
    ap.add_argument("--stub-chat-chars", type=int, default=200, help="characters per chat turn in the stub")
    # the E-53 stub flags that c53.stub_layer reads
    ap.add_argument("--stub-q-scale", type=float, default=2.0)
    ap.add_argument("--stub-q-max", type=float, default=0.0)
    ap.add_argument("--stub-big-k", type=float, default=0.0)
    ap.add_argument("--stub-big-k-dim", type=int, default=0)
    ap.add_argument("--stub-break-identity", action="store_true")
    ap.add_argument("--stub-swap-heads", action="store_true")
    ap.add_argument("--stub-nan", action="store_true")
    ap.add_argument("--stub-no-tensors", action="store_true", help="the first prompt records nothing (fail fast)")
    ap.add_argument("--stub-no-markers", action="store_true", help="a chat template without turn markers (A2 stop)")
    ap.add_argument("--stub-fail", action="store_true", help="an error after the engine start")
    args = ap.parse_args()
    out, cap_dir = args.out, args.capture
    out.mkdir(parents=True, exist_ok=True)
    cap_dir.mkdir(parents=True, exist_ok=True)
    plan = stub_plan(args) if args.stub else dict(PLAN, heads=args.heads, kv_heads=args.kv_heads,
                                                  global_layers=args.global_layers)
    shapes = (args.heads, args.kv_heads, c53.D)
    record = {"args": {k: str(v) for k, v in vars(args).items()}, "started_at": c53.now(), "status": "started",
              "stub": bool(args.stub), "rehearsal": bool(args.rehearsal), "model_id": args.model_id,
              "revision": args.revision, "plan": plan, "engine_kwargs": ENGINE, "prompts": [], "fail_fast": None}

    def save(status: str | None = None) -> None:
        if status:
            record["status"] = status
        (out / "capture_run.json").write_text(json.dumps(record, indent=1, default=str))

    def stop(problems: list[str]) -> None:
        record.update(fail_fast=problems, finished_at=c53.now())
        save("fail_fast")
        print(json.dumps({"status": "fail_fast", "problems": problems[:10]}), flush=True)
        sys.exit(3)

    try:
        t0 = time.time()
        if args.stub:
            record.update(gpu={"name": args.stub_gpu}, torch=torch.__version__, vllm="STUB")
            roots = {"vllm_py": [args.stub_text_root], "torch_py": [args.stub_text_root],
                     "docs": [args.stub_text_root]}
            base, encode, bos, render = args.stub_text_root, toy_encode, TOY_SPECIAL["<bos>"], toy_chat
            if args.stub_no_markers:
                render = lambda turns: "<bos>" + "".join(t["content"] for t in turns)   # noqa: E731
            special = frozenset(TOY_SPECIAL.values())
            exclude = load_exclude(args.stub_exclude or E53_FILES)
        else:
            record.update(gpu={"name": torch.cuda.get_device_name(0),
                               "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                               "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))},
                          torch=torch.__version__)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(args.snapshot))
            roots, base = c53.image_roots()
            encode, bos = (lambda s: tok.encode(s, add_special_tokens=False)), tok.bos_token_id
            render = lambda turns: tok.apply_chat_template(turns, tokenize=False)   # noqa: E731
            special = frozenset(int(i) for i in tok.all_special_ids)
            exclude = load_exclude()
            record["tokenizer"] = {"class": type(tok).__name__, "bos_token_id": bos, "special_ids": sorted(special)}
        record["exclude"] = {"paths": len(exclude["paths"]), "sha256": len(exclude["sha256"])}
        ppath = out / "prompts.json"
        prompts = None
        if ppath.exists():                                       # a rerun uses the same prompts
            old = json.loads(ppath.read_text())
            if old.get("plan") == record["plan"]:
                prompts = old["prompts"]
        if prompts is None:
            src = c53.sources(roots)
            record["source_files"] = {k: len(v) for k, v in src.items()}
            prompts = build_prompts(src, encode, bos, base, special, render, exclude, plan,
                                    chat_chars=args.stub_chat_chars if args.stub else CHAT_CHARS)
            ppath.write_text(json.dumps({"plan": record["plan"], "prompts": prompts}))
        record["prompts_built"] = [{k: p.get(k) for k in ("id", "kind", "family", "n_tokens", "token_sha256",
                                                          "bos_at_0", "special_after_0", "skipped_special",
                                                          "excluded", "turns")}
                                   | {"files": len(p["files"])} for p in prompts]
        record["phases"] = {"prompts": round(time.time() - t0, 1)}
        short = [p["id"] for p, s in zip(prompts, plan["prompts"]) if p["n_tokens"] != s["tokens"]]
        if short:
            raise RuntimeError(f"prompts {short} have fewer tokens than the plan")
        # (A2, D-92: no S5 rehearsal) the prompt rules of V2 before the engine starts, so a wrong chat template stops
        # the run within minutes: a chat prompt has BOS at 0 and 2 or more special ids after it, a plain prompt none
        bad = [f"chat prompt {p['id']}: BOS at 0 {p['bos_at_0']}, special ids after 0 {p['special_after_0']}"
               for p, s in zip(prompts, plan["prompts"])
               if s["kind"] == "chat" and not (p["bos_at_0"] and p["special_after_0"] >= 2)]
        bad += [f"plain prompt {p['id']}: {p['special_after_0']} special ids after position 0"
                for p, s in zip(prompts, plan["prompts"]) if s["kind"] == "plain" and p["special_after_0"]]
        if bad:
            stop(bad)
        save()

        if args.stub:
            expected = [5 + 6 * i for i in range(args.stub_layers)]
            record["config"] = {"global_layers": expected, "layer_types_count": len(expected),
                                "num_attention_heads": args.heads, "note": "stub"}
            gen = torch.Generator().manual_seed(args.stub_seed)
            w_l = torch.tensor(args.stub_w).to(torch.bfloat16).double()
            ws = {layer: torch.full((c53.D,), float(w_l), dtype=torch.float64) for layer in expected}
            knorm = {layer: w.float() for layer, w in ws.items()}
            record["hooks"] = {"route": "stub",
                               "hooked": [{"layer": layer, "name": f"stub.layers.{layer}.self_attn.attn",
                                           "num_heads": args.heads, "num_kv_heads": args.kv_heads, "head_size": c53.D,
                                           "head_size_v": c53.D, "sliding_window": None} for layer in expected]}
            if args.stub_fail:
                raise RuntimeError("stub failure (--stub-fail)")
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the session sets it too
            from vllm import LLM, SamplingParams
            import vllm

            record["vllm"] = vllm.__version__
            t = time.time()
            llm = LLM(model=args.model_id, revision=args.revision, **ENGINE)
            record["phases"]["engine"] = round(time.time() - t, 1)
            vc = llm.llm_engine.vllm_config
            record["engine"] = c53.engine_record(vc)
            try:
                record["engine"]["chunked_prefill"] = bool(vc.scheduler_config.enable_chunked_prefill)
            except Exception as exc:  # reported only
                record["engine"]["chunked_prefill_error"] = f"{type(exc).__name__}: {exc}"[:200]
            cfg = vc.model_config.hf_text_config
            layer_types = list(cfg.layer_types)
            expected = c53.global_layers(layer_types, int(cfg.num_hidden_layers),
                                         int(getattr(cfg, "num_kv_shared_layers", 0) or 0))
            record["config"] = {"global_layers": expected, "layer_types_count": len(layer_types)} | {
                k: c53.cfg_value(cfg, k, expected) for k in ("num_attention_heads", "num_key_value_heads",
                                                            "num_global_key_value_heads", "head_dim",
                                                            "global_head_dim", "attention_k_eq_v",
                                                            "num_kv_shared_layers", "rms_norm_eps")}
            store = c53.Store(plan["q_rows"])
            try:
                reg = llm.apply_model(lambda model: c53.register(model, store))[0]
                route = "LLM.apply_model"
            except Exception as exc:  # the attribute path, as E-53
                record["apply_model_error"] = f"{type(exc).__name__}: {exc}"[:300]
                model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
                reg, route = c53.register(model, store), "model_executor.driver_worker.worker.model_runner"
            record["hooks"] = {"route": route, "hooked": reg["hooked"]}
            knorm = reg["k_norm"]
            params = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
        torch.save(knorm, cap_dir / "k_norm.pt")
        record["k_norm_layers"] = sorted(knorm)
        bad = c53.hook_problems(record["hooks"]["hooked"], expected, shapes, plan["global_layers"])
        if bad:
            stop(bad)
        save()

        for p in prompts:
            length = p["n_tokens"]
            t = time.time()
            if args.stub:
                errors, chunks = [], {}
                data = {} if (args.stub_no_tensors and p["id"] == 0) else \
                    {layer: c53.stub_layer(length, plan["q_rows"], ws[layer], gen, args) for layer in expected}
                first = None
            else:
                store.begin(length)
                res = llm.generate([{"prompt_token_ids": p["token_ids"]}], params, use_tqdm=False)
                data, errors = store.end(), list(store.errors)
                chunks = {str(layer): len(part["k"]) for layer, part in store.parts.items()}   # one hook call each
                first = list(res[0].outputs[0].token_ids[:1])
            problems = c53.check_prompt(data, expected, length, plan["q_rows"], errors, shapes)
            entry = {"id": p["id"], "seconds": round(time.time() - t, 2), "first_token": first, "problems": problems,
                     "chunks": chunks}
            if problems and p["id"] == 0:
                record["prompts"].append(entry)
                stop(problems)
            entry["files"] = c53.save_prompt(cap_dir, p["id"], data)
            entry["seconds_with_save"] = round(time.time() - t, 2)
            record["prompts"].append(entry)
            save()
            print(json.dumps({"prompt": p["id"], "layers": len(data), "problems": len(problems),
                              "seconds": entry["seconds_with_save"]}), flush=True)
        record["phases"]["total"] = round(time.time() - t0, 1)
        record.update(finished_at=c53.now())
        save("ok")
    except SystemExit:
        raise
    except BaseException as exc:  # recorded, then re-raised so the session sees a non-zero exit
        record.update(error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-4000:],
                      finished_at=c53.now())
        save("error")
        raise
    print(json.dumps({"status": record["status"], "prompts": len(record["prompts"])}), flush=True)


if __name__ == "__main__":
    main()
