"""C1, C3, C4, C5 of RESEARCH_JOURNAL.md E-23: end-to-end pilot on a small dense model.

Reference backend: CPU fp32. Candidate backend: MPS fp16 (or CPU bf16 as the predeclared
fallback). Batch size 1, exact greedy decoding, logits taken on CPU float32.

Per prompt (all logits are recomputed, nothing is kept across prompts to bound memory):
  ref_free   reference free run           -> tokens R, logits LR1
  ref_tf     reference teacher forced R   -> logits LR2   (C1 reference: LR1 == LR2 bitwise)
  cand_free  candidate free run           -> tokens Cf     (first divergence vs R)
  cand_tf    candidate teacher forced R   -> logits LC2   (decode-mode flips vs R)
  cand_tf2   candidate teacher forced R   -> logits LC3   (C1 candidate: LC2 == LC3 bitwise)
  prefill    one parallel pass, both backends             (prefill-mode flips, reported only)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness import contracts as C, estimator as E, manifest as MF, modelrun as M  # noqa: E402

CANDIDATES = {
    "mps_fp16": ("mps", torch.float16),
    "cpu_bf16": ("cpu", torch.bfloat16),
}


def run_prompt(i, prompt_ids, steps, ref, cand, cand_dev, out_dir: Path):
    t0 = time.time()
    R, LR1 = M.decode_run(ref, "cpu", prompt_ids, steps)
    _, LR2 = M.decode_run(ref, "cpu", prompt_ids, steps, forced=R)
    Cf, _ = M.decode_run(cand, cand_dev, prompt_ids, steps)
    _, LC2 = M.decode_run(cand, cand_dev, prompt_ids, steps, forced=R)
    _, LC3 = M.decode_run(cand, cand_dev, prompt_ids, steps, forced=R)
    LRp = M.prefill_forced_logits(ref, "cpu", prompt_ids, R)
    LCp = M.prefill_forced_logits(cand, cand_dev, prompt_ids, R)

    Rt = torch.tensor(R)
    margins = C.top2_margin(LR1)
    flip_dec = LC2.argmax(-1) != Rt
    flip_pre = LCp.argmax(-1) != Rt
    first_div_free = next((t for t in range(steps) if Cf[t] != R[t]), None)
    rec = {
        "i": i,
        "c1_ref_diff": C.bitwise_diff_count(LR1, LR2),
        "c1_cand_diff": C.bitwise_diff_count(LC2, LC3),
        "nonfinite_ref": int((~torch.isfinite(LR1)).sum()),
        "nonfinite_cand": int((~torch.isfinite(LC2)).sum()),
        "first_div_free": first_div_free,
        "first_flip_dec": C.first_true(flip_dec),
        "first_flip_pre": C.first_true(flip_pre),
        "seconds": round(time.time() - t0, 1),
    }
    np.savez_compressed(
        out_dir / f"{i:03d}.npz",
        margins=margins.numpy(), flip_dec=flip_dec.numpy(), flip_pre=flip_pre.numpy(),
        kl=C.kl_ref_to_cand(LR1, LC2).numpy(), maxdiff=C.max_abs_logit_diff(LR1, LC2).numpy(),
        ref_tokens=np.array(R), cand_free_tokens=np.array(Cf),
    )
    (out_dir / f"{i:03d}.json").write_text(json.dumps(rec))
    return rec


def aggregate(out_dir: Path, args) -> dict:
    recs, marg, fdec, fpre, kl, md = [], [], [], [], [], []
    for jf in sorted(out_dir.glob("[0-9][0-9][0-9].json")):
        recs.append(json.loads(jf.read_text()))
        z = np.load(jf.with_suffix(".npz"))
        marg.append(z["margins"]); fdec.append(z["flip_dec"]); fpre.append(z["flip_pre"])
        kl.append(z["kl"]); md.append(z["maxdiff"])
    marg, fdec, fpre = np.concatenate(marg), np.concatenate(fdec), np.concatenate(fpre)
    kl, md = np.concatenate(kl), np.concatenate(md)

    c1_ref = all(r["c1_ref_diff"] == 0 for r in recs)
    c1_cand = all(r["c1_cand_diff"] == 0 for r in recs)
    c3_agree = [r["first_div_free"] == r["first_flip_dec"] for r in recs]
    pre_agree = [r["first_flip_pre"] == r["first_flip_dec"] for r in recs]
    cov = E.coverage_experiment(marg, fdec, budget_frac=0.25, n_rep=args.reps, seed=0,
                                interval=args.interval)
    if cov["n_flips"] < 15:
        c4 = "INCONCLUSIVE"
    else:
        c4 = "PASS" if cov["coverage_stratified"] >= 0.90 else "FAIL"

    def q(x):
        return {f"p{p}": float(np.percentile(x, p)) for p in (50, 90, 99, 100)}

    return {
        "n_prompts": len(recs),
        "steps": args.steps,
        "n_positions": int(len(marg)),
        "C1": {"reference_bitwise_stable": c1_ref, "candidate_bitwise_stable": c1_cand,
               "PASS": bool(c1_ref and c1_cand),
               "nonfinite_ref": sum(r["nonfinite_ref"] for r in recs),
               "nonfinite_cand": sum(r["nonfinite_cand"] for r in recs)},
        "C3": {"prompts_agree": int(sum(c3_agree)), "of": len(recs), "PASS": bool(all(c3_agree)),
               "prompts_diverging_free_run": int(sum(r["first_div_free"] is not None for r in recs)),
               "prefill_vs_decode_first_flip_agreement": float(np.mean(pre_agree))},
        "C4": {"verdict": c4, **cov},
        "C5": {"flip_rate_decode": float(fdec.mean()), "flip_rate_prefill": float(fpre.mean()),
               "kl_mean": float(kl.mean()), "kl_quantiles": q(kl),
               "max_abs_logit_diff_quantiles": q(md),
               "margin_quantiles": q(marg)},
        "seconds_per_prompt_mean": float(np.mean([r["seconds"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--candidate", default="mps_fp16", choices=list(CANDIDATES))
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--interval", default="jeffreys_rate", choices=["jeffreys_rate", "beta_binomial"])
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out) / "per_prompt"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        from transformers import AutoTokenizer
        cand_dev, cand_dtype = CANDIDATES[args.candidate]
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = json.loads((ROOT / args.prompts).read_text())[: args.n_prompts]
        print("loading models", flush=True)
        ref = M.load(args.model, "cpu", torch.float32)
        cand = M.load(args.model, cand_dev, cand_dtype)
        MF.write_json(Path(args.out) / "manifest.json", MF.build_manifest(ROOT, {
            "experiment": "E-23/model_pilot", "model": args.model, "reference": "cpu_fp32",
            "candidate": args.candidate, "n_prompts": args.n_prompts, "steps": args.steps,
            "prompt_tokens": args.prompt_tokens, "batch_size": 1, "decoding": "greedy",
            "prompts_file": args.prompts, "prompts_file_sha256": MF.sha256_file(ROOT / args.prompts),
            "interval": args.interval,
        }))
        for i, text in enumerate(prompts):
            if (out_dir / f"{i:03d}.json").exists():
                continue
            ids = tok(text)["input_ids"][: args.prompt_tokens]
            rec = run_prompt(i, ids, args.steps, ref, cand, cand_dev, out_dir)
            print(f"prompt {i:02d}/{len(prompts)}  {rec['seconds']}s  c1=({rec['c1_ref_diff']},{rec['c1_cand_diff']})  "
                  f"div_free={rec['first_div_free']} flip_dec={rec['first_flip_dec']} flip_pre={rec['first_flip_pre']}", flush=True)

    result = aggregate(out_dir, args)
    MF.write_json(Path(args.out) / "pilot_result.json", result)
    print(json.dumps({k: result[k] for k in ("C1", "C3")}, indent=1))
    print("C4:", result["C4"]["verdict"], "coverage", result["C4"]["coverage_stratified"],
          "flips", result["C4"]["n_flips"], "of", result["C4"]["n_positions"])


if __name__ == "__main__":
    main()
