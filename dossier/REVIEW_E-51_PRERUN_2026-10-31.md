# R-69: pre-run review of E-51 (compact K=V decode attention kernel, I-182)

Date: 2026-10-31. Reviewer: Sonnet (a different model from the author).
VERDICT: AMEND_THEN_RUN. The kernels and the stock call are sound. The rule and the harness have 5 blockers (R1 to R5) and 3 plan fixes (R6 to R8).

Scope: `experiments/gemma4_kv_kernel/` at 0fb1b39, E-50 data, R-68, vLLM 3966924d5, vllm-flash-attention b273a02 (the pin) and 9ed6b12 (FA PR 199).
Labels: FACT (I read a source, ran a script or recomputed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).
I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I used read-only `gh` queries and one web search.
This Mac has no Triton and no GPU. I could not compile or time a kernel. Each statement about Triton compile, shared memory, registers and speed is INTERPRETATION.
Scratch files are in `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r69/`. The folder `patched/` there holds a copy of the code with my fixes.

## Verdict in short
- FACT: The 8 tests pass. The stock call matches vLLM's `forward()` at 3966924d5 argument by argument. The vendored `flash_attn_varlen_func` accepts every keyword.
- FACT: I ran the exact kernel source on the CPU through a small `tl` shim. All 8 split and tile combinations agree with the references (error 0.0021 at most). The runs cover both layouts, FP8 and BF16, a partial page and empty splits. The kernel logic is right.
- FACT: The four V3 targets, the kill lines and the go lines recompute from the E-50 files.
- FACT: The rule has five flaws that can give an invalid or a wrong decision (R1 to R5). I tested one fix for each on the copy.
- INTERPRETATION: GO needs a Triton compact kernel that reads 0.69 x copy on its own bytes (0.675 at 64K). That is the FA4 level per byte. Whether Triton reaches it is the open question.
- INTERPRETATION: The most decisive extra number is the stock K=V oracle (O1). It measures the time elasticity of the stock kernel to DRAM bytes. It has no Triton quality in it.

## Required fixes
Apply R1 to R8 as amendment A1. Port the scratch tests into `test_e51.py`, rerun S3, then go to S4. R1 to R5 block S6.

### R1 (blocker): dt must use one configuration at both lengths
- FACT: `family()` takes the fastest row at 2K and the fastest row at X separately. The engine uses one split count at all lengths. V3 ties only `stock_fp8_s32`.
- INTERPRETATION: At 2K the best stock row can be `auto` (1 split at B 19, so no 32-split partial output). The best Triton row at 2K can use fewer splits than the row at X. Then dt contains the difference of two fixed costs. This bias has no fixed sign.
- FACT (with my assumed numbers): Let `auto` be 88 us below `s32` at 2K. Let the compact 2K row be 40 us below its matched row. Then sigma_mma moves from 0.265 to 0.279. The GO corridor at equal efficiency is only 0.25 to 0.266.
- Fix (tested), `analyze_e51.py`:
```python
def family(rows, kernels, X):
    b = GROUPS[X]
    hi = best(rows, kernels, b, X)                      # fastest configuration at X
    lo = next((r for r in rows if hi and r["kernel"] == hi["kernel"] and r["config"] == hi["config"]
               and r["batch"] == b and r["context"] == 2048 and r.get("status") == "ok"), None)
    return {"dt_us": round(hi["us"] - lo["us"], 2), "t2k": lo, "tX": hi} if hi and lo else None
```

### R2 (blocker): a KILL must not depend on a defect of the compact kernel
- FACT: In a memory-bound pair, sigma_layout = 1 - 0.625 x (std rate / compact rate). It also falls when the compact kernel is slow for its own reasons.
- INTERPRETATION: The compact kernel has 1.125 x the MMA work, a second load stream and a second MMA operand (`qr`). Expect register spills, mostly at 4 warps. The operand q alone takes about 128 registers per thread.
- FACT (stub grid): std 0.70 x copy with compact 0.49 gives KILL (sigma_layout 0.106). std 0.55 with compact 0.38 also gives KILL. The only guard is "std >= 0.5 x copy".
- Fix: add oracle rows at the 4 group cells. The oracle is the std Triton kernel with `cache_k=cache_v`. It has the same on-chip work and half the DRAM bytes. Use it for timing only.
- A KILL then needs, in every group, `cmp_memory_bound` (>= 0.6 x copy on dt, the E-50 M1 bar). Or it needs `sigma_alias` below `KILL / 0.375 * 0.5` (0.207 at 32K, 0.167 at 64K). Otherwise the result is INCONCLUSIVE.
- Tested on stub data: equal efficiency gives GO. Compact 0.49 gives INCONCLUSIVE. Compact 0.62 gives PIVOT. A family that is on-chip-bound (alias gain 0) gives KILL.
- The test `test_kill_when_the_layout_gives_nothing` encodes the defect case. Change its expectation to INCONCLUSIVE. Add a test for the on-chip-bound KILL.
```python
# group(): after std_bw
cmp_bw = grow(True) / (cmp_["dt_us"] * 1e-6) / 1e9
alias = family(rows, (f"triton_std_{sfx}_alias",), X)
sigma_alias = round(1 - alias["dt_us"] / std["dt_us"], 4) if alias and alias["dt_us"] > 0 else None
# add to the returned dict: "cmp_memory_bound": bw is not None and cmp_bw >= 0.6 * bw, "sigma_alias": sigma_alias
# decide(): replace the KILL branch
if all(v["sigma_layout"] < KILL[int(X)] for X, v in g.items()):
    proof = all(v["cmp_memory_bound"] or (v["sigma_alias"] is not None and v["sigma_alias"] < KILL[int(X)] / 0.375 * 0.5)
                for X, v in g.items())
    return f"KILL ({tag})" if proof else f"INCONCLUSIVE (low sigma_layout, no proof that the layout is the cause; {tag})"
# kbench_e51.py: TRITON["triton_std_fp8_alias"] = ("fp8", False). In bench_cell, skip it unless (b, length) in GROUP_CELLS
# and pass cache_k=vc if alias else kc. Skip it in correctness(). It is not in DECIDE_KERNELS, so V2 and V5 ignore it.
```

### R3 (blocker): V4 must test the deciding rows
- FACT: The README says "every deciding measurement". The code tests all 340 rows of the four deciding kernels (10 + 10 + 160 + 160). The spread is max / min - 1 over 15 replays.
- INTERPRETATION: One outlier replay in one slow, non-winning configuration voids the run. I have no replay-spread data from an H100, so I cannot give the odds.
- FACT (stub): One non-winning row with 6% spread gives INVALID (V4) now and GO after the fix.
```python
def decision_rows(rows):                       # the 12 rows that enter dt
    out = []
    for X in GROUPS:
        for fam in (("stock_fp8_s32", "stock_fp8_auto"), ("triton_std_fp8",), ("triton_cmp_fp8",)):
            f = family(rows, fam, X)
            out += [f["t2k"], f["tX"]] if f else []
    return out
# checks(): "V4": bool(dec) and all(r["spread"] <= SPREAD_MAX for r in dec),  with dec = decision_rows(rows)
```

### R4 (blocker): run the deciding cells first and keep the decision after a timeout
- FACT: `CELLS` puts 19:32768 sixth and 11:65536 last. A timeout at 1,800 s leaves no `run.json`, so the analysis gives INCONCLUSIVE.
- INTERPRETATION: My time model gives 9 to 30 min for the benchmark with JIT (answer 6). The slow end is near the timeout.
- FACT (stub): A run cut after the four group cells gives INCONCLUSIVE now. It gives the decision after the fix.
```python
CELLS = "19:2048,19:32768,11:2048,11:65536,19:8192,11:8192,11:32768,4:2048,4:32768,4:65536"
GROUP_CELLS = {(19, 2048), (19, 32768), (11, 2048), (11, 65536)}
# main(), after each cell:
if GROUP_CELLS <= {(r["batch"], r["context"]) for r in result["cells"]}:
    (out / "run.json").write_text(json.dumps(record | {"status": "ok_groups", "errors": errors[-50:]}, indent=1))
# analyze_e51.py: OK_STATUS = ("ok", "ok_groups"); decide(): if s["run_status"] not in OK_STATUS: ...
```
The resume test in `session_e51.sh` (`grep '"status": "ok"'`) does not match `ok_groups`. A rerun then starts again, which is right.

### R5 (blocker): validate every swept configuration
- FACT: V2 checks one configuration (splits 8, warps 8, tile 32) at B 2 and L 3,000. The timing winner can be any of 16 configurations per family.
- INTERPRETATION: A miscompile for one constexpr combination is rare, but a wrong kernel is often fast. The best-per-cell rule would pick it.
- Fix (tested under the CPU shim): Loop `correctness()` over `itertools.product(SPLITS, WARPS, TILES)`. Store the error or the exception text in `entry[kname]["by_config"]["splits/warps/tile"]`. Keep `err` for 8/8/32.
- In the analysis, drop a Triton row before `best()` runs when its `by_config` value is missing or above `TOL_CORRECT` (`validated(rows, corr)`). The cost is 64 launches at B 2.
- After the rehearsal, tighten the Triton bound to about 0.006. The observed error is 0.002, and 0.02 cannot see a dropped page.

### R6: fail fast after the correctness cell
- FACT: The stock FA4 FP8 call cannot run on the 4090. Its first real run is on the H100.
- FACT: If a deciding kernel fails there, V2 makes the run INVALID. The sweep still runs for 6 to 25 min.
```python
bad = [k for k in ("stock_fp8_s32", "triton_std_fp8", "triton_cmp_fp8") if "err" not in result["correctness"]["fp8"].get(k, {})]
if bad:
    raise RuntimeError(f"fail fast: {bad} did not run in the correctness cell")   # run.json gets status error
```

### R7: align the caps with the $3 cap
- FACT: The Pod bills from creation. In E-50, an 11 min pull plus a 31 min session cost $2.92. At $3.99/h a $3 cap is 45 min of Pod life.
- FACT: The boot watchdog counts from container start (45 min). With an 11 min pull the Pod then lives 56 min, which costs $3.72.
- FACT: A run that hits the timeout takes about 11 + 3 (setup) + 30 + 1 + 3 (AUTO_STOP delay) = 48 min. That costs $3.19.
- Fix: Set `KB_TIMEOUT` to 1,500 s and the boot watchdog CAP to 2,040 s (34 min after start). Or ask the user at S4 to approve a $3.75 cap. The user decides.

### R8: fix the rehearsal gate
- INTERPRETATION: VERIFY 2 asks for a passing configuration per kernel and cell on the 4090. Std BF16 needs 128 KB of pipelined buffers at tile 32 and 2 stages. The SM89 limit is 99 KB. So std BF16 likely has no configuration there. This says nothing about the H100.
- Fix: Add `--stages` (default 2) to `kbench_e51.py` and pass it as `num_stages`. Use `--stages 1` for BF16 on the 4090.
- Also set `KB_TIMEOUT` to 900 and use the four group cells. Then the session ends before the boot watchdog, and AUTO_STOP is tested (VERIFY 9).

## Optional items
- O1 (high value): Add the stock K=V oracle. Time `stock_fp8_s32` with `k=v` (one tensor) at the 4 group cells.
  - Add a read-only control. The aliased read must take 0.6 x the time of the separate read. Otherwise the premise fails (HYPOTHESIS: L2 absorbs the second read).
  - Then e = sigma_alias / 0.5 is the elasticity. The layout reaches the kill line when e >= 0.41 (32K) or 0.33 (64K). E-50 saw 0.49 across two kernels.
- O2 (high value): Measure FA PR 199 directly.
  - FACT: PR 199 changes only `flash_attn/cute/flash_fwd_sm90.py`, `interface.py` and `named_barrier.py`. Commit 9ed6b12 has b273a02 as its only parent.
  - Copy the three files into `vllm/vllm_flash_attn/cute/`. Replace `flash_attn.cute` by `vllm.vllm_flash_attn.cute` (the cmake rule). No CUDA build is needed. The CuTe JIT runs at the first call.
  - FACT: The PR 60372 patch changes no argument of the `_flash_attn_fwd` call, so `stock_call` works unchanged. The kernel needs `nvidia-cutlass-dsl>=4.6.2`. The image version is not recorded.
  - Run it in a second process after the frozen run. A failure then cannot touch the decision. The result replaces the 0.85 allowance by a measured dt (exploratory unless pre-registered). The 4090 cannot rehearse it.
- O3: Re-measure the 12 winning rows 5 times, interleaved, with a new graph each time (order B A A B). Report sigma with its range. Add an A/A pair (std against std). `MEASUREMENT.md` Section 1.5 asks for 5 repeats.
- O4: Report sigma_layout at the best common configuration of std and compact. Tile 64 may fail for one family only.
- O5: Store regs, spills and shared bytes per Triton configuration (the compiled-kernel handle that the launch returns). Sweep `--stages` 2 and 3 on the H100 if the time allows.
- O6: Cut cost. BF16 Triton is 65% of the Triton timing and 32 of 68 compiles, and it is secondary. Run BF16 only at the 4 group cells with 4 configurations. Make `SPLITS` a runtime argument of the split kernel (4 x fewer compiles). The merge kernel needs it as constexpr.
- O7: Raise fidelity. Allocate the stock caches as one `[blocks, H, N, 2D]` tensor. Use `transpose(1, 2).split(512, -1)` for K and V, as the engine does. Time 24 and 16 graph slots (R-68 recommendation 2).
- O8: Record the `pip list` lines for `nvidia-cutlass-dsl`, `quack-kernels` and `triton` in `versions.txt`.
- O9: Minor items.
  - The README promises an 8K linearity report. `analyze_e51.py` has no code for it.
  - The README says the caches are much larger than L2. B 4 at 2K is 33.6 MB (std FP8) and 21 MB (compact FP8). B 11 at 2K compact FP8 is 57.7 MB against the 50 MB L2.
  - INTERPRETATION: The effect on dt is under 1%.

## Answers to the eight questions
### 1. Kernels (`compact_attn.py`)
- FACT: There is no int32 overflow. The page id becomes int64 before the stride product.
- FACT: The largest cache has 1.85e9 elements (compact, B 11, 64K). That is below 2^31.
- FACT (shim): The masks, the split rule and the merge are right. The scales and the FP8 cast are right. So are the empty splits (L 130 and 64 at 32 splits).
- FACT: The first tile of a split always has a valid token, so no NaN occurs. Only a sequence of length 0 gives NaN. No cell has one.
- INTERPRETATION: The Triton code is legal. The dot shapes are M 16, N 32 or 64, and K 512, 128 or the tile. The `COMPACT` branch is constexpr. The FP8 to BF16 cast exists on SM89 and SM90.
- INTERPRETATION: The pipelined operand buffers at 2 stages need 256 KB for std BF16 tile 64 (fails on the H100, limit 227 KB). Std BF16 tile 32 and std FP8 tile 64 need 128 KB. Std FP8 tile 32 needs 64 KB. Compact needs 40 to 160 KB. The q copy and the cast scratch come on top.
- INTERPRETATION: On the 4090 (99 KB) most BF16 and tile 64 configurations fail. On the H100 the tile 32 configurations should run. Each kernel and cell then keeps a configuration.
### 2. Performance
- FACT: Wave efficiency at 1 CTA per SM is 0.97 at 32 splits for both pairs. At 8 or 16 splits it is 0.92 (B 19) and 0.89 (B 11). At 4 splits it is 0.77 and 0.67.
- FACT: The compact row (V 512 B, then K_rot 128 B) is 128-byte aligned for every head. The two column views waste no sector.
- FACT: The partial output (40 MB at B 19) is equal for both layouts. It cancels in dt after R1.
- INTERPRETATION: Risks for "not memory-bound" are one CTA per SM at 8 warps, fixed 2 stages, and a cast per element. The operand registers are about 128 (std) or 160 (compact), plus 32 to 64 for `acc`.
- INTERPRETATION: Both kernels move the same data on chip per token. V is used twice, or K and V once each. Only the DRAM bytes and the cast count fall.
- INTERPRETATION: So sigma_layout is fair only if both kernels are DRAM-bound (R2, O1).
### 3. Stock call
- FACT: Every argument matches the engine. These are q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, scale, causal, window [-1, -1], block_table, softcap 0 and fa_version 4. Also q_descale None, k_descale and v_descale `(B, 4)` float32 contiguous, and num_splits. Five more arguments are None: alibi_slopes, dynamic_causal, s_aux, mask_mod and aux_tensors.
- FACT: The engine also passes `scheduler_metadata`. The FA4 branch of the wrapper drops it. vLLM owns `vllm/vllm_flash_attn/flash_attn_interface.py`. The vendored repo file has no FA4.
- FACT: For head dim 512 on SM90, `tile_n` is 64. FP8 dequant asserts page == tile_n, and page 64 passes. BF16 page 64 also takes the TMA path.
- INTERPRETATION: The engine's BF16 page 32 took the non-TMA path. So the stock BF16 rows here are not the E-50 BF16 kernel.
- FACT: The FA4 interface has no host sync. `num_splits=0` uses `min(132 // (B x 4), 128, n_blocks)`. That gives 1 split at B 19 and 3 at B 11. INTERPRETATION: graph capture is safe.
- HYPOTHESIS: Two differences can move V3 by a few percent. The caches are separate contiguous tensors here and strided halves in the engine. The batch has 19 and 11 entries here and 24 and 16 graph slots in the engine.
### 4. Correctness cell and V2
- FACT: The references use the dequantized stored values. The compact reference uses qc, qr and the stored rows.
- FACT (shim): The kernel error is 0.0019 to 0.0021. The FP8 storage error against the exact attention is 0.023 to 0.036 for both layouts.
- FACT: FP8 CPU conversion works (torch 2.15 here). INTERPRETATION: the Pod has torch 2.14 with the same conversions.
- The bound 0.02 is too loose (R5).
### 5. Analysis and rule
- FACT: S per sigma is 0.3227 (32K) and 0.4 (64K). The lines 0.155, 0.125 and 0.25 (S 8.1% and 10.0%) recompute.
- FACT: The 15% allowance is above the PR 199 table (1.078 to 1.155 x, that is 7% to 13% less time). It is far above the 0% to 1% end-to-end gain in the PR 60372 test. So sigma_mma is conservative.
- FACT: V5 cannot fail below 3,502 GB/s, which is above the H100 peak. It is weak but harmless.
- FACT: V3 uses the 2K cells, where fixed costs dominate (238 us for 159 MB).
- Wrong decisions: R1 (bias to GO), R2 (false KILL), R3 (false INVALID), R4 (INCONCLUSIVE after a timeout).
### 6. Budget and time
- FACT (my model, 363 call executions per configuration): The sweep GPU time depends on the Triton data rate. It is 3.7 min at 2.1 TB/s, 9.1 min at 0.8 TB/s and 17.8 min at 0.4 TB/s. The stock rows add 12 s.
- HYPOTHESIS: 68 Triton compiles of 3 to 6 s take 4 to 7 min. About 10 FA4 variants of 10 to 30 s take 2 to 5 min. The E-50 log shows 15 s for the first FA4 variants.
- So the total is 9 to 30 min against the README's 10 to 15 min. The cost is in R7.
### 7. Rehearsal
- See the VERIFY list. The 4090 cannot run FA4 FP8 (SM90 only). The stock call, V3 and the H100 times first appear on the H100. R6 limits the loss.
### 8. Hostile questions
- A Triton kernel is a fair stand-in for the layout effect only with R2 and O1. It is not a stand-in for FA4 quality.
- A cheaper and more decisive design exists: O1.
- The FP8 MMA kernel can be measured with no long compile: O2.
- Prior work since 2026-10-31: none found (see the checks).

## Checks I ran
- FACT: `uv run ... pytest test_e51.py` gave 8 passed in 3.5 s.
- FACT: I emulated both kernels on the CPU (`tl_shim.py`, `run_shim.py`). The run has 6 cases (B 1 to 3; L 64, 130, 2,048, 2,049, 3,000), 8 split and tile combinations and 2 layouts. FP8 ran in five cases and BF16 in one. The worst error is 0.0021.
- FACT: The V3 targets come from `m31f8/kernels.json` (main plus combine, in us). They are 225.93 + 12.41 = 238.34, 1,341.10 + 17.04 = 1,358.14, 135.76 + 7.45 = 143.20 and 1,496.55 + 10.64 = 1,507.19.
- FACT: The growth is 1,119.8 and 1,364.0 us, that is 2,135 and 2,097 GB/s (0.70 and 0.69 x copy). The kill line is 0.52 x copy and the go line is 0.69 x copy at 32K.
- FACT: I read `flash_attn.py` and `vllm/vllm_flash_attn/flash_attn_interface.py` at 3966924d5, and `flash_attn/cute/interface.py` at b273a02. I also read the FA PR 199 patch and the vLLM PR 60372 patch.
- FACT: Stub grids (`grid.py`, `grid2.py`, `scen3.py`, `scen4.py`) test the decisions in R1 to R5. On the patched copy 7 of 8 tests pass. The eighth encodes the old KILL.
- FACT: GitHub PR searches (5 queries, updated since 2026-10-25) found vLLM #41944 (drop the V projection; open), #61267 (LoRA fix for `attention_k_eq_v`) and transformers #50189. None stores the K=V cache once.
- FACT: One web search found only cross-layer KV sharing and LMCache recipes. This is not proof of absence.

## VERIFY list for the rehearsal (S5, RTX 4090)
1. The image starts. `import triton, vllm` works. `versions.txt` shows the Triton and `nvidia-cutlass-dsl` versions (O8).
2. Triton accepts `compact_attn.py` (string annotations from `from __future__ import annotations`, constexpr branches). Each FP8 kernel has a passing configuration at stages 2. Each BF16 kernel has one at `--stages 1` (R8). `kbench.json` holds the reason of every failure.
3. Record regs, spills and shared bytes for tile 32 at 4 and 8 warps, both layouts (O5). Spills in the compact kernel only mean a biased sigma_layout. Fix this before the H100.
4. Run the four group cells at full size (peak memory about 9.6 GB). The dt rate of `triton_std_fp8` must reach 0.5 x the 4090 copy rate. If it does not, stop and fix the kernel. The H100 would say INCONCLUSIVE.
5. The oracle rows exist at the group cells. The aliased std kernel is faster than the std kernel (sigma_alias above 0.2 on this memory-bound GPU). Otherwise the oracle premise is weak (O1).
6. `run.json` shows `ok_groups` after the fourth cell and `ok` at the end. Run once with `KB_TIMEOUT` 600. The analysis must still decide.
7. The stock rows have status "error" with the FA4 reason. Fail fast (R6) ends the session in minutes. The analysis prints the expected INVALID or INCONCLUSIVE.
8. Print the largest V4 spread on the 12 decision rows and the share of all rows above 2%. If many rows exceed 2%, change the timing before the H100.
9. `AUTO_STOP` stops the Pod before the boot watchdog fires (`KB_TIMEOUT` 900, CAP 1,800 s).
10. A rerun skips a finished benchmark. A failed run keeps its partial data. Record the Triton times for the H100 time model.
- Only the H100 can show the FA4 stock call (JIT, graph capture, page 64, split combine) and V3. It also shows Triton on SM90 (227 KB shared memory, 8-warp registers, speed). It shows the copy rate (V5), the power cap (V6) and the PR 199 overlay (O2).

## What the design gets right
- The stock call is the engine call. Its arguments, page size and FP8 path were checked against the source, and V3 ties it to E-50.
- dt removes the fixed split, partial-output and launch costs. It maps to the E-50 growth: 11.2 ms and 13.6 ms per step at 10 global layers, equal to `b x (X - 2K)`.
- One Triton code family holds both layouts, so the layout effect is measured inside the family.
- The correctness cell has a partial page, a float64 exact reference and a reference from the stored values.
- The rule is conservative against GO: the 15% allowance and the extra 128-dimension QK work.
- The safety nets work: per-configuration try/except, timeouts, boot watchdog, AUTO_STOP, resume logic and a tested stub session.
- The kill and go lines, the S mapping and the V3 targets are correct.
