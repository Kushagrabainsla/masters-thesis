# I-41 Ingest structurer: a small LLM that compresses prompts and tool outputs before the main model

Date: 2026-10-04. Journal IDs: I-41 (with narrowed variants I-42 to I-45), E-24, E-25, R-08.
Workflow: `.claude/skills/research-gap-dossier`. Labels: FACT (a source or a run showed it),
INTERPRETATION, HYPOTHESIS. Source IDs [S..] are in Section 16.

## 1. One-sentence idea

Put a small, cheap LLM in front of the main agent model. It rewrites each new user prompt and each
tool output into a terse, structured form (facts) before the main model sees it. The main model's
context then grows more slowly and stays append-only, so compaction happens later and the prompt
cache prefix survives longer.

## 2. Why this problem exists

- Agent loops resend the whole history every turn. Providers bill a cached prefix at 0.05x to 0.1x
  of the input price and new content at 1.25x (5 min cache) or 2x (1 h cache) [S13]. A token
  written early is paid once on write and again, at the read rate, on every later turn.
- Prompt caching matches an exact prefix. Any rewrite of the middle of the context (compaction,
  clearing old tool results) invalidates everything after it [S12, S13].
- Context quality degrades as context grows, so harnesses compact or clear (journal Section 7.8,
  C6). Seven of eleven production coding harnesses use threshold-triggered LLM compaction [S34].

## 3. Research question (as first posed)

Can compressing prompts and tool outputs at ingest with a small model, while keeping the main
context append-only, reduce total cost and delay compaction compared with raw history plus
threshold compaction, without lowering task success? It is "no" if (a) compaction is not what
breaks the cache in real use, (b) a perfect free compressor saves little, or (c) the compressor
destroys information the agent needs later.

## 4. Why this was a valid question

The cost argument is plausible from first principles (a token ingested early is re-read many
times) and the cache argument is plausible from prefix-match semantics. What was unknown before
this session: how often compaction actually happens in real use, what fraction of cost tool
outputs drive, and how much of a tool output the agent needs verbatim later.

## 5. What existing research and shipping systems already solve

| Work | What it solves | What remains outside its scope |
|---|---|---|
| ACON, 2510.00615 [S1] | LLM compresses each observation over 1,024 tokens at ingest, conditioned on history, and compresses history over 4,096 tokens. Compressor distilled into Qwen3-8B/14B and Phi-4 keeping about 95% of teacher accuracy. 26 to 54% lower peak tokens. Wall time rose from 73 s to 102 s in one setting | No prompt-cache accounting. History compression rewrites the prefix |
| TokenPilot, 2606.17016, EMNLP 2026 Findings [S2] | "Ingestion-aware compaction" of tool outputs at entry (rule-based), stable placeholders for a byte-identical prefix, lifecycle-aware eviction by a Qwen3.5-35B estimator batched every B = 3 turns to limit cache breaks. 61% and 87% cost reduction on PinchBench and Claw-Eval | Rule-based at ingest, not an LLM structurer. Does not measure prefix lifetime or compactions avoided |
| AgentDiet, 2509.23586 [S3] | A cheaper "reflection" LLM removes useless, redundant and expired content, editing only step s-a so earlier KV cache entries survive. 40 to 60% fewer input tokens, 21 to 36% lower cost | Not structured facts. Coding benchmarks only |
| SWE-Pruner, 2601.16746 [S5]; Squeez, 2604.04979 [S4]; SWE-Pruner Pro, 2607.18213 [S6] | Small models (0.6B, 2B) prune tool outputs line by line, conditioned on a goal the agent states, keeping verbatim lines. 23 to 54% fewer tokens with higher success (SWE-Pruner); 92% removal at 0.86 recall (Squeez) | Query-conditioned, so the agent must state a goal at read time |
| Claude Code WebFetch [S32] | A Haiku-class model reads the fetched page with the main model's prompt and returns only the answer | One tool only (search-only source) |
| Anthropic context editing and compaction [S12, S14] | Server-side clearing of old tool results (`keep`, `clear_at_least` to make a cache break worth it) and server-side compaction at a token threshold | Clearing invalidates the cache from the cleared point (official docs) |
| OpenAI compaction endpoint [S15]; Manus [S17] | Encrypted server-side compaction; append-only context, restorable compression via the file system | Not evaluated publicly |
| Don't Break the Cache, 2601.06007 [S8]; CAPC, 2607.15516 [S7]; Keepalive Economics, 2607.19214 [S9] | Cache strategy for agents (41 to 80% cost cut), a two-tier cost model for compression plus caching, keepalive pings against TTL expiry | Not an ingest structurer |
| TraceLab, 2606.30560 [S10]; claude-code issue #94177 [S11] | 4,265 Claude Code and Codex sessions: 95.7% prefix hit rate, misses mostly after human idle gaps, compaction in 4.5% of Claude sessions. One user's 30 sessions: 43% of cache writes from TTL expiry, 14% from mid-history pruning | Neither measures how often tool output is later needed verbatim |
| Complexity Trap, 2508.21433 [S18]; 2608.16370 [S19] | Observation masking matches LLM summarization at about half the cost, and summaries lengthen trajectories. Compression can triple re-retrieval calls without changing task success | Not cache-aware |
| PatchKV 2609.26219 [S23], KVEraser [S24], CacheBlend [S25], Still [S26], AgentKV [S27] | Serving-side reuse of KV after a middle-span edit, learned KV compaction, phase-aware KV eviction for agents | The infrastructure-level version of the idea is already being built |

## 6. Exact surviving gap

INTERPRETATION. The mechanism (small model compresses tool outputs at ingest, append-only, cache
aware) is published (ACON, TokenPilot, AgentDiet, SWE-Pruner, Squeez) and shipping in pieces
(WebFetch, context editing, server compaction). The cache-lifetime premise is contradicted by E-24,
TraceLab and #94177. What I did not find in public work:

- A measurement of how often coding agents later copy tool-output text verbatim, and how far back
  the source sits. E-24 measured it on one user's traces (Section 9). TraceLab does not report it.

Do not claim as novel: compress-on-ingest, small-model compressors, cache-aware eviction, batched
clearing, cache-break attribution, or the TTL finding.

## 7. Proposed mechanism or investigation

No mechanism is justified. The only open item is a measurement: run the E-24 verbatim analysis on
TraceLab's public traces (Section 10). If it holds at scale it supports a short note: clearing that
keeps the last K tool results, and query-agnostic rewriting, remove the source of a large share of
later edits.

## 8. Strongest adversarial objections

1. Reviewer one-liner: "This is ACON's observation compression with TokenPilot's cache-continuity
   rules." True for the mechanism.
2. The cache premise is wrong for interactive use. Cache misses come from idle gaps beyond the
   TTL, not compaction (E-24 P1, TraceLab, #94177). Fix for that is keepalive [S9] or longer
   retention, not compression.
3. Fidelity: 71% of qualifying edits copy text verbatim from an earlier tool output (E-24 P4).
   A "rewrite as facts" structurer at ingest cannot know which span will be needed. The published
   systems that work keep verbatim lines and condition on a goal (SWE-Pruner, Squeez).
4. Latency: an inline structurer sits on the critical path of every large tool result. ACON
   reports 73 s to 102 s wall time. In E-24 the 3 compactions cost 431 s in total, while 1,496
   tool results would have gone through the compressor (back-of-envelope: at 2 s each that is
   about 50 minutes; the 2 s is an assumption, not a measurement of a hosted model).
5. Scope: harness-level work, far from the thesis focus on GPU and serving infrastructure.

## 9. Validation completed so far

**E-24, trace study on the user's own Claude Code transcripts** (predeclared in journal 6.1).
Setup: 224 local transcript files, 216 main-thread sessions, 10,431 de-duplicated main-thread
requests (forked sessions copy requests across files; 825 duplicates removed), list prices from
[S13]. Results (FACT for this workload):
- Cost at list price: $1,925. Cache reads 74.0%, cache writes 17.3%, output 8.7%, uncached input
  about 0%.
- 97.2% of requests reused at least 90% of the previous prompt from cache.
- P1 (cache premise): of cache-write tokens in requests that broke the cache, TTL expiry after an
  idle gap caused 83.4%, session starts 8.9%, unexplained breaks 7.2%, compaction 0.39%. Below the
  10% threshold, so the premise is FALSE here. TTL-expiry rewrites cost 9.4% of all spend.
  Compaction rewrites cost 0.03%, the compaction calls themselves an estimated 0.07%.
- Compaction is rare and late: 3 events, at 854K to 969K tokens (1M-context sessions), shrinking to
  9K to 20K tokens, taking 76 to 244 s each. Median session peaked at about 26K tokens; 34 of 216
  sessions passed 150K.
- Tool results are 33.9% of appended tokens (Bash 4.07M, Read 1.06M of 5.62M) and drive 31.8% of
  total cost (character-share attribution, a heuristic).
- P2 (oracle ceiling): a perfect free compressor of every tool result saves 15.9% (r = 2), 23.8%
  (r = 4), 28.6% (r = 10). Inside the 15 to 40% band, so PIVOT: real, but within the range already
  published by AgentDiet and TokenPilot.
- P3 (compressor overhead): compressing only results of at least 1,024 tokens with Haiku 4.5 at
  r = 4 saves 15.0% on the main model and costs 0.43% (abstractive) or 0.25% (extractive), net
  14.6% or 14.8%. Not triggered: the small model's own cost is negligible. The cost intuition in
  the idea is right; fidelity and latency are the problems.
- P4 (verbatim dependency): 684 edits with an `old_string` of at least 20 characters. 71.5% appear
  verbatim in an earlier tool result (Bash 280, Read 209). TRUE at the 50% threshold. Post hoc,
  counting the agent's own earlier Write and Edit text as a competing source, 70.9% still have a
  tool result as the most recent source. Of those, 47% come from the last 3 tool results, 52% from
  the last 5, and the top 30% from 24 or more tool results back.
Limits: one user, one harness, Anthropic models, list prices instead of the user's plan, character
share for token attribution, sidechain data too thin to use (99 requests).
What it changed: dropped the cache-lifetime motivation, kept a measurement question about verbatim
reuse, moved the compressor design toward extractive and recoverable.

**E-25, local structurer screen** (predeclared). Setup: Qwen3-0.6B on the M4 GPU (MPS), thinking
off, greedy. 30 `Read` results of at most 8,000 characters (median 592 tokens) that E-24 found to
be the verbatim source of a later edit. Each condition targets about 25% of the input, without
knowing the later edit. Results (`results/e25_summary.json`):

| Condition | Exact survival of the later `old_string` | Mean line recall | Median size kept | Median time |
|---|---|---|---|---|
| A, abstractive "terse structured facts" (the proposed design) | 1 of 30 (3%) | 10% | 16% of chars | 3.3 s (p90 9.7 s) |
| B, model picks lines to keep, verbatim | 4 of 30 (13%) | 29% | 26% | 2.7 s (p90 12.9 s) |
| C, first and last lines (rule) | 6 of 30 (20%) | 33% | 24% | none |
| D, random lines | 0 of 30 | 28% | 25% | none |

Predeclared readings: A is below 30%, so query-agnostic abstractive structuring at about 4x
destroyed the spans the agent later edited, on this model. B did not beat C, so the small model
added nothing over a positional rule at this budget. Decode speed was about 22 tokens per second.
Limits: SCREEN ONLY. A 0.6B model is far weaker than Haiku-class models. The 8,000-character cap
biased the sample toward short reads, most of which a 1,024-token ingest threshold would not even
compress. The structural point does hold without a model: no condition that does not know the
future edit kept the edited span more than 20% of the time at a 25% budget.

What it changed: together with E-24 P4 (71% of edits copy tool-output text verbatim), it means a
structurer that paraphrases or blindly trims would force a re-read before most edits. Working
designs either condition on a goal the agent states (SWE-Pruner, Squeez) or keep the raw output
recoverable (Manus, pi-context-prune).

**E-26, I-43 check on public traces** (predeclared, 2026-10-04). TraceLab could not be used: its
public release keeps only character counts, no tool inputs or output text. The user approved the
SWE-bench part of the Exgentic dataset instead (shards 11 to 20, 528 MB; 391 traces, four harnesses,
five models). Agents there edit mostly through bash and Python, so the unit is any file-editing
action that copies a line of at least 20 characters verbatim from an earlier tool output.

| Population | Tool-dependent edits | Share whose source is outside the last 3 tool outputs (M) |
|---|---|---|
| Pooled, 391 SWE-bench traces | 789 | **43.7%** (predeclared verdict: SURVIVES, threshold 30%) |
| claude_code harness | 393 | 39.7% |
| Claude Opus 4.5 only | 172 | 29.1% (inside the 20 to 30% PIVOT band) |
| User's own Claude Code sessions, same metric | 1,569 | 60.4% |

Share of tool-dependent edits whose source a keep-last-K policy would have cleared:

| K | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| Pooled SWE-bench | 64% | 54% | 44% | 35% | 21% | 10% |
| Claude Opus 4.5 | 55% | 45% | 29% | 19% | 9% | 2% |
| User's own sessions | 75% | 65% | 60% | 55% | 47% | 38% |

Reading: on short benchmark runs, keeping the last 10 to 20 tool outputs covers most references. In
the user's long real sessions, even K = 20 leaves 38% of edit sources behind. The result depends on
the model, and multi-user real sessions with tool text are not public, so the long-session tail
rests on one user.

## 10. Next decisive experiment (only if the user wants this line)

- Hypothesis: on public multi-user coding-agent traces, at least 30% of edits whose source is a
  tool result take it from outside the last 3 tool results (heuristic threshold).
- Data: TraceLab release (4,265 sessions, Claude Code and Codex) [S10], if it contains tool inputs
  and outputs. Downloading it needs the user's permission.
- Steps: port `e24_trace_study.py` verbatim analysis to TraceLab's schema; add Codex `apply_patch`
  context lines as the edit-source signal.
- GO / KILL: under 20% outside the last 3: KILL (keep-last-3 clearing is safe, nothing to write).
  At least 30%: write a short workshop note with a retention-policy simulation (keep-last-K vs
  reference-aware retention), counting re-reads as cost.

## 11. Final evaluation plan

Not planned. The line is not a thesis candidate (Section 15).

## 12. Resource requirements

Local Mac only for E-24 style trace work. No GPU.

## 13. Scope and timeline

At most one week for the Section 10 measurement, with the KILL check after the first day of porting.

## 14. Publication and open-source story

At best a workshop note plus the trace-analysis script. Not a systems thesis.

## 15. Current verdict

**KILL as a thesis mechanism (I-41).** The small-model structurer is published (ACON, AgentDiet,
SWE-Pruner, Squeez) and the cache-continuity design is published (TokenPilot). The cache premise is
false in this workload and in TraceLab's 4,265 sessions: idle-gap TTL expiry, not compaction, breaks
the cache. The oracle saving (24% at r = 4) is real but inside the published range. A rewrite into
"facts" conflicts with the fact that 71% of edits copy text verbatim from tool outputs, and in
E-25 a small abstractive structurer kept the later-edited span in 1 of 30 cases.
As engineering advice the idea is fine, in the extractive and recoverable form: keep raw outputs
recoverable, prune lines rather than paraphrase, compress only large outputs, batch any clearing.

Narrowed variants: I-42 cache-aligned context editing (KILL, publicly proposed in #94177), I-43
reference-aware retention (passed E-26 at 43.7%, now a short-note candidate only, not a thesis;
next step a retention-policy simulation), I-44 edit-tolerant KV reuse
in serving engines (SCANNED, crowded), I-45 TTL-resilient KV retention for agent idle gaps (SCANNED,
crowded, folded into I-28 context).

## 16. Sources

Check levels follow journal Section 0.4. "checked" pages were read through a fetch tool that itself
summarizes with a small model, so exact numbers were cross-checked against the search snippet where
possible.

| ID | Source | Date | Check |
|---|---|---|---|
| S1 | Kang et al., "ACON: Optimizing Context Compression for Long-horizon LLM Agents", arXiv 2510.00615 | 2025-10 | checked (HTML v3) |
| S2 | Xu et al., "TokenPilot: Cache-Efficient Context Management for LLM Agents", arXiv 2606.17016, EMNLP 2026 Findings | 2026-06-15, rev. 2026-08-28 | checked (HTML) |
| S3 | "Improving the Efficiency of LLM Agent Systems through Trajectory Reduction" (AgentDiet), arXiv 2509.23586 | 2025-09 | search-only |
| S4 | Kovács, "Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents", arXiv 2604.04979 | 2026-04-04 | checked |
| S5 | Wang et al., "SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents", arXiv 2601.16746 | 2026-01-23, v4 2026-05-07 | checked |
| S6 | "SWE-Pruner Pro: The Coder LLM Already Knows What to Prune", arXiv 2607.18213 | 2026-07 | search-only |
| S7 | Song, "Cache-Aware Prompt Compression: A Two-Tier Cost Model for LLM API Caching", arXiv 2607.15516 | 2026-07-17 | checked |
| S8 | Lumer et al., "Don't Break the Cache", arXiv 2601.06007 | 2026-01-09 | checked |
| S9 | Khailo, "Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads", arXiv 2607.19214 | 2026-07-21 | checked |
| S10 | Zhu et al., "TraceLab: Characterizing Coding Agent Workloads for LLM Serving", arXiv 2606.30560, github.com/uw-syfi/TraceLab | 2026-06-29 | checked (abstract and HTML) |
| S11 | anthropics/claude-code issue #94177, "Prompt-cache forensics from 30 real sessions" | 2026-09-14 | checked |
| S12 | Claude Platform docs, "Context editing" | read 2026-10-04 | checked |
| S13 | Claude Platform docs, "Prompt caching" (pricing, multipliers, invalidation table) | read 2026-10-04 | checked |
| S14 | Claude Platform docs, server-side compaction `compact_20260112` | read 2026-10-04 | checked (mentioned on S12), details search-only |
| S15 | OpenAI docs, Responses API compaction, and Codex auto-compact | 2026 | search-only |
| S16 | OpenAI, extended prompt cache retention, 24 h default for some orgs since 2026-05-29 | 2026 | search-only |
| S17 | Manus, "Context Engineering for AI Agents: Lessons from Building Manus" | 2025-07 | search-only |
| S18 | Lindenbauer et al., "The Complexity Trap", arXiv 2508.21433 | NeurIPS 2025 DL4Code workshop | search-only |
| S19 | Liu, "What Does Context Compression Cost an Agent?", arXiv 2608.16370 | 2026-08-17 | checked |
| S20 | "Toward Reliable Context Compression for Long-Horizon Agents" (TRACE), arXiv 2608.06503 | 2026-08 | search-only |
| S21 | Ren et al., TACO, "A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression", arXiv 2604.19572 | 2026-04-21 | checked |
| S22 | Semenov, Dorofeev, "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents", arXiv 2606.11213 | 2026-05-01 | checked |
| S23 | Yang et al., "PatchKV: Efficient KV Cache Recovery for Dynamically Edited LLM Contexts", arXiv 2609.26219 | 2026-08-18 | checked |
| S24 | "KVEraser", arXiv 2606.17034 | 2026-06 | search-only |
| S25 | CacheBlend | EuroSys 2025 | prior-knowledge |
| S26 | O'Neill et al., "Still: Amortized KV Cache Compaction in a Single Forward Pass", arXiv 2606.07878 | 2026-06-05 | checked |
| S27 | Liu et al., "AgentKV: Phase-Aware KV Eviction for Agentic LLMs", arXiv 2609.14872 | 2026-09-14 | checked |
| S28 | Narayan et al., "Minions", arXiv 2502.15964 | 2025-02 | search-only |
| S29 | LLMLingua-2, arXiv 2403.12968 | 2024 | search-only |
| S30 | "Token-Oriented Object Notation vs JSON", arXiv 2603.03306 | 2026-03 | search-only |
| S31 | "State Design Matters", arXiv 2602.15858 | 2026-02 | search-only |
| S32 | Third-party analyses of Claude Code WebFetch using Haiku (mikhail.io, Medium) | 2025 to 2026 | search-only |
| S33 | Third-party analyses of Claude Code microcompact; claim that it uses a `cache_edits` capability | 2026 | search-only, UNVERIFIED |
| S34 | "Harness Engineering: ... A Source-Code Study of Eleven Systems", arXiv 2609.00006 | 2026-09 | search-only |
| S35 | Exgentic, "Multi-Benchmark LLM Agent Traces" (HF `DiscoPosse/agent-llm-traces`), 1,781 OpenTelemetry traces incl. a claude_code harness | 2026 | checked (README in local cache; data not downloaded) |
| S36 | pi-context-prune (prunes tool-call trees, keeps originals retrievable) | 2026 | search-only |
