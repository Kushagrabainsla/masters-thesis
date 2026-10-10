---
name: research-gap-dossier
description: Generate, adversarially validate, and document novel research ideas and thesis gaps, especially in ML systems, AI infrastructure, distributed systems, GPU/LLM serving, training, and agent infrastructure. Use when the user asks for new research ideas, unexplored gaps, thesis directions, novelty checks, adversarial review of candidate ideas, validation experiments/POCs, or a self-contained research dossier like a conference-oriented systems proposal. The workflow must use current literature and production-system evidence, run at least two adversarial review passes, define kill-tests before large implementation, and clearly separate what prior work already solves from the exact remaining question.
---

# Research Gap Dossier

## Operating principle

Treat research ideation as falsification, not brainstorming. Do not optimize for producing many clever ideas. Optimize for finding a small number of questions that survive current literature, adjacent systems, scale effects, simple baselines, and cheap experiments.

Use this sequence:

1. Establish constraints and evidence scope.
2. Map the current research and production frontier.
3. Generate candidate gaps from first principles and systems building blocks.
4. Run adversarial review pass 1.
5. Narrow or reformulate candidates.
6. Run adversarial review pass 2 with fresh targeted searches.
7. Design the cheapest decisive validation for each survivor.
8. Produce a concise shortlist and a complete dossier for the strongest survivor.

Never call an idea novel merely because an exact title was not found. State novelty conservatively: "I did not find public work that combines or directly answers X" and name the closest threats.

## 1. Establish constraints

Reuse context already supplied by the user. Do not ask them to repeat it.

If material constraints are missing, ask only what is needed:
- research domain or systems area;
- time budget and thesis/project duration;
- available hardware/data/access;
- desired outcome: publication, open source, career leverage, startup relevance, or a mix;
- preferred artifact: chat summary, Markdown, or document.

For a one-year systems thesis, default to requiring:
- a concrete system or mechanism;
- reproducible evaluation;
- a strong baseline;
- a measurable research claim;
- a realistic path to open source;
- a scope that can be validated early and completed within the year.

Read `references/decision-rubric.md` before ranking candidates.

## 2. Build the frontier map

Novelty is time-sensitive. Search the current web before making substantive novelty claims.

Prioritize evidence in this order:
1. peer-reviewed conference papers and proceedings;
2. recent arXiv papers from credible research groups;
3. official engineering blogs, documentation, RFCs, issue trackers, and repositories for active systems;
4. production traces or workload-characterization studies;
5. secondary summaries only for discovery, never as the main evidence for a technical claim.

Search both the exact idea and its nearest neighbors. Cover at least these evidence classes:
- exact mechanism or primitive;
- adjacent mechanism that could make the proposed idea trivial;
- architecture/workload-specific prior work;
- production systems from relevant companies/projects;
- negative results or workload studies that could make the opportunity disappear at scale.

For fast-moving infrastructure topics, inspect recent repository issues/RFCs/PRs in addition to papers. A paper search alone is insufficient.

Keep a source ledger with: title, year/date, venue/source type, what it proves, what it does not prove, and why it threatens or supports the candidate gap.

## 3. Generate candidate gaps

Generate 5-10 candidates, then aggressively reduce them. Prefer questions that arise from a mismatch between a modern workload and an older systems assumption.

Useful building-block lenses include:
- checkpoint/restart;
- live migration;
- preemption and recovery;
- scheduling and admission control;
- caching and memory hierarchy;
- resource disaggregation;
- leases and reservations;
- speculative execution;
- work stealing;
- load balancing and elasticity;
- isolation and multi-tenancy;
- consistency and ownership transfer;
- backpressure and queueing;
- fault tolerance;
- topology changes;
- hardware-aware placement;
- power/capacity constraints.

Also search for architecture-induced systems problems: ask whether a new model architecture, agent workflow, memory representation, or accelerator changes the state or resource semantics enough to require a new primitive.

For every candidate, write only:
- Problem: what concrete system behavior is inefficient, incorrect, or unknown?
- Why now: what changed recently?
- Existing work: what is already solved?
- Exact gap: the smallest remaining unanswered part.
- Research question: falsifiable and comparative.
- Hypothesis: what should happen if the idea is right?
- Cheapest decisive test.
- Biggest novelty threat.
- Scale risk.

Avoid solution-first ideas. Prefer "What fact is still unknown?" before "What system should we build?"

## 4. Adversarial review pass 1

Attack every candidate using all of these questions:

1. Novelty attack: Is the exact mechanism already published or shipping?
2. Adjacent-work attack: Does another system already provide the needed primitive under a different name?
3. Combination attack: Is this only "Paper A + Paper B" without a new coupling or systems constraint?
4. Simplicity attack: Would stop-and-copy, replay, batching, caching, reactive scheduling, or another simple baseline solve most of the problem?
5. Scale attack: Does the phenomenon disappear under concurrency, batching, multiplexing, or larger clusters?
6. Oracle attack: Even with perfect information, is the maximum possible improvement meaningful?
7. Correctness attack: Is there a real consistency/recovery/state-management issue, or only data movement?
8. Feasibility attack: Can the claim be evaluated with available resources within the user's timeline?
9. Relevance attack: Does the problem matter in a realistic production regime?
10. Scoop-risk attack: Is an active upstream project obviously converging on the same contribution?

Drop weak candidates. Reformulate candidates when only a narrower unanswered core survives.

Read `references/adversarial-checklist.md` for the decision rules.

## 5. Adversarial review pass 2

Run a fresh targeted search for each survivor. Do not reuse only the first-pass evidence.

Specifically search for:
- the revised exact research question;
- the proposed mechanism plus synonyms;
- the strongest simple baseline;
- the most recent 6-12 months of papers;
- current repos/RFCs/issues for relevant serving/runtime projects;
- workload-characterization evidence at realistic scale;
- cross-domain systems that solve the same state or scheduling problem elsewhere.

Then ask a hostile reviewer question:

> "What is the shortest argument a strong systems reviewer could use to reject this as already solved, trivial, unimportant, or unevaluable?"

Revise again. A candidate survives only if the answer to that reviewer is concrete and evidence-backed.

## 6. Design validation before implementation

Separate questions already answered by literature from questions that still require experiments. Do not reproduce established results unless needed as a baseline or integration check.

Choose the smallest experiment that tests the remaining uncertainty:

- Problem existence unknown -> workload measurement or trace study.
- Maximum upside unknown -> perfect-future/oracle upper bound.
- Mechanism feasibility unknown -> microbenchmark or minimal POC.
- Scale behavior unknown -> concurrency sweep or trace replay.
- Correctness unknown -> deterministic equivalence/restore/cutover test.
- Simple baseline may be enough -> benchmark the simple baseline first.

Use synthetic workloads only for screening. Before a thesis commitment, move to real models, public/production traces, or a real serving engine when possible.

Define GO / PIVOT / KILL conditions before running the experiment. If numeric thresholds are heuristics rather than literature standards, label them as such.

When local hardware is available, explicitly separate:
- what can be screened locally;
- what requires NVIDIA GPUs or a cluster;
- what requires real multi-node/network measurements for the final claim.

## 7. Select survivors

Use the rubric in `references/decision-rubric.md`.

Return a small decision table with:
- candidate;
- exact surviving question;
- what prior work already answers;
- strongest novelty threat;
- evidence still missing;
- next validation step;
- status: GO, PIVOT, MEASURE-FIRST, or KILL;
- confidence in the public gap: low / medium / high.

Do not rank a candidate highly only because it is difficult. Prefer a crisp, measurable gap over an ambitious but vague system.

## 8. Produce the final dossier

For the strongest surviving idea, follow `references/dossier-template.md`.

Default style:
- simple and terse;
- titles, subtitles, paragraphs, bullets, and simple tables only;
- no decorative visuals;
- explain technical ideas in plain language first, then precise language;
- distinguish facts, interpretation, hypotheses, and proposed work;
- include negative evidence and reasons the idea may fail;
- include experiments already run and what they changed;
- include the next POC and exact evaluation plan;
- include primary-source references.

If the user asks for a DOCX, use the document-creation skill and preserve this simple style. If no artifact is requested, output the dossier in Markdown.

## 9. Required research-question form

Prefer questions shaped like:

> Can [new information/mechanism/primitive] improve or enable [measurable systems outcome] under [specific workload/architecture/constraint], compared with [strong baseline], while preserving [SLO/correctness/cost constraint]?

A valid question must expose what result would make the answer "no."

## 10. Non-negotiable research behavior

- Never invent a gap.
- Never treat lack of search hits as proof of novelty.
- Never hide prior work that weakens the idea.
- Never compare only against a weak baseline when a stronger public system exists.
- Never build a complex scheduler before checking the oracle upper bound when the opportunity itself is uncertain.
- Never generalize a local single-user result to enterprise scale without a scale argument or experiment.
- Never generalize a synthetic result to production without clearly labeling the limitation.
- Prefer killing two attractive ideas over preserving three weak ones.
- Preserve failed experiments in the dossier when they materially shaped the surviving question.
