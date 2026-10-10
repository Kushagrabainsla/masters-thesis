# Agentic Harnesses: An Evidence-Graded Literature Review

Date: 2026-09-30. Author: Claude (Opus 5.5) at the request of Kushagra Bainsla.
Journal links: Section 4 entry of 2026-09-30 and Section 7.8 of `RESEARCH_JOURNAL.md`.
Status: literature review. Not a thesis decision. No experiments were run for this report.

---

## 0. Scope, method and how to read this

**What "harness" means here.** An agentic harness (also called scaffold, agent framework, or
execution harness) is everything around the model weights that turns a language model into an
agent: the control loop, the tool and action interface, context and memory management, state
that persists across steps or sessions, verification and stopping rules, orchestration of
sub-agents, and the permission and sandbox layer. A 2026 survey formalizes the same split as
"foundation model plus execution harness" with six runtime responsibilities: observation,
context, control, action, state and verification [S1].

**Method.** Web and arXiv searches on 2026-09-30, opening primary abstract pages where possible.
Every source in Section 9 carries a check level:

| Level | Meaning |
|---|---|
| checked | Primary abstract or page opened on 2026-09-30 and the cited numbers confirmed there |
| search-only | Numbers come from search-result text or secondary pages. Primary not opened (or blocked) |
| prior-knowledge | Well-known work cited from my training knowledge, not re-opened this session |

**Evidence labels used for each claim.**

| Label | Meaning in this report |
|---|---|
| ESTABLISHED | Replicated by at least two independent groups with different methods, at least one peer-reviewed, no strong contrary evidence |
| WORKING HYPOTHESIS | Supported by at least one controlled study, but replication is thin, magnitudes conflict, or results are model- or task-dependent |
| SPECULATIVE | Argued in position papers, vendor posts or single uncontrolled reports. Little controlled evidence |
| OPEN | The claim is that a question is unresolved. No effect is asserted, so the replication test does not apply |

**Four limits to state up front.**

1. **There is no scientific consensus in the strict sense.** The field is about three years old.
   There are no meta-analyses that pool effect sizes across harness studies, and no standards
   body has issued a consensus statement on harness design. The nearest thing to an expert
   consensus document is the International AI Safety Report 2026 [S40], which covers agents
   only at a high level. Where this report says ESTABLISHED, it means "replicated", not
   "settled by a field-wide consensus process".
2. **Most evidence is preprints.** A large share of the 2026 harness literature is arXiv
   preprints, several single-author, none yet replicated. Peer-reviewed anchors are SWE-agent
   (NeurIPS 2024), CodeAct (ICML 2024), HAL (ICLR 2026), ABC and MAST (NeurIPS 2025),
   Huang et al. (ICLR 2024), Kamoi et al. (TACL 2024), AgentDojo (NeurIPS 2024).
3. **Many sources are not independent.** HAL and "AI Agents That Matter" share authors
   (Kapoor, Stroebl, Narayanan). HAL also shares authors with ABC [S36] (Kapoor, Jin, Zhu, Kang,
   Liang) and with the reliability paper [S42] (Kapoor, Kirgis, Narayanan). Nasr et al. [S44] and
   CaMeL [S45] share five authors (Carlini, Shumailov, Hayes, Terzis, Tramèr), and AgentDojo [S43]
   shares Debenedetti and Tramèr with CaMeL. SWE-agent and mini-swe-agent come from one group. Vendor
   posts (Anthropic, OpenAI, Cognition) evaluate their own systems on private evals. I flag
   this per claim.
4. **Replication across time is structurally hard.** API models are deprecated within months,
   so a 2024 harness result often cannot be re-run on the same model in 2026. Benchmarks are
   contaminated or retired (Section 4, C8). This is a property of the field, not of any one paper.

---

## 1. Executive summary

| # | Claim | Label | Strongest support | Main uncertainty |
|---|---|---|---|---|
| C1 | For a fixed model, the harness changes measured task success, sometimes by more than a model swap | ESTABLISHED (direction), magnitude contested | SWE-agent ACI [S5], HAL [S8], Starace pre-registered GAIA study [S12] | Accuracy gaps range from 0 to about 28 points depending on task difficulty, model family and context budget. Several careful studies find small or non-significant accuracy gaps [S11, S13] |
| C2 | Harness choice changes cost and token use far more reliably than it changes accuracy | WORKING HYPOTHESIS, strong | Vats and Golev (up to 40x tokens per solved task, accuracy gaps with CIs including zero) [S11], Complexity Trap [S20], Arjmandi (1.2 to 1.6x cost) [S13], AI Agents That Matter [S7] | Few studies report both with CIs |
| C3 | Simple harnesses are often competitive with complex ones | WORKING HYPOTHESIS | Agentless [S6], mini-swe-agent [S16], Complexity Trap [S20], Fan et al. bash-only [S17], AI Agents That Matter [S7] | Meta-Harness shows searched harnesses beat hand-built ones [S18], so "simple wins" may just mean "hand-built complexity is poorly targeted" |
| C4 | Multi-agent designs help on decomposable tasks and hurt on sequential ones. No universal advantage | ESTABLISHED (no universal advantage), mechanism a working hypothesis | Kim et al. 260 configurations, budget-matched [S24], MAST [S25], MAD critiques [S27, S28], Starace [S12] | The only budget-matched frontier comparison found is one preprint [S24]. Most other reported gains are not compute-matched. Anthropic reports token usage explains 80% of variance on BrowseComp [S26] |
| C5 | Intrinsic self-correction without external feedback is unreliable. Loops work when the harness supplies a real verifier (tests, execution, checkers) | ESTABLISHED for pre-2025 models, open for RL-trained reasoning models | Huang et al. [S29], Kamoi et al. survey [S30], Large Language Monkeys [S31] | RL post-training may have changed intrinsic self-correction. Kamoi et al. already note fine-tuning can enable it |
| C6 | Performance degrades as context grows, so context management is a first-order harness function | ESTABLISHED (degradation), OPEN (best strategy) | Lost in the Middle [S19], Chroma 18-model report [S21], Fan et al. [S17], tool-count degradation [S23] | Which management strategy is best is contested: masking vs summarization vs evolving playbooks [S20, S22, S17] |
| C7 | Models post-trained inside one harness learn that harness's conventions and transfer poorly | WORKING HYPOTHESIS, mixed evidence | CHART (direct transfer evidence) [S35], Le et al. (eval-harness sensitivity, and pooled-harness RL gave no transfer gain) [S34] | Arjmandi finds no average native-harness advantage for frontier vendor pairs [S13]. One widely repeated number could not be traced to a source (Section 8) |
| C8 | Published agent benchmark scores are confounded by harness non-disclosure, contamination and flawed graders | ESTABLISHED | ABC checklist [S36], SWE-bench+ [S37], SWE-bench Illusion [S38], OpenAI retiring SWE-bench Verified [S39], HAL log analysis [S8] | Size of the bias per benchmark varies |
| C9 | Reliability (consistency across runs, robustness) lags average accuracy | ESTABLISHED for the lag. WORKING HYPOTHESIS that it improves more slowly | tau-bench pass^k [S41], "Towards a Science of AI Agent Reliability" (15 models) [S42], International AI Safety Report 2026 [S40] | Metrics are not standardized. The "improves more slowly" trend rests on one group's paper [S42] |
| C10 | Prompt injection is not solved by model-level or in-band defenses. Harness-level isolation is the most credible direction, at a utility cost | WORKING HYPOTHESIS, strong (in-band defenses fail under adaptive attack), WORKING HYPOTHESIS (architectural defenses suffice) | Nasr et al. [S44], AgentDojo [S43], CaMeL [S45], MCP measurement studies [S46, S47] | The adaptive-attack result rests on one paper [S44] whose authors overlap with S43 and S45. Architectural defenses are young and have narrow evaluations |
| C11 | The real-world productivity effect of agentic harnesses is not reliably measured yet | OPEN | METR RCT 2025 (n=16) [S48], METR 2026 redesign notice [S49], earlier Copilot RCTs [S50, S51] | Selection effects now break the RCT design. Field evidence predates modern agents |
| C12 | Agent harnesses create a distinct serving workload, and program-aware serving can raise throughput | WORKING HYPOTHESIS | Autellix 4 to 15x vs vLLM [S52], Anthropic token multipliers [S26] | Gains reported by the systems' own authors against vanilla baselines. Little independent replication |

The one-sentence version: **the harness is a first-class variable that the field has only
started to control for since late 2025. Its effect on cost is large in the studies that measured it
(1.2x to 40x, from few sources), its effect on accuracy is real but highly conditional, and most of the prescriptive advice about which
harness design is "best" is still working hypothesis or speculation.**

---

## 2. Definition and taxonomy

The components below are the ones that experimental papers actually vary. The right column
says where the evidence for each sits in Section 4.

| Component | What it decides | Example variants | Evidence |
|---|---|---|---|
| Control loop | When to think, act, observe, stop | ReAct loop, fixed pipeline (Agentless), planner-executor, initializer plus worker sessions | C1, C3 |
| Action interface (ACI) | What actions exist and how they are expressed | JSON function calls, Python code actions (CodeAct), bash only, custom file editor with linter | C1, C3 |
| Context management | What stays in the window | Raw history, observation masking, LLM summarization, retrieval of tools, evolving playbooks | C6 |
| State and memory | What persists outside the window | Progress files, git history, feature lists, vector memory, MemGPT-style paging | C6 |
| Verification | How the harness knows it is done or wrong | Unit tests, linters, execution feedback, LLM judge, self-critique | C5 |
| Orchestration | How many model instances and how they talk | Single agent, orchestrator plus sub-agents, debate, decentralized peers | C4 |
| Permissions and isolation | What an action may touch | Sandboxes, allow-lists, dual-LLM and capability tracking (CaMeL) | C10 |

A practitioner framing that shaped the vocabulary: Anthropic's "Building effective agents"
(December 2024) separates *workflows* (LLM calls orchestrated along predefined code paths) from
*agents* (the model directs its own process and tool use), and recommends the simplest design
that works [S2, prior-knowledge]. This is design guidance, not evidence.

---

## 3. How understanding evolved

**Phase 1, 2022 to mid-2023: prompting patterns as "agents".** ReAct (ICLR 2023) interleaved
reasoning traces and actions [S3]. Toolformer (NeurIPS 2023) taught tool calls through
self-supervision. Reflexion (NeurIPS 2023) added verbal self-reflection across trials
[S4]. Generative Agents and Voyager showed memory and skill libraries. AutoGPT-style open-ended
loops drew attention but little rigorous evaluation. The implicit assumption: the model is the
agent, and the loop is a thin wrapper. (prior-knowledge for this phase)

**Phase 2, late 2023 to 2024: the interface matters, and complexity is questioned.**
SWE-bench (ICLR 2024) gave a realistic, execution-graded target. SWE-agent (NeurIPS 2024)
introduced the term agent-computer interface and showed that interface design alone changes
results for the same model [S5]. CodeAct (ICML 2024) showed code as the action format beats
JSON or text actions by up to 20% across 17 models [S15]. Almost at once came the pushback:
Agentless reached 32.0% on SWE-bench Lite at $0.70 per issue with a fixed three-phase pipeline
and no autonomous tool choice [S6], and "AI Agents That Matter" showed simple baselines sit on
the cost-accuracy Pareto frontier while complex state-of-the-art agents do not [S7]. Huang et
al. (ICLR 2024) and Kamoi et al. (TACL 2024) deflated intrinsic self-correction [S29, S30].
Multi-agent debate was first reported as helpful (Du et al., prior-knowledge) and then
challenged [S27]. Evaluation validity problems surfaced: SWE-bench+ found solution leakage in
32.67% of successful resolutions [S37].

**Phase 3, 2025: native tool calling, reasoning models, context engineering.** Function
calling became native, and MCP (announced November 2024) standardized tool servers
(prior-knowledge). Reasoning models trained with RL on agentic tasks shifted work from the
harness into the weights. Practitioner writing moved from "prompt engineering" to "context
engineering" and to long-running harnesses with external state (progress files, git) [S9].
Anthropic reported a multi-agent research system beating a single agent by 90.2% on an
internal research eval, and separately noted that token usage explained 80% of the variance in
BrowseComp performance [S26]. Cognition argued
the opposite ("Don't build multi-agents", prior-knowledge). Peer-reviewed failure analysis of
multi-agent systems arrived (MAST) [S25], as did evaluation-validity work (ABC) [S36] and
reliability metrics. Security work showed in-band prompt-injection defenses fail under adaptive
attack [S44], and CaMeL proposed design-level isolation [S45]. The METR RCT found experienced
developers 19% slower with early-2025 tools [S48].

**Phase 4, late 2025 to September 2026: the harness becomes an explicit experimental
variable.** HAL (ICLR 2026) standardized cross-harness evaluation at scale [S8]. OpenAI stopped
treating SWE-bench Verified as a frontier measure in February 2026 citing contamination and
flawed tests [S39]. A wave of controlled 2026 studies now varies only the harness and reports
the effect [S11 to S14, S17], position papers ask for mandatory harness disclosure [S10], and
researchers optimize the harness itself automatically (Meta-Harness) [S18]. The newest thread
asks whether RL-trained models become coupled to their training harness [S34, S35].

**Net trajectory.** The field moved from "the model is the agent" (2022) to "the interface
matters and simple is competitive" (2024) to "the harness is a confound that must be
controlled, disclosed and possibly optimized" (2026). What has *not* changed since 2024: real
verifiers beat self-critique, and benchmark validity is fragile.

---

## 4. Findings by claim

Each claim lists evidence, quality notes, and what would change the conclusion.

### C1. For a fixed model, the harness changes measured success

**Label:** ESTABLISHED for direction. Magnitude is contested.

**Evidence.**
- SWE-agent (NeurIPS 2024): a custom ACI raised GPT-4 Turbo to 12.5% pass@1 on full SWE-bench,
  with ablations over interface design choices [S5]. Peer-reviewed, one model family, 2024 models.
- HAL (ICLR 2026): 21,730 rollouts, 9 models, 9 benchmarks, about $40,000, with task-specific and
  generalist scaffolds evaluated under one harness [S8]. Peer-reviewed. Shares authors with S7.
- Starace (June 2026, pre-registered): three scaffolds (ReAct, Planner-Actor-Rater multi-agent,
  planner-then-executor), five models, GAIA Levels 1 and 2, three attempts per question. Up to
  28 points within one model at the harder level. Pre-registered hypothesis of at least a 10-point
  gap confirmed. More capable models were *not* less scaffold-sensitive [S12]. Single author, preprint.
- Terminal-Bench leaderboard, Claude Opus 4.6: 58.0 under Claude Code vs 62.9 under Terminus-2 on
  version 2.0, and 70.1 vs 63.8 on version 2.1 (tbench.ai release notes for 2.1). The ranking
  reverses between benchmark versions, which is itself evidence of sensitivity. Leaderboard data,
  not a controlled study, so it carries no weight on its own [S14 covers 2.0 only].
- SWE-bench Pro: Claude Opus 4.5 at 45.9% under Scale's standardized scaffold vs about 49.8 to
  51.8% under Claude Code, Cursor and Auggie [S32, search-only, secondary source (Augment blog)].
- Zhang et al. (position, May 2026) report cases where harness-induced variance exceeds
  model-induced variance, including ranking reversals [S10]. Position paper.

**Contrary or moderating evidence.**
- Vats and Golev (2026): three open-source harnesses, two models, 50 tasks, 300 trials. Paired
  within-model pass-rate differences of 0 to 8 points. Per the abstract, the 95% paired-task bootstrap CIs
  include zero except for the largest gap, so the largest gap is the one significant accuracy
  difference [S11]. Small n, one trial per cell.
- Arjmandi (September 2026): contamination-controlled private suite, 256 tasks, 792 of 800 runs,
  four models. The paired native-vs-neutral contrasts use 80 tasks (61 repository, 19 contest). No
  average advantage for a vendor's native harness (-1.25 points for Opus, 95% CI -10.0 to +7.5;
  +1.25 for GPT-5.5, CI -4.4 to +6.9), but strong task heterogeneity: native trails by 9.0 points on
  repository tasks and leads by 23.7 points on contest tasks (p = 0.003) [S13]. Single author, preprint, private tasks.
- Fan et al. (September 2026): 176 configurations, four models, SWE-bench Verified and Terminal-Bench
  2.1. Planning helped weaker models on accuracy and mainly cut cost for stronger ones with little
  accuracy change. Context management mattered mainly when the
  context budget was tight [S17].

**Synthesis (INTERPRETATION).** The harness effect on accuracy is conditional. It is large on
harder tasks, under tight context budgets, and for particular model-harness pairs, and small or
non-significant on average for strong models on easier tasks. Studies that report one large
number usually pick the extreme cell. Studies with CIs find averages near zero but large,
significant task-level gaps (S13: 23.7 points, p = 0.003), and the pre-registered study found the
largest gap (S12: 28 points). So the pattern is "conditional on task type and model pair", not
"careful studies find small effects". Both can be true at once.

**Quality notes.** Sample sizes are small (50 to 256 tasks, 80 paired tasks in S13, 1 to 3 attempts per task in most
2026 studies). Few report variance across seeds. Leaderboard comparisons confound harness with
submission effort, tool budgets and turn limits.

**What would change this.** A multi-lab, pre-registered study that crosses at least five
harnesses with at least five models on a contamination-controlled suite, with seeds and CIs,
and a variance decomposition (harness vs model vs interaction).

### C2. The harness moves cost more reliably than accuracy

**Label:** WORKING HYPOTHESIS, strong. Magnitudes conflict (1.2x in S13 to 40x in S11).

**Evidence.**
- Vats and Golev: up to 40x difference in tokens per solved task across harnesses (28 to 37K for
  Goose vs 1.1 to 1.5M for OpenCode), non-overlapping bootstrap CIs, while accuracy gaps were not
  significant. OpenCode showed about 10x more no-action turns [S11].
- Complexity Trap (NeurIPS 2025 DL4Code workshop): observation masking halves cost vs the raw
  agent while matching LLM summarization's solve rate across five model settings [S20].
- Arjmandi: neutral harness cost 1.2 to 1.6x more per solved task, with a stated caveat that 58
  runs lacked usage records [S13].
- AI Agents That Matter (TMLR): simple baselines are Pareto-efficient against complex agents [S7].
- Fan et al.: bash-only interfaces lower cost for strong models [S17].

**Independence.** S7 and S8 share authors. S11, S13, S17, S20 are independent groups.

**Uncertainty.** Cost depends on provider pricing, caching and batching, which change over time
and differ between vendors. Token counts are the more stable measure. Few papers report latency.

### C3. Simple harnesses are often competitive with complex ones

**Label:** WORKING HYPOTHESIS.

**Evidence.** Agentless [S6], mini-swe-agent (about 100 lines of agent logic, reported 65% and
later over 74% on SWE-bench Verified, per the project's own reports) [S16], Complexity Trap
[S20], Fan et al. [S17], AI Agents That Matter [S7].

**Contrary evidence.** Meta-Harness (March 2026) searched harness code with an agentic proposer
and beat hand-engineered baselines: 7.7 points over a state-of-the-art context management
system with 4x fewer context tokens on classification, 4.7 points on 200 IMO-level problems
across five held-out models, and discovered harnesses that surpass the best hand-engineered baselines on TerminalBench-2 [S18]. Anthropic's
long-running-agent harness (initializer agent, feature list, progress file, git commits)
argues that some structure is needed for multi-session work [S9, vendor, no controlled comparison].

**Synthesis (INTERPRETATION).** The better-supported statement is not "simple beats complex"
but "complexity added by hand, without measurement, is often wasted". Targeted harness changes
found by search or by ablation do help. The "bitter lesson" reading, that stronger models make
harnesses matter less, is not supported by Starace's finding that capable models were not less
scaffold-sensitive [S12], though one GAIA preprint cannot settle it. That reading stays SPECULATIVE.

### C4. Multi-agent architectures: task-contingent, not universally better

**Label:** ESTABLISHED that there is no universal advantage. The mechanism is a working hypothesis.

**Evidence.**
- Kim et al. (December 2025): 260 configurations, 6 benchmarks, 5 architectures, 3 model families
  (OpenAI, Google, Anthropic), with standardized tools and prompts. Single-agent and multi-agent
  runs were matched for total reasoning-token budget (mean 4,800 tokens per trial) and tool
  access, so this is a compute-matched comparison. Effects range from +80.8% on decomposable financial reasoning
  to -70.0% on sequential planning. Independent multi-agent designs amplify errors up to 17.2x.
  Centralized coordination contains errors best. A predictive model reaches cross-validated
  R^2 = 0.373 (0.413 with task-grounded capability metrics) and picks the best architecture on 87%
  of held-out configurations [S24]. Preprint. The R^2 means most variance is unexplained.
- MAST (NeurIPS 2025 Datasets and Benchmarks): 1,600+ annotated traces from 7 frameworks, 14 failure
  modes in 3 categories. Specification and system-design issues about 41.8% and inter-agent
  misalignment about 36.9% of failures [S25, search-only for the percentages].
- Multi-agent debate: "Should we be going MAD?" found debate does not reliably beat self-consistency
  or ensembling, partly due to hyperparameter sensitivity [S27]. "Stop Overvaluing Multi-Agent Debate"
  found no MAD method won more than 20% of 36 configurations against chain-of-thought [S28].
- Starace: multi-agent beat ReAct only within the Anthropic model family [S12].
- Anthropic's research system: +90.2% over single-agent Opus 4 on an internal research eval.
  Separately, on BrowseComp (a public benchmark) token usage explained 80% of performance variance,
  and multi-agent runs used about 15x the tokens of chat [S26]. Vendor, internal eval for the
  90.2%, not compute-matched.

**Open debate.** Whether multi-agent gains survive when the single-agent baseline gets the same
token budget. Kim et al. [S24] is a budget-matched comparison and still finds gains up to +80.8% and losses
down to -70.0%, but it is one preprint with its own budget definition. Most other positive
reports do not compute-match. Anthropic's BrowseComp figure shows tokens matter on that
benchmark. It does not say how much of the separate 90.2% internal-eval gain is extra compute.

### C5. Self-correction needs an external verifier

**Label:** ESTABLISHED for models up to 2024. OPEN for RL-trained reasoning models.

**Evidence.**
- Huang et al. (ICLR 2024): earlier self-correction gains relied on oracle labels. Without them,
  intrinsic self-correction does not improve and can hurt reasoning [S29].
- Kamoi et al. (TACL 2024, critical survey): no prior work showed successful self-correction with
  feedback from prompted LLMs on general tasks. It works where reliable external feedback exists,
  and large-scale fine-tuning can enable it [S30].
- Large Language Monkeys: repeated sampling raised SWE-bench Lite coverage from 15.9% (1 sample) to
  56% (250 samples) because unit tests can select correct candidates. Without automatic verifiers,
  majority voting and reward models plateau after several hundred samples [S31].
- SWE-agent's interface included a linter that rejects syntactically broken edits, an early example
  of the harness supplying a cheap verifier [S5].

**Uncertainty.** Kamoi et al.'s third finding (fine-tuning enables self-correction) is the opening
through which RL-trained reasoning models may have changed the picture. I found no controlled
2026 replication of Huang et al. on current reasoning models. Treat "self-critique is useless"
as outdated and "external verifiers dominate self-critique" as the defensible claim.

### C6. Context degradation makes context management a first-order function

**Label:** ESTABLISHED that degradation happens. OPEN on the best remedy.

**Evidence for degradation.**
- Lost in the Middle (TACL 2024): accuracy depends on where relevant information sits in the
  context, dropping for middle positions [S19, prior-knowledge].
- Chroma "Context Rot" (2025 technical report): 18 models, including GPT-4.1, Claude 4 and
  Gemini 2.5, degrade as input length grows, even on simple tasks [S21]. Industry report, not
  peer-reviewed, from a vendor with a retrieval product.
- Tool catalogues: RAG-MCP reports tool-selection accuracy rising from 13.62% to 43.13% when
  tools are retrieved rather than all loaded, and cutting prompt tokens by more than half [S23].
- Fan et al.: context management matters more as the budget tightens, mainly by preventing
  overflow failures [S17].

**Contested remedies.**
- Complexity Trap: simple observation masking matches LLM summarization at about half the cost
  [S20].
- Fan et al.: rule-based elision followed by LLM summarization was most efficient. Recoverable
  elision added complexity without accuracy gains [S17].
- ACE: incremental "delta" updates to an evolving playbook avoid "context collapse" from full
  rewrites, +10.6% on agent tasks and +8.6% on financial reasoning [S22].
- Meta-Harness found a context management design 7.7 points better with 4x fewer tokens on one
  task family [S18].

**Synthesis (INTERPRETATION).** Everyone agrees that unbounded history is bad. The disagreement
is whether cheap mechanical rules (drop old observations) are enough. Current evidence favours
cheap rules first, with LLM summarization as a second stage, but the results come from coding
tasks. Evidence for other domains is thin.

### C7. Model-harness coupling through post-training

**Label:** WORKING HYPOTHESIS with mixed evidence.

**Evidence for coupling.**
- Le et al. (September 2026): Qwen3-8B, SWE-bench Verified. The evaluation harness moved mean
  solve rate from 2.14% to 9.27% (4.3x) while the training recipe moved results 1.16x, so base
  rates are very small. Pooling harnesses during RL gave no more portable capability on held-out
  harnesses (+0.25 points, 95% CI -0.48 to +1.02) [S34]. This is evaluation-harness sensitivity
  on a weak model plus a null for pooled training, so it is indirect evidence for coupling.
- CHART (September 2026, the most direct transfer evidence): search agents trained on static harness pools parallelize on at most 5%
  of held-out-harness turns, vs 89% with a rotating harness curriculum [S35]. This shows a
  specific behavior (parallel search) fails to transfer across instruction formats.

**Evidence against strong coupling at the frontier.** Arjmandi found no average advantage for a
vendor's native harness with its own model [S13].

**Synthesis (INTERPRETATION).** Coupling is clearly real for small open models trained with RL
in one harness. For frontier models trained across many harnesses, the average effect may wash
out while task-level effects remain. Both studies are fresh preprints, so this could move
quickly.

### C8. Agent benchmark scores are confounded

**Label:** ESTABLISHED.

**Evidence.**
- ABC (NeurIPS 2025): of ten popular agentic benchmarks, seven had outcome-validity flaws, seven
  had task-validity flaws, and all had reporting gaps. Errors can shift performance estimates by
  up to 100% in relative terms. Applying the checklist to CVE-Bench cut overestimation by 33% [S36].
- SWE-bench+: 32.67% of successful resolutions involved solution leakage in the issue text, and
  31.08% of passing patches were suspicious due to weak tests. Filtering dropped SWE-Agent+GPT-4
  from 12.47% to 3.97% [S37].
- SWE-bench Illusion (ICSE 2026 SEIP track): models identify buggy file paths from the issue text
  alone at up to 76% on SWE-bench repositories but up to 53% on other repositories, consistent
  with memorization [S38].
- OpenAI (February 2026) stopped treating SWE-bench Verified as a frontier measure, citing models
  reproducing gold patches from task IDs and tests that reject correct solutions or demand
  unspecified behavior [S39, search-only, primary page blocked].
- HAL's log analysis found agents searching HuggingFace for benchmark answers instead of solving
  tasks, and misusing payment methods in booking tasks [S8].
- Two 2026 preprints argue results must disclose the harness [S10] and that fixed scaffolds make
  execution-critical decisions the model gets credited for [S33].
- The International AI Safety Report 2026 notes some models can tell test settings from deployment
  ("evaluation awareness"), which further weakens benchmark validity [S40, search-only].

**Independence.** S36 (ABC) and S8 (HAL) share authors (Kapoor, Jin, Zhu, Kang, Liang). S37,
S38 and S39 are separate groups with different methods (issue-text leakage, memorization probes,
an internal audit). With that overlap noted, this is still the most robust finding in the review.

### C9. Reliability lags capability

**Label:** ESTABLISHED for the gap between accuracy and consistency. WORKING HYPOTHESIS for
"improves more slowly", which rests on S42 alone (its authors overlap with HAL, S8).

**Evidence.**
- tau-bench (2024): gpt-4o reached about 61% pass^1 on the retail domain but about 25% pass^8, the
  probability of succeeding on all 8 trials [S41, search-only].
- "Towards a Science of AI Agent Reliability" (ICML 2026): 12 metrics across consistency,
  robustness, predictability and safety, 15 models, two benchmarks. Recent capability gains
  produced only small reliability gains [S42].
- International AI Safety Report 2026: reliability drops sharply as tasks need more sequential
  steps [S40, search-only].
- HAL: higher reasoning effort reduced accuracy in the majority of runs, a sign that settings
  expected to help can hurt [S8].

**Harness relevance (INTERPRETATION).** If a model is inconsistent across runs, the harness is
where retry, verification and rollback live. This is an argument for measuring harnesses on
pass^k, not pass@1.

### C10. Prompt injection and harness-level security

**Label:** WORKING HYPOTHESIS (strong) that in-band defenses fail against adaptive attackers. It
rests on S44, whose authors overlap with S45 and S43. AgentDojo and the MCP studies do not
themselves test adaptive bypass of in-band defenses. WORKING HYPOTHESIS that architectural (harness-level) isolation is sufficient.

**Evidence.**
- AgentDojo (NeurIPS 2024 D&B): 97 user tasks and 629 security cases across banking, Slack,
  travel and workspace suites, designed for adaptive attacks and defenses [S43].
- Nasr, Carlini et al. (October 2025, USENIX Security 2026 per its proceedings link): adaptive
  attacks bypassed 12 published defenses, most above 90% attack success, human red-teaming at
  100%, against defenses that reported near-zero success originally [S44, search-only for per-
  defense numbers]. Authors span OpenAI, Anthropic and Google DeepMind.
- CaMeL (SaTML 2026): separates control flow (from the trusted query) from untrusted data, tracks
  capabilities, and solves 77% of AgentDojo tasks with provable security vs 84% undefended [S45].
- MCP ecosystem: MCPTox measured 36.5% average tool-poisoning success across 20 models and 45
  live servers, up to 72.8% for one model [S46, search-only]. A static study of 1,899 MCP servers
  found 7.2% with general vulnerabilities and 5.5% with MCP-specific attack vectors [S47,
  search-only].

**Open debate.** Whether architectural isolation scales to open-ended tasks without a large
utility cost. CaMeL's 7-point utility loss is on AgentDojo, whose tasks suit explicit planning.

### C11. Real-world productivity effect

**Label:** OPEN. No effect is asserted, so the replication test does not apply.

**Evidence.**
- METR RCT (July 2025): 16 experienced open-source developers, 246 tasks, average about 2 hours
  each, in repositories averaging 22,000+ stars. With early-2025 tools (mainly Cursor with Claude
  3.5 and 3.7 Sonnet) tasks took 19% longer (CI +2% to +39%). Developers forecast a 24% speedup
  beforehand and believed afterwards they had been sped up 20% [S48].
- METR follow-up (notice of 2026-02-24): 57 developers, 800+ tasks, 143 repositories. Point
  estimates of -18% (CI -38% to +9%) for returning developers and -4% (CI -15% to +9%) for new
  ones. The sign flipped from the 2025 study: negative means time saved, so these are speedups.
  METR says the design no longer works: developers declined to join or to submit tasks
  because they did not want to work without AI (30% to 50% skipped some tasks), so the estimate
  is likely a lower bound on the true speedup and a bad proxy [S49]. METR believes developers are now more sped up,
  based on participant conversations, which is not measured evidence.
- Earlier RCTs of Copilot-style completion (not agentic harnesses): 55.8% faster on a controlled
  JavaScript task, n = 95 [S50, prior-knowledge], and about 26% more completed tasks in field
  experiments at Microsoft, Accenture and a Fortune 100 firm [S51, prior-knowledge].
- METR time horizons: 50%-success task length doubled about every 7 months from 2019 to 2024
  [S53], with a January 2026 update estimating a post-2023 doubling time of about 4.3 months [S54,
  search-only]. Time horizons are measured with METR's own scaffolds, so they are
  model-plus-harness measurements, not model-only.

**Uncertainty.** No current RCT measures agentic harnesses (Claude Code, Codex and similar) in
the field with a design that survives selection effects. The perception-reality gap from 2025
means self-reports should not be treated as productivity data.

### C12. Harnesses as a serving workload

**Label:** WORKING HYPOTHESIS.

**Evidence.** Autellix treats agent programs as scheduling units and reports 4 to 15x throughput
over vLLM by reducing head-of-line blocking at program and call level [S52, search-only]. Agents
use about 4x and multi-agent systems about 15x the tokens of chat in Anthropic's data [S26].
Newer systems (Continuum, SMetric, TOPAS, all 2025 to 2026 preprints) target KV retention across
tool calls and session-level scheduling (search-only, not reviewed in depth).

**Quality notes.** Gains are reported by each system's authors against vanilla baselines on
synthetic or replayed traces. I found no public, representative trace of production agent
harness traffic, which limits how far any of these results generalize. This matters for this
thesis (Section 7).

---

## 5. Evidence quality assessment

| Dimension | State of the literature |
|---|---|
| Study design | Mostly observational benchmark runs. A few pre-registered or factorial designs in late 2025 and 2026 (S12, S17, S24). One RCT line on human productivity (S48, S49) |
| Sample sizes | Small. 50 to 500 tasks, 1 to 3 attempts per task is typical. Exceptions: HAL (21,730 rollouts), Kim et al. (260 configurations), MAST (1,600+ traces) |
| Statistics | CIs and significance tests are the exception (S11, S13, S34, S48, S49). Most papers report point estimates with no seed variance |
| Replication | Direct replications are rare. Convergent evidence from different designs exists for C1 (direction), C5, C8, C9 and C10 |
| Effect sizes | Accuracy effects of harness choice span 0 to 28 points. Cost effects span 1.2x to 40x. Multi-agent effects span -70% to +81% |
| Biases | Vendor self-evaluation on private evals (S26, S9, S39). Benchmark contamination (C8). Publication bias toward positive harness results. Leaderboard effort confounds. Many single-author preprints in 2026 |
| Reproducibility | Weak. Closed models are deprecated, so exact re-runs expire. Private suites (S13) prevent contamination but block outside replication. HAL's release of 2.5B tokens of logs is an exception |

---

## 6. Areas of active debate

1. **Does the harness matter less as models improve?** "Bitter lesson" arguments say yes.
   Starace found capable models were not less scaffold-sensitive [S12], and Meta-Harness found
   large gains from harness search on frontier models [S18]. Unresolved.
2. **Are multi-agent gains just extra compute?** One budget-matched frontier comparison exists
   (Kim et al. [S24], a preprint) and finds large gains and large losses, so compute alone does
   not explain the pattern. Independent replication is missing.
3. **Summarize, mask or curate context?** Three recent studies recommend three different
   strategies [S17, S20, S22].
4. **Native-harness advantage.** Vendor benchmark tables imply one. The only controlled test
   found none on average [S13].
5. **Can harness disclosure become a norm?** Position papers ask for it [S10, S33]. No venue
   requires it yet, as far as I found.
6. **Are architectural security defenses usable?** CaMeL shows the trade-off on one benchmark.
   Broader evidence is missing.
7. **What is the real productivity effect?** METR says its own design has stopped working [S49].

---

## 7. Relevance to this thesis (INTERPRETATION, not a decision)

- The journal already holds I-28 (agent-aware GPU scheduling, PIVOT). C12 is the part of this
  review closest to the thesis focus. The gap that looks real is the lack of representative,
  public traces of harness-generated inference traffic (tool-call pauses, prefix reuse across
  turns, sub-agent fan-out). This is my reading of what I did *not* find, and "I did not find"
  is not proof of absence (journal rule 0.2.6). No idea is registered from this review. If the
  user wants one, it goes through the `research-gap-dossier` workflow with kill criteria first.
- C8 and the harness-effect results are also a methodological warning for I-28: any
  agentic-workload evaluation in the thesis should report the harness, pass^k and token cost,
  not only pass@1.
- "Same Request, Different Answer" (2609.04748, already in Section 7.1 of the journal) showed
  prefix caching changes agent trajectories in 36.2% of episodes at 16-bit and 75.0% at 4-bit
  (80 episodes). That is a serving-side
  analogue of the harness confound: the infrastructure is part of the agent.

---

## 8. Claims excluded or downgraded

- **"Models trained in one harness score 53.5 to 57.9% on it and collapse to 19.0 to 28.0% in a
  mismatched harness."** This appeared in search summaries attributed to several 2026 technical
  reports. I could not find it in the KAT-Coder-V2.5 report's text or trace it to a primary
  source. Not used as evidence. Do not cite until the source is found.
- **Mindstudio, futureagi, Medium and similar blog summaries** of harness effects were seen in
  searches and not used.
- **Vats and Golev date.** The arXiv abstract page shows v1 submitted 8 June 2026 under ID
  2607.22585, which implies July. The numbers are used, the date is uncertain.
- **Terminal-Bench and SWE-bench Pro harness pairs** come from leaderboards and secondary pages,
  not controlled studies. They illustrate C1 but carry no weight on their own and are not listed
  as support for it. Corrected 2026-09-30 after review R-06: the earlier text gave the
  Terminal-Bench 2.1 scores (70.1 vs 63.8) as 2.0, where the order is reversed (58.0 vs 62.9),
  and mixed models in the SWE-bench Pro range.

---

## 9. Sources

Check levels: checked, search-only, prior-knowledge (defined in Section 0).

| ID | Source | Venue and date | Check |
|---|---|---|---|
| S1 | Guo et al., "From Question Answering to Task Completion: A Survey on Agent System and Harness Design", [arXiv 2606.20683](https://arxiv.org/abs/2606.20683) | preprint, 2026-06-14 | checked |
| S2 | Anthropic, "Building effective agents", [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents) | vendor blog, 2024-12 | prior-knowledge |
| S3 | Yao et al., "ReAct", [arXiv 2210.03629](https://arxiv.org/abs/2210.03629) | ICLR 2023 | prior-knowledge |
| S4 | Shinn et al., "Reflexion", [arXiv 2303.11366](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 | prior-knowledge |
| S5 | Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", [arXiv 2405.15793](https://arxiv.org/abs/2405.15793) | NeurIPS 2024 | checked |
| S6 | Xia et al., "Agentless: Demystifying LLM-based Software Engineering Agents", [arXiv 2407.01489](https://arxiv.org/abs/2407.01489) | preprint 2024, later FSE 2025 (prior-knowledge) | checked |
| S7 | Kapoor, Stroebl, Siegel, Nadgir, Narayanan, "AI Agents That Matter", [arXiv 2407.01502](https://arxiv.org/abs/2407.01502) | TMLR | search-only |
| S8 | Kapoor, Stroebl, Kirgis et al., "Holistic Agent Leaderboard", [arXiv 2510.11977](https://arxiv.org/abs/2510.11977) | ICLR 2026 | checked |
| S9 | Anthropic, "Effective harnesses for long-running agents", [anthropic.com/engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | vendor blog, 2025-11-26 | search-only |
| S10 | Zhang, Wang, Ge, Xu, Hamm, Reddy, "Stop Comparing LLM Agents Without Disclosing the Harness", [arXiv 2605.23950](https://arxiv.org/abs/2605.23950) | position preprint, 2026-05-07 | checked |
| S11 | Vats, Golev, "The Scaffold Effect in Coding Agents", [arXiv 2607.22585](https://arxiv.org/abs/2607.22585) | preprint, v1 2026-06-08 (ID implies July) | checked (abstract: 0 to 8 points, 40x tokens, CIs). Per-harness token counts and the no-action-turn figure are search-only (Emergent Mind summary) |
| S12 | Starace, "Scaffold Effects on GAIA: A Controlled Comparison", [arXiv 2606.08529](https://arxiv.org/abs/2606.08529) | pre-registered preprint, 2026-06-07 | checked |
| S13 | Arjmandi, "Harness or Model? Isolating the Harness Effect in Agentic Coding", [arXiv 2609.11987](https://arxiv.org/abs/2609.11987) | preprint, 2026-09-08 | checked |
| S14 | Merrill et al., "Terminal-Bench", [arXiv 2601.11868](https://arxiv.org/abs/2601.11868) | preprint, 2026-01-17 | checked (abstract, covers version 2.0). Harness pair numbers are from the [tbench.ai 2.1 release notes](https://www.tbench.ai/news/terminal-bench-2-1), search-only |
| S15 | Wang et al., "Executable Code Actions Elicit Better LLM Agents" (CodeAct), [arXiv 2402.01030](https://arxiv.org/abs/2402.01030) | ICML 2024 | checked |
| S16 | mini-swe-agent, [github.com/SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | project README | search-only |
| S17 | Fan et al., "An Empirical Study of Harness Design for Coding Agents", [arXiv 2609.20804](https://arxiv.org/abs/2609.20804) | preprint, 2026-09-17 | checked |
| S18 | Lee et al., "Meta-Harness: End-to-End Optimization of Model Harnesses", [arXiv 2603.28052](https://arxiv.org/abs/2603.28052) | preprint, 2026-03-30 | checked |
| S19 | Liu et al., "Lost in the Middle", [arXiv 2307.03172](https://arxiv.org/abs/2307.03172) | TACL 2024 | prior-knowledge |
| S20 | Lindenbauer et al., "The Complexity Trap", [arXiv 2508.21433](https://arxiv.org/abs/2508.21433) | NeurIPS 2025 DL4Code workshop | search-only |
| S21 | Chroma, "Context Rot", [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot) | industry report, 2025 | search-only |
| S22 | Zhang et al., "Agentic Context Engineering", [arXiv 2510.04618](https://arxiv.org/abs/2510.04618) | ICLR 2026 (arXiv comments line) | search-only |
| S23 | "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection", [arXiv 2505.03275](https://arxiv.org/abs/2505.03275) | preprint, 2025-05 | search-only |
| S24 | Kim et al., "Towards a Science of Scaling Agent Systems", [arXiv 2512.08296](https://arxiv.org/abs/2512.08296) | preprint, 2025-12 | checked |
| S25 | Cemri, Pan, Yang et al., "Why Do Multi-Agent LLM Systems Fail?", [arXiv 2503.13657](https://arxiv.org/abs/2503.13657) | NeurIPS 2025 D&B | search-only |
| S26 | Anthropic, "How we built our multi-agent research system", [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system), also summarized by [Willison](https://simonwillison.net/2025/Jun/14/multi-agent-research-system/) | vendor blog, 2025-06 | search-only |
| S27 | Smit et al., "Should we be going MAD?", [arXiv 2311.17371](https://arxiv.org/abs/2311.17371) | ICML 2024 (prior-knowledge) | search-only |
| S28 | "Stop Overvaluing Multi-Agent Debate", [arXiv 2502.08788](https://arxiv.org/abs/2502.08788) | preprint, 2025-02 | search-only |
| S29 | Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet", [arXiv 2310.01798](https://arxiv.org/abs/2310.01798) | ICLR 2024 | search-only |
| S30 | Kamoi et al., "When Can LLMs Actually Correct Their Own Mistakes?", [TACL 2024](https://aclanthology.org/2024.tacl-1.78/) | TACL 2024 | search-only |
| S31 | Brown et al., "Large Language Monkeys", [arXiv 2407.21787](https://arxiv.org/abs/2407.21787) | preprint 2024 | search-only |
| S32 | Scale AI, SWE-Bench Pro leaderboard, [labs.scale.com](https://labs.scale.com/leaderboard/swe_bench_pro), numbers via secondary pages | leaderboard | search-only |
| S33 | Zhang, Motaali et al., "The Double Measurement Confound in Agent Benchmarks", [arXiv 2609.09218](https://arxiv.org/abs/2609.09218) | preprint, 2026-09-06 | checked |
| S34 | Le et al., "What Does Multi-Harness RL Learn?", [arXiv 2609.04518](https://arxiv.org/abs/2609.04518) | preprint, 2026-09-03 | checked |
| S35 | Zhang, Lin, Zhang, Fetahu, Chen, "CHART: A Harness-Rotation Curriculum", [arXiv 2609.22247](https://arxiv.org/abs/2609.22247) | preprint, 2026-09-05 | checked |
| S36 | Zhu et al., "Establishing Best Practices for Building Rigorous Agentic Benchmarks", [arXiv 2507.02825](https://arxiv.org/abs/2507.02825) | NeurIPS 2025 | search-only |
| S37 | Aleithan et al., "SWE-Bench+", [arXiv 2410.06992](https://arxiv.org/abs/2410.06992) | preprint 2024 | search-only |
| S38 | Liang et al., "The SWE-Bench Illusion", [arXiv 2506.12286](https://arxiv.org/abs/2506.12286) | ICSE 2026 SEIP | search-only |
| S39 | OpenAI, "Why SWE-bench Verified no longer measures frontier coding", [openai.com](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | vendor post, 2026-02 | search-only (page returned 403) |
| S40 | International AI Safety Report 2026 | expert report, 2026-02 | search-only |
| S41 | Yao et al., "tau-bench", [arXiv 2406.12045](https://arxiv.org/abs/2406.12045) | preprint 2024 | search-only |
| S42 | "Towards a Science of AI Agent Reliability", [arXiv 2602.16666](https://arxiv.org/abs/2602.16666) | ICML 2026 | checked |
| S43 | Debenedetti et al., "AgentDojo", [NeurIPS 2024 D&B](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2024 | search-only |
| S44 | Nasr, Carlini et al., "The Attacker Moves Second", [arXiv 2510.09023](https://arxiv.org/abs/2510.09023) | USENIX Security 2026 | search-only |
| S45 | Debenedetti et al., "Defeating Prompt Injections by Design" (CaMeL), [arXiv 2503.18813](https://arxiv.org/abs/2503.18813) | SaTML 2026 | search-only |
| S46 | MCPTox tool-poisoning benchmark (2025-08) | preprint | search-only |
| S47 | Hasan et al., empirical study of 1,899 MCP servers | preprint 2025 | search-only |
| S48 | Becker, Rush, Barnes, Rein, "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity", [arXiv 2507.09089](https://arxiv.org/abs/2507.09089) | preprint, 2025-07-10 | search-only |
| S49 | METR, "We are Changing our Developer Productivity Experiment Design", [metr.org](https://metr.org/blog/2026-02-24-uplift-update/) | 2026-02-24 | checked |
| S50 | Peng et al., "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot", [arXiv 2302.06590](https://arxiv.org/abs/2302.06590) | preprint 2023 | prior-knowledge |
| S51 | Cui, Demirer et al., "The Effects of Generative AI on High-Skilled Work: Evidence from Three Field Experiments with Software Developers" | SSRN 2024 | prior-knowledge |
| S52 | Luo et al., "Autellix", [arXiv 2502.13965](https://arxiv.org/abs/2502.13965) | preprint 2025-02 | search-only |
| S53 | METR, "Measuring AI Ability to Complete Long Tasks", [metr.org](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) | 2025-03-19 | search-only |
| S54 | METR, "Time Horizon 1.1", [metr.org](https://metr.org/blog/2026-1-29-time-horizon-1-1/) | 2026-01-29 | search-only |
| S55 | Yehudai et al., "Survey on Evaluation of LLM-based Agents", [arXiv 2503.16416](https://arxiv.org/abs/2503.16416) | ACL Findings 2025 | search-only. Identifies cost-efficiency, safety and robustness as the main evaluation gaps, consistent with C2, C9, C10 |

**Before citing any of these in the thesis:** upgrade search-only and prior-knowledge entries to
checked by opening the primary page, and re-confirm the 2026 preprints for revisions.
