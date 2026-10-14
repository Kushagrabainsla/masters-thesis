# R-19: Assumption removal (2026-10-11)

Asked by the user on 2026-10-08 ("Lets try one more time, this time try to think of removing
assumptions... Lets do that for AI infra as well"), approved 2026-10-11 ("sure, go ahead!").
Method: list the assumptions that current AI infrastructure makes, keep the ones that are breaking now
(with evidence), and keep only those whose removal breaks something concrete (a wrong result, a crash,
a measurable waste). Then check prior art, attack with Sonnet, and design a premise check for the best
survivor. Labels: FACT, SNIPPET, INTERPRETATION.

## 1. The assumption list (main session, first principles)

Triage key: KEEP (goes to the prior-art check), CROWDED (known to be crowded from R-09 to R-18 or from
this session's searches), WEAK (not clearly breaking, or nothing concrete breaks).

| # | Assumption | What removes it | Triage |
|---|---|---|---|
| A1 | The model version is fixed for a session | Long agent sessions, frequent weight updates | KEEP (P4) |
| A2 | A human reads the output as it streams | Agents and programs consume whole outputs | KEEP (P5) |
| A3 | Requests are independent; the server cannot see the future | Agent programs expose structure | CROWDED (Parrot, Autellix, KVFlow) |
| A4 | Actions have no side effects | Agents act on the world | CROWDED (sagas, durable execution, I-74) |
| A5 | Each token costs about the same | Reasoning, MoE, sparse attention, multimodal | WEAK (cost models exist) |
| A6 | The server holds no session state | Server-side conversation state | KEEP (P7) |
| A7 | The GPU is the only expensive resource | Agent sandboxes, CPU paths | CROWDED (R-18) |
| A8 | One model serves a request | Compound AI, routers, cascades | CROWDED |
| A9 | Weights are read-only while serving | Online RL in production, test-time training | KEEP (merged into P4) |
| A10 | GPUs in a job are homogeneous and co-located | Heterogeneous and cross-region jobs | CROWDED |
| A11 | The datacenter can draw full power at any time | Grid demand response | KEEP (P2) |
| A12 | Failures are independent and fail-stop | Fail-slow, silent corruption | CROWDED |
| A13 | The job writes its own checkpoints | Transparent C/R | CROWDED (I-35, I-99) |
| A14 | Training data is static and preprocessed | RL environments generate data online | CROWDED |
| A15 | A token's result does not depend on its batch | Batch-variant kernels | CROWDED (I-30) |
| A16 | A model is its weights | Weights plus kernels plus precision | Our I-23 line |
| A17 | The operator is trusted | Confidential GPUs | KEEP (P6) |
| A18 | One provider serves a whole session | Multi-provider routing and failover | WEAK (R-18 T03, small) |
| A20 | Inference has only latency SLOs | Background agents with loose deadlines | CROWDED (batch APIs, BatchGen) |
| A21 | Parallelism is fixed during serving | Elastic serving | CROWDED |
| A22 | Cluster users are humans submitting a few jobs | AI agents running many short experiments | KEEP (P3) |
| A24 | Inference runs only on servers | Device and cloud split | CROWDED |
| A25 | Context is private to one session | Shared multi-agent context | CROWDED (KVCOMM, DroidSpeak) |
| A27 | The GPU network is lossless and local | Cross-DC training | CROWDED (DiLoCo) |
| A29 | Arrivals are independent | Correlated agent bursts | CROWDED (I-80) |
| A30 | KV is consumed by the model that made it | Cross-model KV | CROWDED |
| A33 | Compute providers are honest | Decentralized GPU markets | KEEP (merged into P6) |
| A37 | Checkpoints exist only for recovery | Model branching and merging | CROWDED (Xet dedup) |
| A42 | Prompts are short and written by humans | Machine-generated, repetitive prompts | CROWDED (prefix caching) |
| A43 | GPU kernels are written by experts and are trusted | AI-generated kernels | KEEP (P1) |
| A46 | GPUs of one model perform the same | Chip-to-chip variability | CROWDED (stragglers) |
| A47 | The model is passive; the infrastructure controls it | Agents that manage their own context and compute | CROWDED (Leyline 2606.01065, SGLang #27574) |
| A48 | One model is shared by all users | Per-user adapters that change daily | WEAK (multi-LoRA serving exists) |

## 2. Prior-art checks

- P1 (A43, main session): see Section 3.
- P2, P3, P4: `dossier/R-19_PRIORART_P2-P4_2026-10-11.md` (Sonnet agent).
- P5, P6, P7: `dossier/R-19_PRIORART_P5-P7_2026-10-11.md` (Sonnet agent).

## 3. P1: AI-generated kernels (main session)

- Evidence it is breaking: FACT (SNIPPET for some): FlashInfer-Bench (2601.00227, MLSys 2026) injects
  the best agent-generated kernels into SGLang and vLLM through `apply()`; CAKE (2608.12629, NVIDIA and
  CMU) ports more than 100 TensorRT-LLM kernels; FlashInfer issue #4254 tracks CAKE-generated kernels.
- What breaks: SNIPPET: a contract-grade verifier (2608.12700, 2026-08-13) finds 39.5% of 2,638
  machine-generated kernels that a public harness accepted are broken beyond any tolerance argument.
  Reward hacking is documented (Kevin, ICLR 2026; KernelBench-Verified 2607.16241).
- Prior art on verification: KernelBench-Verified, CUDA-Harness (2609.00058), ProofWright
  (2511.12294), Volta (PTX equivalence), Kernel Contracts (2604.22032), the contract verifier above.
- Remaining gap: offline verification is crowded. Deployment-time checking of substituted kernels on
  real traffic and real input distributions is not described in the FlashInfer-Bench abstract. That is
  the question of I-23 (correctness-aware kernel qualification and selection).
- Verdict: CROWDED as a new idea. It strengthens the motivation of I-23 (the kernel author is no longer
  trusted).
