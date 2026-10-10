# GPU Efficiency for Training and Inference: An Evidence-Graded Literature Review

Date: 2026-09-30. Author: Claude (Opus 5.5) at the request of Kushagra Bainsla.
Journal links: Section 4 entry of 2026-09-30 and Section 7.9 of `RESEARCH_JOURNAL.md`.
Status: literature review. Not a thesis decision. No experiments were run for this report.
Revised 2026-09-30 after a correctness review (journal R-07): two independent reviewers, disputed
items re-checked against the primary PDFs. Fixes are listed in Section 8.
Companion: a Claude Doc version with the MFU chart and the evolution diagram drawn,
https://claude.ai/code/artifact/4b94def3-488a-453a-a2fe-331981c8781e . This file is the repo record
and is self-contained; the chart data is reproduced as tables here. If the two ever disagree,
this file plus the journal win (journal rule 0.8).

---

## 0. Scope, method and how to read this

**Scope.** Efficiency of NVIDIA-class GPUs (and TPUs where they inform the same question) for
transformer training and LLM inference, 2017 to September 2026. Out of scope: chip design,
non-transformer workloads, price forecasting.

**Method.** arXiv, conference and operator pages opened on 2026-09-30 (S2 excepted, see Section 9).
For 15 papers the PDF was downloaded and the relevant passages were read in the body text. During
the R-07 correctness review the PDFs for S1, S10, S14, S18, S19, S34, S47 and the S44 blog page were
re-read for the corrected claims, but their check levels in Section 9 are left as first assigned.
Every source in Section 9
carries a check level (same scale as journal Section 0.4, plus prior-knowledge):

| Level | Meaning |
|---|---|
| full-text | The relevant passage was read in the paper body (PDF text extracted and searched) |
| checked | Abstract or landing page opened on 2026-09-30 and the cited numbers confirmed there |
| search-only | Numbers come from search-result text only. None are used as evidence in this report |
| prior-knowledge | Well-known work cited from training knowledge, not re-opened this session |

**Evidence labels used for each claim.**

| Label | Meaning in this report |
|---|---|
| CONSENSUS | Mechanism understood, reproduced by independent groups on more than one hardware generation, and shipped in production stacks |
| WORKING HYPOTHESIS | Supported by several credible results, but magnitude depends on workload, hardware or baseline, or replication is partial |
| SPECULATIVE | Single-source, vendor-only or extrapolated evidence |

CONSENSUS here plays the role ESTABLISHED plays in `LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md`.

**Four limits to state up front.**

1. **This field has no meta-analyses, RCTs or formal consensus statements.** Its evidence is
   benchmark papers, vendor and lab technical reports, and a few cluster trace studies. MLCommons
   (MLPerf) is the nearest thing to a standards body, and it publishes benchmark results, not
   consensus statements. "CONSENSUS" means "replicated and adopted", not "settled by a process".
2. **n is tiny.** A typical systems study covers 2 to 6 models on 1 to 3 hardware types. Most
   report no confidence intervals and no run-to-run variance.
3. **Baselines move and authors configure them.** A 2x claim against vLLM version X may be 1.2x
   against version X+2. Independent replication is rare outside MLPerf and a few benchmarks such
   as Spec-Bench [S39].
4. **Production reports are large in GPU-hours but n = 1 in organizations.** Meta, ByteDance,
   Google and Moonshot report their own clusters with their own metric definitions. None is
   audited.

I did not rerun any experiment. There is no NVIDIA GPU available to this project (journal
Section 1.1).

---

## 1. Executive summary

| # | Claim | Label | Strongest support | Main uncertainty |
|---|---|---|---|---|
| C1 | Peak FLOPS has outgrown memory and interconnect bandwidth | CONSENSUS | Gholami et al. 3.0x vs 1.6x vs 1.4x per 2 years [S3]; H100 datasheet ratio [S4]; Pope et al. [S29] | Trend fit; exact rates depend on which parts are sampled |
| C2 | Small-batch decode is memory-bandwidth-bound | CONSENSUS | Roofline [S2]; Pope et al. [S29]; vLLM [S31] | None on direction |
| C3 | nvidia-smi "GPU utilization" is a weak efficiency signal | WORKING HYPOTHESIS (downgraded in R-07) | Hu et al. note it in one passing sentence and cite other work [S23]; PaLM's motivation for MFU [S1] | Thin direct evidence. The 25% power-sampling result [S5] is about power readings, not utilization |
| C4 | Reported training MFU at 2K to 16K GPUs is about 38 to 59%. The top end rose only from 56.3% (2022) to 59.1% (2024), and the 2024 figure uses model and optimizer changes. The only H100 point at 8K to 16K GPUs (Llama 3) is 38 to 43% | CONSENSUS (range) | PaLM 46.2% [S1]; Korthikanti 54.2 to 56.3% [S10]; MegaScale 55.2 to 59.1% [S18]; Llama 3 38 to 43% [S20] | FLOP-counting conventions differ; MegaScale changes the model and batch; no variance reported |
| C5 | 16-bit mixed precision matches FP32 quality | CONSENSUS | Micikevicius 2018 [S7]; used in [S1], [S19], [S20] | None |
| C6 | IO-aware fused attention is a large, general win | CONSENSUS | FlashAttention 3x on GPT-2 [S11]; FlashAttention-3 75% of H100 peak [S12] | One lab lineage, but adopted everywhere |
| C7 | Continuous batching plus paged KV raise serving throughput several-fold over static batching | CONSENSUS | Orca 36.9x vs FasterTransformer [S30]; vLLM 2 to 4x [S31] | Magnitude depends on baseline age |
| C8 | Speculative decoding gives about 2x at low batch | CONSENSUS | Leviathan 2 to 3x [S37]; Chen 2 to 2.5x [S38]; Spec-Bench 1.8 to 2.4x, independent [S39] | High-batch regime contested (debate 4, Section 6) |
| C9 | Large synchronous jobs fail on hour scales; MTTF falls about as 1/N | CONSENSUS | Kokolis 7.9 h at 1,024 GPUs [S22]; Llama 3 466 interruptions in 54 days [S20]; OPT 35+ manual restarts [S21]; Hu et al. [S23] | Rates vary by cluster and year. The 1/N fit comes from Kokolis alone; the others are qualitative |
| C10 | FP8 training matches BF16 with a careful recipe | WORKING HYPOTHESIS | Micikevicius 2022 [S13]; DeepSeek <0.25% loss gap, measured on 16B and 230B MoE models, not on V3 itself [S19]; counterpoint Fishman, where standard FP8 diverges after about 200B tokens [S14] | Long-horizon failures are recent; recipes are model-specific; no BF16 comparison exists for the 671B run |
| C11 | Prefill/decode disaggregation improves goodput | WORKING HYPOTHESIS | DistServe [S34]; Splitwise [S35]; Mooncake production [S36] | All self-reported; no neutral comparison with chunked prefill [S33] |
| C12 | FP8 inference quantization is near-lossless for current LLMs; INT8 loses 1 to 3% and INT4 weight-only is competitive | WORKING HYPOTHESIS | Kurtic 500K+ evaluations [S42]; AWQ for 4-bit weights [S43]; counterpoint Kumar [S16] | One model family; over-trained future models may degrade more |
| C13 | Stragglers waste about 10% of LLM GPU-hours, mostly from imbalance, not hardware | WORKING HYPOTHESIS | Lin et al. [S24]; MegaScale needed straggler handling [S18] | One operator's trace for the number |
| C14 | Compute needed to reach a fixed perplexity on WikiText and Penn Treebank halves about every 8 months | WORKING HYPOTHESIS | Ho et al., 95% CI 5 to 14 months [S52]; Hernandez & Brown 16 months for vision [S51] | Observational fits on heterogeneous published results; perplexity on two datasets only; authors call the data noisy |
| C15 | A typical text prompt costs about 0.2 to 0.3 Wh | WORKING HYPOTHESIS | Google 0.24 Wh median [S45]; Epoch about 0.3 Wh, range 0.1 to 4 [S46] | Operator self-report plus a model; long-context tail excluded |
| C16 | FP4 (NVFP4) pretraining matches FP8 | SPECULATIVE | NVIDIA, 12B model, 10T tokens [S15] | Single vendor, single run |

Eight claims are CONSENSUS, seven WORKING HYPOTHESIS, one SPECULATIVE. (The first draft had nine,
six and one. C3 was downgraded in R-07.)

The one-sentence version: **GPU efficiency is now limited by moving bytes and keeping thousands
of GPUs in step, not by raw FLOPs. The mechanisms are settled; the size of most claimed gains is
not, because the evidence is mostly self-reported benchmarks on a few models against baselines the
authors chose.**

---

## 2. Measuring efficiency: which number means what

There is no single "GPU efficiency" number. Headline claims often switch between these five.

| Metric | What it measures | Blind spot | Status |
|---|---|---|---|
| nvidia-smi "GPU utilization" | Fraction of sample time in which any kernel was running | A kernel using 1 SM or stalled on memory still counts as 100% | Weak indicator; Hu et al. say so explicitly [S23] |
| MFU (model FLOPs utilization) | Useful model FLOPs per second over peak FLOPs | FLOP convention; dense vs sparse peak | Standard training metric since PaLM [S1] |
| HFU (hardware FLOPs utilization) | All executed FLOPs including recomputation, over peak | Rewards wasted work | Reported next to MFU |
| Goodput / ETTR | Training: useful time over wall time [S22]. Serving: max request rate meeting SLOs per GPU [S34] | Two unrelated meanings share one word | Working definitions, not standardized |
| Energy per token or per request | Joules at device, node or facility boundary | Boundary choice moves the answer 2x or more [S49] | No agreed method (C15) |

**MFU definition.** PaLM defines MFU as observed tokens per second times FLOPs per token, over
peak FLOPs, counting forward and backward passes but not rematerialization [S1, Appendix B]:

    MFU = tokens/s x (6N + 12 L H Q T) / peak FLOP/s

N is parameters, L layers, H heads, Q head dimension, T sequence length. Two practical caveats:

- NVIDIA lists sparse peaks. The H100 SXM page gives 1,979 TFLOPS BF16 "with sparsity" [S4], so
  the dense peak for MFU is about 989 TFLOPS. MFU computed against the sparse peak is halved.
- MoE and long-context models break the 6N shortcut, so their MFU figures are not comparable to
  dense-model figures without restating the FLOP count.

**Roofline.** A kernel is capped by the lower of peak compute and bandwidth times arithmetic
intensity [S2]. H100 SXM ridge point: 989 TFLOPS / 3.35 TB/s, about 295 FLOP per byte (my
derivation from [S4]). A BF16 matrix-vector product, which is batch-1 decode, is about 1 FLOP per
byte. That gap of two orders of magnitude is why decode is memory-bound (C2).

**Latency metrics are contested.** TTFT and TPOT are the de facto standard. Etalon argues they
miss user-visible mid-stream stalls and proposes a fluidity index [S6]. No replacement is widely
adopted.

**Power readings are sampled, not integrated.** On A100 and H100, nvidia-smi samples about 25% of
runtime for power. The micro-benchmarks covered 70+ GPUs. Applying the authors' recommended
practices cut energy measurement error against an external meter by 35% on average and up to 65%
in the test cases they present [S5].

---

## 3. How understanding evolved

Each era solved the previous bottleneck and exposed the next one. Eras overlap; years are the
cited papers' publication years.

**2017 to 2019: FLOPs were the limit.** Mixed precision training (FP32 master weights plus loss
scaling, about 2x less memory) made 16-bit standard [S7]. Efficiency meant more tensor-core FLOPs
per dollar and data-parallel scaling. Outside energy estimates from this period were later judged
off by 100x to 100,000x [S48].

**2019 to 2022: memory capacity was the limit.** ZeRO (2020) [S8] and Megatron 3D parallelism
(2021) [S9] sharded models too big for one GPU. Selective recomputation cut activation memory 5x
[S10]. PaLM defined MFU (2022) [S1]. Chinchilla showed parameters and training tokens should scale
together [S53].

**2022 to 2024: memory bandwidth was the limit.** FlashAttention showed attention is bound by HBM
traffic, not FLOPs [S11]. Orca (2022) [S30] and vLLM (2023) [S31] turned serving into a batching
and KV-memory problem. PyTorch FSDP (2023) brought ZeRO-style sharding into the core framework
[S17]. Inference cost entered scaling laws: train smaller models on more tokens when inference
demand is large [S54].

**2024 to 2026: reliability, precision and correctness are the limit.** Llama 3 logged 419
unexpected interruptions in 54 days [S20]; 1,024-GPU jobs fail about every 8 hours [S22]. FP8
training reached production [S19] and FP4 pretraining trials began [S15]. Batch-dependent outputs
[S44] and FP8 accumulation error [S19] became open research questions.

**Beliefs that were revised, not just extended.**

- **Speedup magnitudes shrank on independent re-measurement, for at least one method.** The
  original speculative decoding papers report 2 to 3x on T5-XXL and Chinchilla 70B [S37, S38].
  Those models were not re-measured. On a shared benchmark (Vicuna-7B, one RTX 3090), EAGLE, which
  its authors report at 2.9 to 3.1x, measured 1.8 to 2.4x (2.08x overall). A different method,
  prompt-lookup decoding, got 1.1 to 1.3x on translation and QA [S39].
- **Energy estimates came down.** The often-repeated 3 Wh per chatbot query was revised to about
  0.3 Wh [S46], close to Google's measured 0.24 Wh [S45].
- **"Utilization" was redefined.** From nvidia-smi busy time (Philly, PAI, as reported in [S23]) to
  MFU [S1] to effective training time [S20, S22], because each earlier metric hid a loss the next
  one exposes.
- **Algorithmic progress rivals hardware progress.** Compute for AlexNet-level ImageNet accuracy
  fell 44x from 2012 to 2019 vs 11x from Moore's law [S51]; compute to reach a fixed perplexity on
  WikiText and Penn Treebank halves about every 8 months [S52]. Both are observational fits on
  different measures.

---

## 4. Findings by claim

### C1 and C2. Bandwidth, not FLOPs, bounds decode

**Label:** CONSENSUS.

**Evidence.** Peak server FLOPS grew 3.0x per 2 years vs 1.6x for DRAM bandwidth and 1.4x for
interconnect [S3]. On PaLM 540B, Pope et al. reached 76% MFU on large-batch input processing, while
low-batch generation was latency-bound at 29 ms per token with int8 weights [S29]. vLLM measured
that prior serving systems used only 20.4 to 38.2% of KV-cache memory for actual token states
[S31], showing memory capacity caps the batch.

**Quality notes.** Mechanism follows from arithmetic intensity and holds on every GPU generation.
The Gholami rates are a trend fit over many parts.

### C3. nvidia-smi utilization is weak

**Label:** WORKING HYPOTHESIS. The first draft said CONSENSUS. R-07 downgraded it: the direct
evidence is one hedged sentence in Hu et al. that itself cites other work, plus PaLM's design
choice. The claim is widely held, but this review did not find several independent measurements of
the gap between busy-time utilization and useful work.

**Evidence.** In Shanghai AI Lab's LLM clusters, median nvidia-smi utilization was 97% and 99%,
vs 48% in Microsoft Philly and 4% in Alibaba PAI. Hu et al. use those numbers to show LLM jobs are
compute-intensive and add in passing that the metric "may sometimes be a weak utilization
indicator" [S23]. PaLM introduced MFU to compare systems independent of implementation choices
[S1]. Separately, nvidia-smi power readings have their own sampling problem [S5]; that is a
finding about power, not utilization.

### C4. Reported training MFU at scale is about 38 to 59%, with little rise since 2022

**Label:** CONSENSUS for the range. The first draft said "about 40 to 55%, and the ceiling has not
risen since 2022". R-07 found that MegaScale reports 59.1% at 3,072 GPUs and Korthikanti 56.3%
without data parallelism, both above 55%.

**Evidence (data behind the chart in the companion doc).**

| System | Hardware | Year | MFU | Note | Source |
|---|---|---|---|---|---|
| GPT-3 175B | V100 | 2020 | 21.3% | PaLM's figure uses OpenAI's 24.6 non-attention TFLOP/s per GPU, so it omits attention FLOPs | [S1] Table 3 |
| Gopher 280B | TPU v3 | 2021 | 32.5% | Computed by PaLM from training speed | [S1] Table 3 |
| Megatron 1T | 3,072 A100 | 2021 | 52% of peak | Counts recomputation FLOPs, closer to HFU | [S9] |
| MT-NLG 530B | 2,240 A100 | 2022 | 30.2% with attention (29.7% without) | PaLM computes both; 30.2% matches the formula in Section 2 | [S1] Table 3 and text |
| PaLM 540B | 6,144 TPU v4 | 2022 | 46.2% with attention (45.7% without); HFU 57.8% | | [S1] |
| Megatron, selective recompute, 1T | 512 A100, no data parallelism | 2022 | 56.3% (HFU 57.0%) | Table 5 of [S10] | [S10] |
| Megatron, selective recompute, 530B | 2,240 A100 (8-way data parallel) | 2022 | 54.2% (42.1% with full recompute) | 56.0% at 280 GPUs without data parallelism | [S10] |
| MegaScale 175B | 3,072 GPUs | 2024 | 59.1% | Model changes, see note below | [S18] Table 2 |
| MegaScale 175B | 6,144 GPUs | 2024 | 57.3% | | [S18] Table 2 |
| MegaScale 175B | 12,288 GPUs | 2024 | 55.2% | 1.34x over Megatron-LM | [S18] Table 2 |
| Llama 3 405B | 8K to 16K H100 | 2024 | 38 to 43% | 41% on 16K GPUs vs 43% on 8K | [S20] Table 4 |

**MegaScale is not plain dense training.** Its ablation (Table 3, 175B on 256 GPUs) goes from 47.7%
to 65.3%. The steps include a parallel transformer block (+4.6 points), sliding-window attention
(+1.0) and LAMB with a 3x larger batch (+3.0) [S18]. These change the model, the FLOP count or the
optimizer, so the 55.2 to 59.1% figures are not like-for-like with the Megatron rows.

**Synthesis (INTERPRETATION).** Across the rows the top end moved from 56.3% (2022) to 59.1%
(2024), and the 2024 figure uses model changes. The one H100 data point at 8K to 16K GPUs, Llama 3,
reports lower MFU than the 2022 A100 and TPU v4 runs (41 to 43% at 8K context, 38% in the 128K
context stage [S20] Table 4), consistent with compute outgrowing interconnect (C1). It is one run
with its own parallelism choices, so this is a reading, not a measured cause.

**Quality notes.** Four organizations. FLOP conventions differ (attention FLOPs included or not,
recompute counted or not). No report gives variance across runs.

### C5. 16-bit mixed precision

**Label:** CONSENSUS. FP32 master weights and loss scaling cut memory nearly 2x without accuracy
loss [S7]. BF16 is the training precision in the PaLM and Llama 3 runs [S1, S20]. DeepSeek-V3 is the
exception: FP8 GEMMs with BF16 or FP32 elsewhere [S19], see C10.

### C6. IO-aware kernels

**Label:** CONSENSUS.

**Evidence.** FlashAttention tiles attention in on-chip SRAM: 15% end-to-end on BERT-large vs the
MLPerf record, 3x on GPT-2 at 1K context, 2.4x on long-range arena [S11]. FlashAttention-3 reaches
740 TFLOPS FP16 on H100 (75% utilization), close to 1.2 PFLOPS in FP8, with 2.6x lower numerical
error than a baseline FP8 attention [S12].

**Independence.** Same lab lineage; general adoption is the replication.

### C7. Continuous batching and paged KV

**Label:** CONSENSUS.

**Evidence.** Orca's iteration-level scheduling: 36.9x throughput over FasterTransformer at equal
latency [S30]. vLLM's PagedAttention: 2 to 4x throughput over prior systems [S31]. Both mechanisms
are now baselines in later papers [S33, S34]. SGLang's RadixAttention prefix reuse reports up to
6.4x on shared-prefix workloads [S32].

### C8. Speculative decoding

**Label:** CONSENSUS at low batch. Contested at high batch.

**Evidence.** Leviathan et al.: 2 to 3x on T5-XXL [S37]. Chen et al.: 2 to 2.5x on Chinchilla 70B
[S38]. Spec-Bench, one shared setup (Vicuna-7B, FP16, one RTX 3090): best method EAGLE 1.8 to 2.4x;
prompt-lookup decoding about 2.4x on summarization but 1.1 to 1.3x on translation and QA [S39].
The Spec-Bench survey table lists 2.9 to 3.1x as EAGLE's reported speedup (Vicuna 7B to 33B),
against 2.08x overall measured on Vicuna-7B by Spec-Bench, so this method's speedup was lower on
third-party testing, though the models and hardware differ. Spec-Bench
does not re-measure the T5-XXL or Chinchilla results, and its 1.8 to 2.4x range overlaps the
original 2 to 3x, so it cannot be read as a like-for-like replication of [S37, S38].

**Contrary or moderating evidence.** MagicDec reports up to 2.51x for Llama 3.1-8B at batch 32 to
256 with long contexts, where KV reads dominate [S40]. TurboSpec states speculation "may degrade"
serving performance if added naively [S41].

### C9. Failures at scale

**Label:** CONSENSUS.

**Evidence.**

| Source | Setting | Key figure |
|---|---|---|
| Llama 3 [S20] | 405B, up to 16K H100, 54-day snapshot | 466 interruptions, 419 unexpected; about 78% hardware-related; GPU issues 58.7%, faulty GPU 30.1%, HBM3 17.2%; effective training time above 90% |
| Kokolis et al. [S22] | Meta RSC-1 and RSC-2, 11 months, 4M jobs, 150M+ A100 GPU-hours | MTTF 7.9 h for 1,024-GPU jobs vs 47.7 days for 8-GPU jobs; MTTF proportional to 1/N for jobs above 32 GPUs |
| OPT-175B [S21] | 2 months of training | At least 35 manual and an estimated 70+ automatic restarts; 100+ hosts cycled |
| Hu et al. [S23] | Shanghai AI Lab, 6 months | Infrastructure failures dominate long pretraining jobs; most other errors occur at job start |
| ByteRobust [S25] | 9,600 GPUs, 3 months | 97% ETTR |

**Synthesis (INTERPRETATION).** OPT (2022) reports restart counts, not an effective-training-time
ratio, so the two eras are a rough contrast and not a measured trend. Operators report above 90%
(Meta) and 97% (ByteDance) by 2024 to 2025. These are self-reported with slightly different
definitions and not audited: "achievable with heavy engineering", not typical. Only Kokolis
quantifies the 1/N scaling of MTTF; Llama 3 and OPT are qualitative support.

**Checkpointing systems.** GEMINI checkpoints to peer CPU memory and recovers more than 13x faster
than remote-storage baselines [S26]. ByteCheckpoint reports 54.2x lower average checkpoint stalls,
up to 9.96x faster saves and 8.80x faster loads, and load-time resharding [S27]. Both are
self-reported against their own baselines; neither is benchmarked against the other.

**Silent data corruption.** Established at fleet scale for CPUs [S28]. The Llama 3 report lists
suspected SDC among hardware causes [S20], but GPU-specific SDC rates are not public.

### C10. FP8 training

**Label:** WORKING HYPOTHESIS, close to consensus for well-engineered stacks.

**Evidence.** E4M3 and E5M2 matched 16-bit quality up to 175B parameters with unchanged
hyperparameters [S13]. DeepSeek-V3 (671B MoE) trains with FP8 GEMMs and, in its own words, is the
first to validate such a framework at this scale ("for the first time"; a self-claim). Only the
GEMMs are FP8. The embedding, output head, MoE gating, normalization and attention operators stay
in BF16 or FP32, and master weights, gradients and optimizer states are kept in higher precision,
with some optimizer states in BF16. The "relative loss error vs BF16 below 0.25%" was measured on
two smaller MoE models, about 16B parameters on 1.33T tokens and about 230B on about 0.9T tokens
(Appendix B.1). No BF16 run of the 671B model exists to compare against [S19, full-text].

**What keeps it short of consensus.**
- DeepSeek reports that FP8 GEMM accumulation on H800 keeps about 14 bits. In a preliminary test on
  two random matrices with K = 4096, this gave a maximum relative error of nearly 2%; it is not a
  measurement from a training run. The fix was promoting partial sums to FP32 every 128 elements,
  plus tile-wise (1 x 128) and block-wise (128 x 128) scaling [S19, full-text].
- Fishman et al. found that standard FP8 diverges after about 200B tokens (the text says 220B),
  from SwiGLU outlier amplification. Smooth-SwiGLU let FP8 train stably to 2T tokens, the headline
  scale of the paper, with "up to about 34%" throughput gain on 256 Gaudi2 [S14, full-text re-read
  in R-07].
- Kumar et al. fit precision-aware scaling laws suggesting that training larger models at lower
  precision may be compute-optimal [S16]. One group's fits.

### C11. Prefill/decode disaggregation

**Label:** WORKING HYPOTHESIS.

**Evidence.** DistServe: up to 7.4x more requests or 12.6x tighter SLO [S34]. The 12.6x is against
vLLM on summarization (OPT-66B). Against DeepSpeed-MII, which already uses chunked prefill, the
request-rate range is 1.6x to 7.4x, and the tighter-SLO gains seen in the evaluation text are 1.4x
(code completion) and 2.6x (summarization) [S34, full-text evaluation]. Splitwise:
1.4x throughput at 20% lower cost, or 2.35x throughput at the same power and cost [S35]. Mooncake:
75% more requests in Kimi production [S36].

**Contrary or moderating evidence.** Sarathi-Serve gets much of the interference benefit on
colocated GPUs with chunked prefill: up to 2.6x serving capacity for Mistral-7B on one A100, 3.7x
for Yi-34B on two A100s, 5.6x with pipeline parallelism for Falcon-180B [S33]. No neutral benchmark
compares disaggregation and chunked prefill under the same traces.

### C12. Inference quantization

**Label:** WORKING HYPOTHESIS, strongly supported.

**Evidence.** Kurtic et al., 500,000+ evaluations on Llama 3.1: FP8 W8A8 effectively lossless, INT8
W8A8 1 to 3% degradation, INT4 weight-only rivals 8-bit; W4A16 is most cost-efficient for
synchronous serving, W8A8 for continuous batching [S42]. AWQ: protecting only about 1% of salient
weights greatly reduces 4-bit weight quantization error. Separately, its TinyChat inference
framework reports more than 3x speedup over the Hugging Face FP16 implementation; that speedup is
not an accuracy result [S43]. Only FP8 is reported as effectively lossless; INT8 costs 1 to 3%.

**Counterpoint.** Kumar et al. find post-training quantization damage grows with pretraining data
and can make extra data harmful [S16]. If this holds, heavily over-trained future models will
quantize worse than Llama 3.1.

### C13. Stragglers

**Label:** WORKING HYPOTHESIS (single operator for the number).

**Evidence.** ByteDance trace, January to May 2024: 42.5% of jobs straggle; over 10% of jobs waste
at least 21.3% of allocated GPU-hours; about 1% waste at least 45%; 10.4% of all allocated GPU-hours
are wasted. Main causes: pipeline-stage imbalance, sequence-length imbalance, garbage-collector
pauses. Hardware was not a significant cause. Slowdowns came more from computation than
communication and did not correlate with job size [S24, full-text].

**Tension.** Meta attributes most unexpected *interruptions* to hardware [S20]; ByteDance finds
hardware is not a significant cause of *slowdowns*. Different failure modes on different clusters,
so both can hold.

### C14. Algorithmic progress

**Label:** WORKING HYPOTHESIS. Compute to reach a fixed perplexity on WikiText and Penn Treebank
halved about every 8 months, 95% CI 5 to 14, over 200+ LM evaluations 2012 to 2023; compute scaling
contributed more than algorithms. The authors call their data noisy, and a subset fit on
non-transformers alone was statistically insignificant [S52]. This is a perplexity result on two
datasets, not a general capability measure. Vision: 44x in 7 years, doubling every 16 months
[S51].

### C15. Energy per query and aggregate demand

**Label:** WORKING HYPOTHESIS.

| Estimate | Value | Boundary | Evidence type |
|---|---|---|---|
| Google, Gemini Apps [S45] | 0.24 Wh median text prompt; 33x lower energy and 44x lower carbon over 12 months | Accelerators, host CPU and DRAM, idle capacity, facility overhead | Operator measurement, not audited |
| Epoch AI, GPT-4o [S46] | About 0.3 Wh typical; range 0.1 to 4 Wh; 2.5 to 40 Wh with 10K+ token inputs | Modeled: about 100B active parameters, H100, 10% utilization, 70% of peak power | Outside estimate; revises an older 3 Wh figure about 10x down |
| Samsi et al. [S47] | About 3 to 4 J per output token, LLaMA 65B at 512 max output length on 8, 16 and 32 shards, run mainly on V100 | GPU only (DCGM, rank-0 node energy times node count) | Measured, one model family. The first draft said A100; the paper ran its 8, 16 and 32 shard experiments on V100 |

**Training footprint.** ML stayed under 15% of Google's energy use for three years; four practices
can cut training energy up to 100x and CO2 up to 1000x [S48]. BLOOM 176B: about 24.7 t CO2e from
dynamic power, 50.5 t including manufacturing and idle consumption [S49].

**Power capping.** LLaMA 65B on four A100s, 250 W to 175 W cap: energy down 23.21%, time up 6.7%. A
150 W cap raised time 19.49% [S47]. One model, one GPU type.

**Aggregate.** DOE, summarizing LBNL: data centers used about 4.4% of US electricity in 2023,
projected 6.7 to 12% by 2028 [S50]. The near-2x range reflects unknown chip shipments and
utilization.

**Quality notes.** "Median prompt" hides the long-context and reasoning tail; Epoch's own range
for 10K+ token inputs is roughly 8 to 130x its typical figure. Few papers state their measurement
boundary.

### C16. FP4 pretraining

**Label:** SPECULATIVE. A 12B model on 10T tokens in NVFP4 matched an FP8 baseline in training loss
and downstream accuracy [S15]. One vendor, one run, no independent replication.

---

## 5. Evidence quality assessment

| Dimension | State of the literature |
|---|---|
| Study design | Benchmark comparisons against author-chosen baselines; a few production trace studies (S20, S22, S23, S24). No RCTs, no meta-analyses |
| Sample sizes | 2 to 6 models on 1 to 3 hardware types is typical. Exceptions: Kokolis (4M jobs, 150M+ GPU-hours), Kurtic (500K+ evaluations) |
| Statistics | CIs and variance are the exception (S52 gives a CI; S22 fits Gamma CIs). Most systems papers report point estimates |
| Replication | Independent re-measurement is rare. Where it exists (Spec-Bench S39), EAGLE measured below its reported speedup, but the models and hardware differ, so it is not a clean replication |
| Effect sizes | Serving speedups 1.1x to 36.9x depending on baseline and task; training MFU 21 to 59%; per-query energy 0.1 to 40 Wh depending on assumptions |
| Biases | Author-configured baselines; vendor incentives (S15 NVIDIA on NVFP4, S45 Google on energy); operator self-reports not audited; negative results rarely published |
| Reproducibility | Weak for production claims (private clusters, private traces). Stronger for open kernels and engines (FlashAttention, vLLM, SGLang), which others can rerun |

---

## 6. Areas of active debate

1. **How low can training precision go?** FP8 works with fine-grained scaling and FP32 promotion
   [S19], but the standard recipe diverged after about 200B tokens in [S14]. DeepSeek's BF16
   comparison covers only 16B and 230B models. NVFP4 has one public large run [S15]. Would
   settle it: independent multi-lab FP4 runs over 1T+ tokens with published loss curves.
2. **Does quantization stay cheap as models see more data?** [S42] vs the prediction in [S16].
   Would settle it: the same audit on heavily over-trained small models.
3. **Disaggregation: always, or only at scale?** [S34, S35, S36] vs [S33]. KV transfer cost and two
   pools make the break-even workload- and interconnect-dependent.
4. **Speculative decoding under load.** Established at low batch [S39]; [S40] vs [S41] at high
   batch.
5. **What should serving be measured by?** TTFT and TPOT vs fluidity index [S6] vs SLO goodput
   [S34].
6. **Is MFU the right training target?** It ignores failures [S22] and stragglers [S24]. ETTR
   captures more but definitions are not shared.
7. **How big is AI's energy footprint?** Per-query numbers roughly agree [S45, S46]; aggregate
   projections span almost 2x [S50].
8. **Numerical determinism vs speed.** Thinking Machines got 80 distinct completions from 1,000
   temperature-0 runs of Qwen3-235B-A22B-Instruct-2507 on vLLM, and 1 with batch-invariant kernels.
   Their speed cost is from a separate test, Qwen-3-8B on one GPU, 1,000 sequences of 90 to 110
   output tokens: 26 s for default vLLM, 55 s for unoptimized deterministic vLLM, 42 s with an
   improved attention kernel [S44, blog, not peer-reviewed]. The cost on the 235B model was not
   reported. FP8 accumulation error [S19] is another instance. No agreed
   budget for output drift per unit of speedup. This is where I-23 sits.

---

## 7. Relevance to this thesis (INTERPRETATION, not a decision)

- **Supports I-23's premise.** FP8 accumulation limited to about 14 bits [S19], FP8 instabilities
  that appeared after about 200B tokens of training [S14], batch-dependent outputs [S44] and no
  agreed drift budget (debate 8) are the conditions I-23 assumes. The Fishman failure shows that a
  validation run shorter than about 200B tokens can miss a precision failure; it does not support a
  2T threshold, which the first draft wrongly implied. I found no neutral study that measures how often the fastest kernel for a
  shape is unsafe. "I did not find" is not proof of absence (journal rule 0.2.6); the journal's
  low-to-medium gap confidence stands.
- **Checkpoint and failure recovery is crowded at the top end.** Operators report 90 to 97%
  effective training time with in-house systems [S20, S25]; GEMINI and ByteCheckpoint cover
  in-memory and resharding checkpoints [S26, S27]. Kokolis et al. note small jobs are the majority
  [S22], the less-served niche if this line is reopened.
- **Communication overlap is one source of large-scale training gains, not the only one.**
  DeepSeek-V3 reports a compute-to-communication ratio near 1:1 for cross-node expert parallelism
  and hides it with DualPipe [S19, full-text]. MegaScale's ablation attributes 6.2 of 17.6 MFU
  points (about 35%) to TP, PP and DP overlap and 8.6 points (about 49%) to algorithmic changes
  (parallel block, sliding-window attention, LAMB) [S18, Table 3, 256 GPUs]. Only DeepSeek-V3 makes
  overlap the headline. This supports I-18's motivation only modestly and also shows how much
  production work exists; its low priority still looks right.
- **Measurement rigor is a contribution in itself.** Any GPU result in the thesis should state its
  FLOP convention, dense or sparse peak, power boundary, baseline version and run-to-run variance.
- **Mac-only constraint.** None of these quantitative claims can be reproduced on the current Mac.
  The review sharpens what to measure once Hopper or Blackwell access exists.

No idea was registered from this review. The user was asked (comment on the companion doc) whether
this section should stay tied to I-23 or be neutral across live ideas. Pending.

---

## 8. Claims excluded, corrected or downgraded

- **IEA "Energy and AI" (2025) and the LBNL 2024 report itself** returned HTTP 403. No number from
  either is used. The US aggregate comes from the DOE press release [S50].
- **Megatron 1T "52% of peak"** [S9] counts recomputation FLOPs, so it is shown as HFU-like and not
  compared directly with MFU values.
- **DeepSeek-V3 "2.788M H800 GPU-hours, $5.576M"** [S19] excludes prior research and ablation runs,
  as the report states. Not used as a cost-of-training figure.
- **H100 ridge point (about 295 FLOP per byte)** is my derivation from datasheet peaks [S4], not a
  published measurement.
- **Correction during drafting.** A first draft said DistServe's 7.4x was measured against a vLLM
  without chunked prefill. The paper's evaluation shows 1.6x to 7.4x against DeepSpeed-MII, which
  does use chunked prefill [S34]. The wrong sentence was live in the companion doc for a few
  minutes and was corrected in the same session.
- **Corrections from the R-07 correctness review (2026-09-30).** Two independent reviewers, then the
  disputed items re-read in the primary PDFs or pages. Each item below was wrong in the first
  version of this file.
  1. **Fishman [S14]:** standard FP8 diverged after about 200B tokens (220B in the text), not 2T.
     2T is the size of the run that succeeded with Smooth-SwiGLU. Fixed in C10, debate 1, Section 7.
  2. **DeepSeek-V3 [S19]:** the "<0.25%" BF16 comparison is from 16B and 230B MoE models, not V3.
     "End to end" was wrong: only GEMMs are FP8. The "about 2%" error is from a test on random
     matrices, not a training run.
  3. **Samsi [S47]:** 3 to 4 J per token is from V100 runs (8, 16, 32 shards), not A100.
  4. **MegaScale [S18]:** overlap is about 35% of the ablation gain, not "largely" the 1.34x. The
     55.2% figure includes model and optimizer changes.
  5. **C4 [S10, S18]:** the ceiling claim was wrong. MegaScale reports 59.1% at 3,072 GPUs and
     Korthikanti 56.3% without data parallelism. Rewritten as a 38 to 59% range.
  6. **Thinking Machines [S44]:** the 42 s vs 26 s timing is from Qwen-3-8B, not the 235B model.
  7. **MT-NLG [S1]:** 30.2% with attention, matching the MFU formula used here. 29.7% is without.
  8. **Count and check levels:** 15 papers were read in full text, not 16. S2 is prior-knowledge
     and was not opened, contrary to the Section 9 header.
  9. **DistServe [S34]:** 12.6x tighter SLO is against vLLM. Against MII it is 1.4x to 2.6x.
  10. **AWQ [S43]:** the ">3x" is TinyChat's speedup, not an error reduction.
  11. **Speculative decoding [S37 to S39]:** the "shrinkage" compared different models and methods.
      The supportable case is EAGLE (reported 2.9 to 3.1x, measured 2.08x overall).
  12. **C3 downgraded** from CONSENSUS to WORKING HYPOTHESIS. Tally is now 8, 7 and 1.
  13. **Scope fixes:** Ho et al. is perplexity on two datasets [S52]; Yang et al.'s 35% and 65% are
      from their test cases [S5]; the BF16 claim in C5 no longer says "every large run"; C12 headline
      now says FP8 near-lossless and INT8 1 to 3%; OPT vs Meta effective time is a rough contrast;
      S17 is now cited in Section 3.
  No reviewer claim on the items above failed re-checking. Reviewer claims not acted on: the
  Patterson "100x to 100,000x" scope (line in Section 3; one reviewer flagged it, not re-read), the
  C13 label (single operator for the number, left at WORKING HYPOTHESIS because S18 gives partial
  support), and the C15 "roughly agree" wording in debate 7.
- **Author names removed where not verified** for S45 and S46 (listed by organization).
- **MagicDec hardware** is not stated in the abstract [S40]; only the batch and model are cited.
- **Not used:** blog summaries and vendor marketing pages for vLLM, TensorRT-LLM and SGLang
  comparisons seen in searches.

---

## 9. Sources

Check levels: full-text, checked, prior-knowledge (defined in Section 0). All opened 2026-09-30
except S2, which is prior-knowledge and was not opened. 15 full-text, 38 checked, 1 prior-knowledge.

| ID | Source | Venue and date | Check |
|---|---|---|---|
| S1 | Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", [arXiv 2204.02311](https://arxiv.org/abs/2204.02311) | JMLR 2023 | full-text (Table 3, Appendix B) |
| S2 | Williams, Waterman, Patterson, "Roofline", [CACM](https://doi.org/10.1145/1498765.1498785) | CACM 2009 | prior-knowledge |
| S3 | Gholami et al., "AI and Memory Wall", [arXiv 2403.14123](https://arxiv.org/abs/2403.14123) | IEEE Micro 2024 | checked |
| S4 | NVIDIA, H100 datasheet, [nvidia.com](https://www.nvidia.com/en-us/data-center/h100/) | vendor page | checked |
| S5 | Yang, Adamek, Armour, "Part-time Power Measurements: nvidia-smi's Lack of Attention", [arXiv 2312.02741](https://arxiv.org/abs/2312.02741) | SC 2024 | checked |
| S6 | Agrawal et al., "Etalon", [arXiv 2407.07000](https://arxiv.org/abs/2407.07000) | preprint 2024 | checked |
| S7 | Micikevicius et al., "Mixed Precision Training", [arXiv 1710.03740](https://arxiv.org/abs/1710.03740) | ICLR 2018 | checked |
| S8 | Rajbhandari et al., "ZeRO", [arXiv 1910.02054](https://arxiv.org/abs/1910.02054) | SC 2020 | checked |
| S9 | Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", [arXiv 2104.04473](https://arxiv.org/abs/2104.04473) | SC 2021 | checked |
| S10 | Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", [arXiv 2205.05198](https://arxiv.org/abs/2205.05198) | MLSys 2023 | checked |
| S11 | Dao et al., "FlashAttention", [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | NeurIPS 2022 | checked |
| S12 | Shah et al., "FlashAttention-3", [arXiv 2407.08608](https://arxiv.org/abs/2407.08608) | NeurIPS 2024 | checked |
| S13 | Micikevicius et al., "FP8 Formats for Deep Learning", [arXiv 2209.05433](https://arxiv.org/abs/2209.05433) | preprint 2022 | checked |
| S14 | Fishman et al., "Scaling FP8 Training to Trillion-Token LLMs", [arXiv 2409.12517](https://arxiv.org/abs/2409.12517) | ICLR 2025 | checked |
| S15 | NVIDIA, "Pretraining Large Language Models with NVFP4", [arXiv 2509.25149](https://arxiv.org/abs/2509.25149) | preprint 2025 | checked |
| S16 | Kumar et al., "Scaling Laws for Precision", [arXiv 2411.04330](https://arxiv.org/abs/2411.04330) | preprint 2024 | checked |
| S17 | Zhao et al., "PyTorch FSDP", [arXiv 2304.11277](https://arxiv.org/abs/2304.11277) | VLDB 2023 | checked |
| S18 | Jiang et al., "MegaScale", [arXiv 2402.15627](https://arxiv.org/abs/2402.15627) | NSDI 2024 | checked |
| S19 | DeepSeek-AI, "DeepSeek-V3 Technical Report", [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) | preprint 2024 | full-text (Sections 1, 3.2, 3.3) |
| S20 | Llama Team, Meta, "The Llama 3 Herd of Models", [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) | preprint 2024 | full-text (Section 3.3, Tables 4 and 5) |
| S21 | Zhang et al., "OPT", [arXiv 2205.01068](https://arxiv.org/abs/2205.01068) | preprint 2022 | full-text (Section 2.5) |
| S22 | Kokolis et al., "Revisiting Reliability in Large-Scale Machine Learning Research Clusters", [arXiv 2410.21680](https://arxiv.org/abs/2410.21680) | HPCA 2025 | full-text (metrics, MTTF sections) |
| S23 | Hu et al., "Characterization of Large Language Model Development in the Datacenter", [arXiv 2403.07648](https://arxiv.org/abs/2403.07648) | NSDI 2024 | full-text (Sections 1, 3) |
| S24 | Lin et al., "Understanding Stragglers in Large Model Training Using What-if Analysis", [arXiv 2505.05713](https://arxiv.org/abs/2505.05713) | OSDI 2025 | full-text (Sections 1, 4.1) |
| S25 | ByteDance, "Robust LLM Training Infrastructure at ByteDance" (ByteRobust), [arXiv 2509.16293](https://arxiv.org/abs/2509.16293) | preprint 2025 | checked |
| S26 | Wang et al., "GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints", [amazon.science](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints) | SOSP 2023 | checked |
| S27 | Wan et al., "ByteCheckpoint", [usenix.org](https://www.usenix.org/conference/nsdi25/presentation/wan-borui) | NSDI 2025 | checked |
| S28 | Dixit et al., "Silent Data Corruptions at Scale", [arXiv 2102.11245](https://arxiv.org/abs/2102.11245) | preprint 2021 | checked |
| S29 | Pope et al., "Efficiently Scaling Transformer Inference", [arXiv 2211.05102](https://arxiv.org/abs/2211.05102) | MLSys 2023 | checked |
| S30 | Yu et al., "Orca", [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 2022 | checked |
| S31 | Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM), [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | SOSP 2023 | full-text (Sections 1, 2) |
| S32 | Zheng et al., "SGLang", [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | checked |
| S33 | Agrawal et al., "Sarathi-Serve", [arXiv 2403.02310](https://arxiv.org/abs/2403.02310) | OSDI 2024 | full-text (abstract, Section 1) |
| S34 | Zhong et al., "DistServe", [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) | OSDI 2024 | full-text (abstract, evaluation baselines) |
| S35 | Patel et al., "Splitwise", [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) | ISCA 2024 | full-text (abstract, Section 1) |
| S36 | Qin et al., "Mooncake", [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) | FAST 2025 | checked |
| S37 | Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) | ICML 2023 | full-text |
| S38 | Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", [arXiv 2302.01318](https://arxiv.org/abs/2302.01318) | preprint 2023 | full-text |
| S39 | Xia et al., "Unlocking Efficiency in LLM Inference" (Spec-Bench), [arXiv 2401.07851](https://arxiv.org/abs/2401.07851) | ACL Findings 2024 | full-text (comparative evaluation) |
| S40 | Sadhukhan et al., "MagicDec", [arXiv 2408.11049](https://arxiv.org/abs/2408.11049) | preprint 2024 | checked |
| S41 | "TurboSpec", [arXiv 2406.14066](https://arxiv.org/abs/2406.14066) | preprint 2024 | checked |
| S42 | Kurtic et al., "Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization", [arXiv 2411.02355](https://arxiv.org/abs/2411.02355) | preprint 2024 | checked |
| S43 | Lin et al., "AWQ", [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | MLSys 2024 (best paper) | checked |
| S44 | Thinking Machines, "Defeating Nondeterminism in LLM Inference", [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | blog, 2025-09 | checked |
| S45 | Google, "Measuring the Environmental Impact of Delivering AI at Google Scale", [arXiv 2508.15734](https://arxiv.org/abs/2508.15734) | preprint 2025-08 | checked |
| S46 | Epoch AI, "How Much Energy Does ChatGPT Use?", [epoch.ai](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use) | 2025 | checked |
| S47 | Samsi et al., "From Words to Watts", [arXiv 2310.03003](https://arxiv.org/abs/2310.03003) | IEEE HPEC 2023 | full-text (energy per token, power capping) |
| S48 | Patterson et al., "The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink", [arXiv 2204.05149](https://arxiv.org/abs/2204.05149) | IEEE Computer 2022 | checked |
| S49 | Luccioni et al., "Estimating the Carbon Footprint of BLOOM", [arXiv 2211.02001](https://arxiv.org/abs/2211.02001) | JMLR 2023 | checked |
| S50 | US DOE, "DOE Releases New Report Evaluating Increase in Electricity Demand from Data Centers", [energy.gov](https://www.energy.gov/articles/doe-releases-new-report-evaluating-increase-electricity-demand-data-centers) | press release, 2024-12 | checked |
| S51 | Hernandez, Brown, "Measuring the Algorithmic Efficiency of Neural Networks", [arXiv 2005.04305](https://arxiv.org/abs/2005.04305) | preprint 2020 | checked |
| S52 | Ho et al., "Algorithmic Progress in Language Models", [arXiv 2403.05812](https://arxiv.org/abs/2403.05812) | NeurIPS 2024 | checked |
| S53 | Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | NeurIPS 2022 | checked |
| S54 | Sardana et al., "Beyond Chinchilla-Optimal", [arXiv 2401.00448](https://arxiv.org/abs/2401.00448) | ICML 2024 | checked |

Not opened (HTTP 403): IEA, "Energy and AI" (2025); LBNL, "2024 United States Data Center Energy
Usage Report". Neither is used for any number.

**Before citing any of these in the thesis:** open the full text of every "checked" entry you
rely on, and re-check 2025 to 2026 preprints for revisions. Venue labels marked here were not
re-verified against proceedings pages.
