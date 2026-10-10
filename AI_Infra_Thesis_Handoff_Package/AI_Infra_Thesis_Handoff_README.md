# AI Infrastructure Thesis Research Handoff — Canonical Context

Date: 2026-09-16

This file is a plain-text companion to the two DOCX files. The canonical detailed narrative is in `AI_Infra_Thesis_Master_Handoff_Dossier.docx`; exact experiment details are in `AI_Infra_Thesis_Experiment_Evidence_Ledger.docx`.

## Current state in one paragraph
The broad typed/heterogeneous-state-runtime thesis has been killed because current vLLM already supports typed cache specs and multi-group hybrid coordination. A generic REQUIRED/RECONSTRUCTABLE/EPHEMERAL contract passed a source-pinned vLLM planning POC, but this only proved software expressibility. A general min-cost state planner was deprioritized because the best simple per-row policy matched it in 86.7% of 324 normalized-cost cases. Real Qwen3.5 Mac tests confirmed heterogeneous cache growth and showed suffix replay cost roughly tracks suffix length, but the latest divergent-state frozen-KV POC was semantically incorrect; its negative timings are invalid. A one-case activation-output oracle was correct and only 9.5% faster, but the setup is too small/biased to kill the idea. Current recommendation: continue a corrected long-context Mac falsification of the narrow hybrid-state recovery question while in parallel performing a fresh adversarial dossier on compute-state-aware collective engine arbitration. Do not pick the thesis yet.

## Hard warnings
- Do not use V1 synthetic screens as empirical evidence.
- Do not use frozen-KV v1.1 negative timings; semantic recovery was incorrect.
- Do not call the normalized planner a hardware result.
- Do not claim hybrid-state novelty without reading HYPIC, Sparse Prefix Caching, ReplaySSM, current vLLM issues/RFCs and Kimi K3/hybrid work.
- Do not promote collective-engine arbitration until it gets the same adversarial treatment.

## Current parallel work
1. Hybrid state: correct replay semantics, long-context oracle sweep, FLOP model, novelty re-check.
2. Collective engine arbitration: fresh two-pass dossier around SM-driven vs copy-engine/zero-SM communication selected using live compute resource pressure.

## Pinned vLLM commit used in POCs
`e0705731be04aac90838f318fcdae15c56a53080`

See the DOCX files for all raw numbers, source URLs, artifacts, and decision history.
