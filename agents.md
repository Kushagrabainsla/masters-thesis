# AGENTS.md

Context file for AI agents working in this repo.

## Read first: research source of truth
- `RESEARCH_JOURNAL.md` at the repo root is the single source of truth for all thesis research: every idea, experiment, review, kill, the current state and the next steps. Read its Sections 0 and 1 before doing any research work. Search it by ID (`I-23`, `E-17`, `R-05`) instead of loading the whole file.
- After any research work, update it following Section 0.6 of the journal: new ideas get a registry row and dossier, experiments get kill criteria written before they run, reviews get an R-nn entry, and Section 1 is rewritten at the end of the session.
- Never delete history in the journal. Retractions go in its Section 8. An AI recommendation is not a decision: log who decided in its Section 10.
- Older files (`dossier/`, the handoff .docx files) are historical snapshots. When they disagree with the journal, the journal wins.
- `archive/` holds superseded files. Never delete research files: move obsolete ones to `archive/` and add a row to `archive/README.md`. Delete only true junk (`__pycache__`, `.DS_Store`).

## Focus areas
- Agentic systems: multi-agent orchestration, agent registries, priority-aware routing, tool-calling protocols, MCP (Model Context Protocol)
- GPU efficiency: distributed training fault tolerance, checkpoint-restore systems, GPU memory management, NCCL/collective communication, CUDA
- ML infrastructure: fault-tolerant execution, resource-aware scheduling, recovery for training and large-scale inference on heterogeneous compute

## Technical background relevant to this codebase
- Experience building CRIU-based checkpoint-restore systems for multi-GPU, cross-node ML training on Kubernetes (process-level state capture for fault-tolerant, preemptible training)
- Experience with multi-agent orchestration design: hot-reloadable agent registries, priority-aware inference routers with tiered model pools, append-only ledgers for per-agent failure recovery, trust layers that auto-resolve low-stakes actions while hard-gating irreversible ones
- Experience with event-driven distributed systems: queue-decoupled ingestion, jittered backoff, tuned batch concurrency

## Stack conventions
- Languages: Python, Go, C++, Rust, TypeScript
- ML infra: PyTorch, CUDA, cuda-checkpoint, NCCL, distributed training, GPU scheduling
- Systems: Kubernetes, Docker, Kafka, gRPC

## Communication style for agents
- Write all content in ASD-STE100 (Simplified Technical English). This applies to chat replies, journal entries, dossiers, READMEs, commit messages and code comments.
  - Write one topic in each sentence. Procedural sentences: 20 words or fewer. Descriptive sentences: 25 words or fewer. Paragraphs: 6 sentences or fewer.
  - Start each instruction with a verb. Write one instruction in each sentence. Put a condition first ("If X, do Y.").
  - Use active voice and simple tenses (present, past, future).
  - Use approved STE words with one meaning ("use", not "utilize"; "start", not "initiate"). Do not use phrasal verbs, idioms or slang. Keep technical names and technical verbs as they are.
  - Use articles ("the", "a"). Do not remove words to make text shorter.
- Direct, no unnecessary hedging
- Simple & technical language, no em dashes
- Prefer explaining mechanisms over restating what code already shows