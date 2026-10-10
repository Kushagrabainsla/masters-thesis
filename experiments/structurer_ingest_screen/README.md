# structurer_ingest_screen (I-41, E-24, E-25)

Tests for the "ingest structurer" idea: a small LLM rewrites prompts and tool outputs before the
main agent model sees them. Predeclared criteria and results are in `RESEARCH_JOURNAL.md`,
Section 6.1 (E-24, E-25). The write-up is `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md`.

## Privacy

Both scripts read the user's own Claude Code transcripts in `~/.claude/projects` (read-only).
Only aggregate numbers are written into `results/`. The E-25 samples and per-sample model outputs
contain the user's code, so they go to a scratch directory outside the repo.

## Run

E-24 uses the system Python (standard library only):

```bash
python3 e24_trace_study.py --out results/e24_results.json --samples-out /path/to/scratch/e25_samples.json
```

E-25 needs torch and transformers (the repo's shared `.venv`, see journal Section 9.4) and a cached
`Qwen/Qwen3-0.6B`:

```bash
HF_HUB_OFFLINE=1 ~/Downloads/mac_hybrid_selective_recovery_poc/.venv/bin/python e25_structurer_screen.py --samples /path/to/scratch/e25_samples.json --out results/e25_summary.json --detail-out /path/to/scratch/e25_detail.json
```

The script excludes the session that wrote it (`EXCLUDE_SESSION`). Transcripts are live files, so
totals drift slightly between runs.

E-26 needs pyarrow and the Exgentic SWE-bench shards in `~/Downloads/agent-llm-traces/data/` (see
journal Section 9.4). It also re-reads the local transcripts for a cross-check:

```bash
~/Downloads/agent-llm-traces/.venv/bin/python e26_public_trace_check.py --out results/e26_results.json
```
