# References

This folder lists the papers and other sources that the research used. Each topic folder has a `README.md` with one table. Each row gives the paper, a link, the authors, the date, what we used it for, how well we checked it and the files that cite it.

The folder has links only, not PDFs. Most arXiv licenses do not let us share copies of the PDFs.

## Topics

<!-- index:start -->
| Folder | Topic | arXiv papers | Other sources |
|---|---|---|---|
| [`01-numerics-kernel-dispatch/`](01-numerics-kernel-dispatch/README.md) | Numerics, determinism, conformance, kernel dispatch | 25 | 7 |
| [`02-hybrid-state-prefix-caching/`](02-hybrid-state-prefix-caching/README.md) | Hybrid state reuse, prefix caching, recovery | 9 | 9 |
| [`03-collectives-communication/`](03-collectives-communication/README.md) | Collectives and compute-communication overlap | 11 | 5 |
| [`04-migration-checkpoint-fault-tolerance/`](04-migration-checkpoint-fault-tolerance/README.md) | Migration, checkpoint/restore, fault tolerance | 15 | 3 |
| [`05-rl-rollout-mismatch/`](05-rl-rollout-mismatch/README.md) | RL rollout mismatch | 3 | 0 |
| [`06-elastic-training-traces/`](06-elastic-training-traces/README.md) | Elastic training, traces, multi-tenancy (all doc-only) | 0 | 0 |
| [`07-handoff-frontier-signals/`](07-handoff-frontier-signals/README.md) | Frontier signals cited by the 2026-09-22 handoff | 6 | 12 |
| [`08-agentic-harnesses/`](08-agentic-harnesses/README.md) | Agentic harnesses (added 2026-09-30) | 38 | 0 |
| [`09-gpu-efficiency/`](09-gpu-efficiency/README.md) | GPU efficiency review (added 2026-09-30) | 46 | 1 |
| [`10-agent-context-compression/`](10-agent-context-compression/README.md) | Context compression and prompt caching for agents (added 2026-10-04, R-08) | 24 | 2 |
| [`11-cross-domain-directions-r09/`](11-cross-domain-directions-r09/README.md) | Cross-domain directions (added 2026-10-06, R-09) | 28 | 12 |
| [`12-mechanism-transfer-routes-r10-r13/`](12-mechanism-transfer-routes-r10-r13/README.md) | Mechanism-transfer routes (added 2026-10-07, R-10) | 40 | 5 |
| [`13-route-search-r17/`](13-route-search-r17/README.md) | Route search 3 (added 2026-10-08, R-17) | 83 | 3 |
| [`14-orphan-scan-r18/`](14-orphan-scan-r18/README.md) | Orphan-finding scan (added 2026-10-08, R-18) | 35 | 2 |
| [`17-seed-topic-search-r20/`](17-seed-topic-search-r20/README.md) | Seed-topic gap search (added 2026-10-16, R-20) | 34 | 5 |
| [`18-current-frontier-r21/`](18-current-frontier-r21/README.md) | Current-frontier systems review (added 2026-10-17, R-21) | 12 | 8 |
| [`19-continued-frontier-r22/`](19-continued-frontier-r22/README.md) | Continued frontier systems review (added 2026-10-17, R-22) | 4 | 10 |
| [`15-assumption-removal-r19/`](15-assumption-removal-r19/README.md) | Assumption removal (R-19, 2026-10-11). Not yet in journal Section 7 | 53 | 0 |
| [`16-other-journal-citations/`](16-other-journal-citations/README.md) | Other papers cited in the journal outside Section 7 | 22 | 0 |
| | **Total** | **488** | **84** |
<!-- index:end -->

## How to read the tables

- **What we used it for:** For papers in the journal ledger, this is the "What it shows" and "Effect" text from journal Section 7. For other papers, it is the line from the dossier that cites the paper.
- **Check:** The check level from journal Section 0.4:
  - `full-text`: we read the full text or the source code.
  - `checked`: we opened the primary page (the abstract or the issue text) and confirmed the claim.
  - `search-only`: we saw only a search-result snippet.
  - `doc-only`: an earlier repo document cites it, and we did not verify it.
  - `dossier` or `journal`: the paper is cited in a dossier or in the journal outside Section 7. It has no check level.
- **Cited in:** Links to the journal section or the dossier file that cites the paper.

Journal Section 8 lists errors, retractions and sources that must not be cited. Read it before you cite a paper from this folder.

## Other files

- `references.bib`: BibTeX entries for every arXiv paper, for LaTeX.
- `_tools/build_references.py`: the script that makes this folder.
- `_tools/arxiv_metadata.json`: the titles and authors from the arXiv API, cached.

## Update

`RESEARCH_JOURNAL.md` is the source of truth. Do not edit the topic files by hand. After you add papers to the journal or a dossier, run the script again:

```bash
python3 references/_tools/build_references.py
```
