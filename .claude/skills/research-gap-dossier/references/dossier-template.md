# Research Dossier Template

Use this as the default structure for the strongest surviving idea. Keep it concise enough to read in one sitting, but complete enough that another researcher can continue the work.

# [Idea title]

## 1. One-sentence idea
Explain the research direction in plain language.

## 2. Why this problem exists
Explain the workload/architecture/system change that creates the problem.

## 3. Research question
Give one precise, falsifiable question.

## 4. Why this is a valid question
Explain why the answer is not already obvious and what scientific/system uncertainty remains.

## 5. What existing research already solves
Use a compact table:

| Work | What it solves | What remains outside its scope |
|---|---|---|

Include the strongest papers and shipping systems, not a long bibliography dump.

## 6. Exact surviving gap
Write the smallest unanswered technical statement. Also state what must **not** be claimed as novel.

## 7. Proposed mechanism or investigation
Describe the likely design only after the gap is established. If mechanism is not yet justified, say that the next step is measurement rather than system construction.

## 8. Strongest adversarial objections
List 3-5 reasons the thesis could fail, including the simplest competing solution and the most threatening adjacent work.

## 9. Validation completed so far
For each experiment:
- question tested;
- setup;
- result;
- interpretation;
- limitation;
- how it changed the research question.

## 10. Next decisive POC / experiment
Specify:
- exact hypothesis;
- hardware/software;
- workload/model/traces;
- implementation steps;
- baselines;
- metrics;
- correctness checks;
- expected output files;
- GO / PIVOT / KILL criteria.

## 11. Final evaluation plan
Include realistic scale, multiple workloads/models, tail latency/SLOs, throughput/goodput, resource cost, correctness, and sensitivity analysis as appropriate.

## 12. Resource requirements
Separate local screening from final GPU/cluster needs.

## 13. Scope and timeline
Give a short phased plan with early kill gates before expensive engineering.

## 14. Publication and open-source story
State the generalizable research contribution and the artifact that would be released.

## 15. Current verdict
Use one of: GO / PIVOT / MEASURE-FIRST / KILL. Give the main reason in one paragraph.

## 16. Sources
Prefer primary sources. Include title, venue/source, date, and link.
