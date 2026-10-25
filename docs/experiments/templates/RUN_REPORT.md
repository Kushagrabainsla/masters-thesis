# E-nn run report: <session> (<date>)

Add this to the experiment README after stage S8, and copy the short form into the journal E-nn
"Result" line.

## 1. Decision
The preregistered outcome (GO, PIVOT, KILL, INCONCLUSIVE or INVALID) in one line, and the rule that
produced it.

## 2. What ran
| Item | Value |
|---|---|
| Pod id, GPU x count, data center, cloud | |
| Price per hour, Pod time, billed cost | |
| Image tag and digest, driver, CUDA | |
| Design commit, amendments in force | |
| Runs planned, ok, failed, skipped by the cap | |

## 3. Validity
Each validity check with its value and pass or fail. Telemetry: suspect runs. Microbenchmark at start
and end. One-GPU check.

## 4. Primary result
The numbers behind the decision, with intervals. Use the `compare_runs.py` verdicts for speed and
Wilson intervals for counts.

## 5. Exploratory results
Labelled EXPLORATORY. They do not change the decision.

## 6. Incidents
What happened, the evidence, the cost, and what changes next time.

## 7. Post-run review
Reviewer, verdict, findings, and what changed because of them (`dossier/REVIEW_<E-nn>_POSTRUN_<date>.md`).

## 8. What it changes
Effect on the idea's status, the next experiment, and any journal decision for the user.

## 9. Artifacts
Results path, ledger run ids, hardware log row, review file.
