# E-nn: <title> (preregistration)

Copy this into `experiments/<name>/README.md`. Fill every field before stage S2. Commit it: that
commit is the frozen design. Later changes go in "Amendments" below, never in the frozen text.

Status: PREDECLARED <date>. NOT RUN.
Idea: I-nn. Journal entry: Section 6.1 E-nn.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
One falsifiable sentence. What result means "no"?

## 2. Rubber duck
What the idea claims, what is new since the last check, what would kill it early, and why this
order of work.

## 3. Hypotheses
- H1 (primary):
- H0 (what we expect if the idea is wrong):

## 4. Setup (frozen)
- Hardware: GPU type and count, cloud type, CUDA floor. Why this GPU.
- Software: image tag (and digest after the run), library versions, model ids and revisions.
- Data: source, size, selection seed, file hash.
- Configs: the factor that changes, the values, and everything that stays the same.

## 5. Metrics
- Primary metric, unit, direction:
- Minimum effect that matters (MDE):
- Secondary metrics (reported only):
- Repeats per configuration:

## 6. Validity checks (any failure makes the run INVALID)
- V1:
- V2:

## 7. Decision rule (frozen)
- GO (or CONTINUE):
- PIVOT:
- KILL:
- Otherwise INCONCLUSIVE, with the failed condition named.

## 8. Controls and baselines
- Baseline (what we compare against):
- A/A run of the baseline:
- Positive control (must show an effect):
- Negative control (must show no effect):

## 9. Budget
- Pod time limit, price per hour from a live read, cost cap:
- Order of work with a time budget per step (the most decisive data first):
- What the time cap cuts first:

## 10. Threats to validity
List each one and how the design handles it, or say that it stays open.

## 11. Files
Every script, what it does, and its tests.

## 12. Verification record (stage S3 and S5)
Unit tests, Mac dry run, stub run, rehearsal: date, result, defects found and fixed.

## Amendments
A1 (<date>, after R-nn, before any run): what changed and why. Mark which rules are unchanged.
