# E-31 independent pre-run review

Act as a hostile OSDI, SOSP, or MLSys systems reviewer.

Review the frozen E-31 plan and implementation for I-122.
Do not execute any code.
Do not edit any file.
Use only read, grep, and glob operations.

Read these files in full:

- `experiments/versioned_session_commit/README.md`
- `experiments/versioned_session_commit/model.py`
- `experiments/versioned_session_commit/run_e31.py`
- `experiments/versioned_session_commit/decide_e31.py`
- `experiments/versioned_session_commit/test_model.py`
- `experiments/versioned_session_commit/Dockerfile`
- `experiments/versioned_session_commit/run_docker.sh`

Read the E-31 and I-122 entries in `RESEARCH_JOURNAL.md`.

Attack these areas:

1. Does the model encode the expected result into its definitions?
2. Does each policy get a fair implementation?
3. Are the invariants sufficient for safety, completeness, and output ordering?
4. Can a policy pass by dropping valid work or events without recovery?
5. Do the source scenarios faithfully test the stated issue classes?
6. Does the bounded search cover meaningful distinct states and transitions?
7. Do the scale sweep and cost measures support the frozen decision rules?
8. Does `decide_e31.py` implement every rule correctly and in the declared order?
9. Can the conservative baselines fail outside the source scenarios without invalidating the run?
10. Are the Docker controls and manifest sufficient for reproducibility?
11. Is any result predetermined, circular, or too abstract to guide an implementation test?
12. Can a simpler experiment kill I-122 more directly?

Classify each finding as BLOCKER, MAJOR, or MINOR.
Give file and line references.
Separate facts from interpretations.

End with one verdict:

- `SAFE_TO_RUN`
- `AMEND_BEFORE_RUN`
- `KILL_EXPERIMENT_DESIGN`

If amendments are required, give the smallest exact amendment set.
Do not assess the experiment from expected output.
No experiment results exist.

Use simple technical English.

