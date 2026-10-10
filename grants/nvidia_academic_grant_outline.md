# NVIDIA Academic Grant Program outline (for the advisor to submit)

Track: **AI Inference, Agents, and Systems Software.** Faculty must be the applicant (full-time faculty at an institution that awards PhDs, one proposal per quarter, one award per person per calendar year). The quarterly window opens Thursday 2026-10-01 per the Vanderbilt CTTC page. That page states no closing date, so confirm on NVIDIA's call for proposals. Format: NVIDIA's template, at most 4 pages excluding appendices. This outline is a skeleton. Fill it in once the template is in hand. Check whether SJSU requires a university technology-transfer disclosure before submission (Vanderbilt does, SJSU not checked).

## What the program says it wants (official, fetched 2026-09-24)

- Topics in scope include "scalable inference systems and distributed runtimes" and "inference efficiency via quantization and pruning, efficient decoding systems".
- Awards: up to two NVIDIA DGX Spark units and up to 30,000 H100 80GB GPU hours (maximum 8 concurrent GPUs, hours expire 6 months after award).
- Requirement: projects "must incorporate NVIDIA models, such as Nemotron, Cosmos, or Omniverse wherever possible. All others must make extensive use of NVIDIA software and AI distributions, such as NeMo microservices, NVIDIA Dynamo, FLARE, or CUDA-X GPU-accelerated libraries."

## Outline

1. **Title and one-line claim.** Certifying output equivalence of fast LLM inference kernels on NVIDIA Hopper and Blackwell GPUs.
2. **Problem.** Same as the NAIRR proposal, sections 2 and 3: kernels are treated as interchangeable, recent evidence says they are not, and the closest study disclaims other GPU generations, MoE, NVFP4 and multi-GPU.
3. **Fit with the program.** Inference efficiency via quantization (block-scaled FP8, NVFP4) and inference systems, with a correctness contract that makes faster kernels safe to adopt.
4. **NVIDIA technology used (required).** CUDA-X libraries (cuBLAS, CUTLASS), TensorRT-LLM and its quantization tooling as a backend, NVIDIA Dynamo as a serving baseline where applicable, Nsight for profiling, Nemotron-family open models in the model grid. Confirm each against the template's wording before claiming it.
5. **Research questions and predeclared decision rules.** RQ1 and RQ2 from the NAIRR proposal, section 4.
6. **Method, baselines, statistics.** Section 5 of the NAIRR proposal.
7. **Preliminary evidence.** The laptop pilot, stated honestly: harness core validated, one estimator falsified, no claims about GPU kernels.
8. **Resource request, right-sized.** Ask for hours the plan justifies. The NAIRR estimate is about 350 GPU-hours for the core study. A request of about 1,000 H100 hours covers the Hopper cells, the second model family, and held-out reruns. Ask for the DGX Spark units only if there is a concrete use, for example local Blackwell development. Do not ask for 30,000 hours.
9. **Timeline and deliverables.** Three months to the first decision gate, then the write-up. Open-source harness, result bundles, upstream issue reports, thesis.
10. **Team.** Faculty applicant, student, prior ACCESS and NAIRR experience in the lab.

## Open items

- Get the actual template and CFP, then map this outline to it.
- Confirm the window dates and whether a deadline exists for the October cycle.
- Decide the resource split with the NAIRR and ACCESS requests so the three do not overlap on the same cells.
