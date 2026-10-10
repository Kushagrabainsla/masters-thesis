# NSF ACCESS Explore request (draft)

Status: DRAFT for Kushagra Bainsla to review, edit and submit as PI. ACCESS asks for the fields below plus a CV (PDF) and a signed advisor letter on institutional letterhead (PDF). Submit through the ACCESS allocations portal (allocations.access-ci.org).

## Fields

| Field | Entry |
|---|---|
| Project title | Certifying output equivalence of fast LLM inference kernels on Hopper GPUs |
| Allocation type | Explore ACCESS (graduate student PI, advisor letter attached) |
| Resource | NCSA Delta GPU. Preferred node type: 8-way NVIDIA H200 (141 GB, NVLink). Fallback: 4-way A100 nodes |
| Amount | Fill in after reading the ACCESS credit exchange calculator (allocations.access-ci.org/exchange_calculator) for "Delta GPU". Target: about 300 H200 GPU-hours. Delta's user guide charges 1 SU per GPU-hour on H200 nodes. The credit-to-SU rate is not confirmed |
| Field of science | Computer and Information Science and Engineering, Computer Systems |
| Keywords | LLM inference, GPU kernels, numerical reproducibility, FP8 quantization, vLLM, benchmarking |
| Other resources | A NAIRR Pilot Start-Up request for Blackwell GPUs (a different architecture) is being submitted separately. There is no other funding |

## Project summary (paste into the abstract field)

LLM inference servers ship several implementations of the same operation and choose the fastest. Recent work shows that swapping one backend for another can change the tokens a model generates, even when both implement the same interface. For example, replacing CUTLASS INT8 kernels with Triton kernels inside vLLM gave zero sequence agreement in every tested case. That study covered one consumer GPU, one model family, and no mixture-of-experts models or multi-GPU runs, and it states that its results do not extend to other GPU generations. This project measures, on Hopper (H200) GPUs, how often substituting one backend for another for block-scaled FP8 and BF16 operators and fused mixture-of-experts kernels violates an explicit output contract, and whether the fastest contract-satisfying backend varies across batch size and prefill versus decode enough to matter. We will build an open-source harness that records bitwise, token-level and logit-level equivalence with run manifests, evaluate open-weight dense and mixture-of-experts models under vLLM and FlashInfer, and report results with statistical upper bounds. Explore-scale resources are appropriate because this is a benchmarking and code-development effort: we need short single-GPU jobs to validate the harness on the target hardware, run a first grid, and size a larger request. The harness has already been validated on a laptop with fault-injection tests, so GPU time goes to the science, not to setup.

## Resource usage plan (if a form field asks)

Single-GPU jobs, most under 2 hours, and tensor-parallel jobs on one node for the larger models. Software: Python, PyTorch, vLLM, FlashInfer, CUTLASS, Triton, Nsight tools, containers. No multi-node runs and no licensed software. Data: public models and public or synthetic text. Estimated 300 GPU-hours, with results and code released openly.

## Checklist

- [ ] Advisor letter signed on letterhead, PDF (draft in `advisor_letter_draft.md`)
- [ ] CV or resume, PDF
- [ ] Read the exchange calculator and fill in the amount
- [ ] Confirm the institutional email on the ACCESS account
