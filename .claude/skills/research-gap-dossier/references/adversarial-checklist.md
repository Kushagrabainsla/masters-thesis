# Adversarial Review Checklist

## A. Prior-art attack

For each candidate, fill in:

| Question | Answer |
|---|---|
| Closest paper | |
| Closest shipping/open-source system | |
| Closest adjacent-domain mechanism | |
| What they already solve | |
| What they explicitly do not solve | |
| Why our question is technically different | |

If the difference is only wording, kill the candidate.

## B. Simplest-solution attack

Try these before designing a new mechanism:
- stop and copy;
- recompute/replay;
- batching/multiplexing;
- caching/prefetching;
- reactive scheduling;
- static provisioning;
- overprovisioning;
- existing checkpointing;
- existing load balancing;
- current runtime hints.

If a simple baseline captures most of the theoretical gain, narrow or kill the idea.

## C. Scale attack

Ask how the claim changes at:
- one request/user;
- tens of concurrent requests;
- hundreds/thousands of requests;
- multiple replicas;
- multi-GPU/model-parallel serving;
- different network bandwidths;
- different context lengths/model sizes.

Look for statistical multiplexing, batching, cache sharing, queue smoothing, and contention that reverse local conclusions.

## D. Oracle attack

When future information or scheduling is central, compute a perfect-information upper bound before building the predictor or scheduler.

Interpretation:
- Small oracle gain -> kill the mechanism.
- Large oracle gain but weak realistic predictor -> research may be prediction/information exposure rather than scheduling.
- Large oracle gain and strong predictor -> mechanism is worth prototyping.

## E. Reviewer rejection test

Write the likely rejection in one sentence, for example:
- "This is already Llumnix with a different state representation."
- "This is Continuum plus ConServe."
- "The claimed slack disappears under continuous batching."
- "A 10 ms stop-and-copy solves the problem."

The revised contribution must directly answer that rejection with data or a concrete systems distinction.

## F. Evidence discipline

Separate:
- **Fact**: directly supported by a source or measurement.
- **Interpretation**: what the fact suggests.
- **Hypothesis**: what the proposed work predicts.
- **Contribution**: what must still be built or demonstrated.

Do not turn interpretation into fact.
