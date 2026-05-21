# Research Domain Introduction and Literature Review Log

**Student:** Kushagra Bainsla
**Program:** MSCS, San Jose State University
**Lab:** MICoSys Lab (Sengupta Lab)
**Status:** Active exploration. Not yet committed to a final project.

---

## Domain (Brief)

**Machine Learning for Reliable and Efficient AI Infrastructure**

The domain applies machine learning to make large scale AI training and serving systems both more reliable and more efficient. The two goals are coupled at scale. A failure prediction system that adds too much overhead is useless. A scheduler that ignores reliability wastes compute on jobs that will crash anyway. Research at this intersection produces work that matters to every organization running AI at scale.

The domain is a natural extension of the MICoSys Lab's existing strengths in time series reliability, adversarial robustness, and graph based machine learning, applied to a new and rapidly growing application area.

---

## How This Document Works

This is a working log. Each entry below records a step in narrowing down the research direction. Entries are dated and additive. Nothing is committed yet. The goal is to converge on one specific project after enough lit review and discussion with Prof. Sengupta.

---

## Log Entries

### Entry 1: Initial Domain Selection
**Date:** May 19, 2026

Selected the broad domain of machine learning for AI systems reliability. The reasoning:

- Lab strengths in time series reliability, adversarial robustness, and graph machine learning all transfer naturally to this domain
- The application area (AI infrastructure) is large, growing, and underserved by academic research
- The domain matches my career trajectory through LinkedIn AI Infrastructure
- Both Prof. Sengupta's lab notes and recent literature suggest systems reliability is one of three active lab directions

This is the starting point. The domain is broad and needs to be narrowed through further reading and discussion.

---

### Entry 2: Reframing as Reliability and Efficiency
**Date:** May 20, 2026

After reflection, the framing should not be reliability alone. At scale, reliability and efficiency are coupled. A checkpoint that comes too late wastes compute. A scheduler that ignores reliability assigns work to failing nodes. Research at the intersection is more useful and more honest than research that treats either one alone.

The lab is well placed to do this work because the same techniques (time series, anomaly detection, adversarial robustness, graph models) serve both goals.

---

### Entry 3: Market and Industry Numbers
**Date:** May 20, 2026

Gathered hard data to verify the field is worth entering.

**Failure rates at scale:**
- Meta Llama 3: 16,384 H100 GPUs, 54 days, 419 unexpected failures (1 per 3 hours)
- OPT-175B: 992 A100 GPUs, 90 days, 112 failures (about 1.2 per day)
- Microsoft GPU clusters: MTBF ranges from minutes to days

**Training run costs:**
- GPT-4: 78 to 100+ million USD
- Gemini Ultra 1.0: 192 million USD
- GPT-5 (reported): 500 million USD per run
- Training cost growth: 287,000 times since 2017

**Market size:**
- IDC: AI infrastructure spending 334 billion USD in 2025, projected 900+ billion by 2029
- Gartner: Total AI spending exceeds 2 trillion USD in 2026
- Multiple market firms agree on 20 to 27 percent CAGR
- MLOps and observability sub-segment: 28.7 percent annual growth

**Academic activity:**
- Dedicated venues: MLSys, SOSP, NSDI, SoCC, OSDI, USENIX ATC and FAST
- MLPerf Storage v2.0 (Aug 2025): First industry standard benchmark for AI training checkpointing
- Several flagship papers in the last 12 months

**Conclusion:** The field is real, large, growing, and academically open. Worth entering.

---

### Entry 4: Initial Literature Review Set
**Date:** May 20, 2026

Compiled an initial reading list of 12 to 13 papers organized by sub-area.

**GPU Failure Prediction:**
- Liu et al. (SYSTOR 2023). Predicting GPU Failures With High Precision Under Deep Learning Workloads.

**Silent Data Corruption:**
- LLM-PRISM (arXiv 2026). Characterizing SDC from Permanent GPU Faults.
- Exploring SDC as a Reliability Challenge in LLM Training (arXiv 2026).

**Localizing Failures in Training:**
- Localizing Irregularities in LLM Training with Mega-scale Telemetry (NSDI 2025).

**Decentralized Training Integrity:**
- SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training (arXiv 2026).

**LLM Inference Serving:**
- FastServe (arXiv 2023).
- PARS: Prompt-Aware Scheduling for Low-Latency LLM Serving (arXiv 2026).

**RL for Cluster Scheduling:**
- RLTune (SoCC 2025).
- Large-scale ML Cluster Scheduling via Multi-agent Graph RL (arXiv).

**Adversarial Robustness of Monitoring:**
- Adversarial Analysis of ML-Based Anomaly Detection (2022).
- Robustness Testing of Data and Knowledge Driven Anomaly Detection (arXiv 2022).

**Network Scaling Effects:**
- When Scaling Fails: Network and Fabric Effects on Distributed GPU Training Performance (arXiv 2026).

**Production Reliability Case Study:**
- Ranganathan et al. Enhancing Reliability in AI Inference Services (arXiv 2025).

The set covers enough breadth to identify gaps. Next step: read closely and find narrower problems.

---

### Entry 5: Identified Open Research Gaps (Broad Level)
**Date:** May 20, 2026

From the initial reading, several gaps stand out:

1. Silent data corruption detection is still primitive. Multivariate time series ML over loss, gradients, and hardware telemetry has not been systematically applied.
2. Cascading failure prediction in GPU clusters is underexplored. Graph based models are a natural fit.
3. Inference scheduling assumes hardware works correctly. Failure aware scheduling is missing.
4. Monitoring systems themselves can be attacked. Adversarial robustness of AI infrastructure monitoring is essentially unstudied.
5. Checkpointing policies are mostly static. Predictive ML driven policies are underexplored.

Each of these could grow into a master's project.

---

### Entry 6: LinkedIn Internship Project Discovered as a Possible Direction
**Date:** May 20, 2026 (later in the day)

Met with the LinkedIn manager and learned my summer project will be on container checkpointing and restore for distributed training jobs at the LinkedIn AI Infrastructure team. The work uses CRIU (open source container checkpointing) and targets three use cases:

- Preemption handling, to avoid losing hours of work
- Hardware failure recovery (current failure rate around 5 percent of nodes)
- Node maintenance without job interruption

The proof of concept exists. The internship work involves productionizing it at scale across thousands of GPUs. There is also a potential academic paper opportunity with the original author of CRIU.

**Why this is interesting for research:**

The LinkedIn project handles the systems engineering (how to actually pause and resume containerized training jobs). The research project could focus on the ML policy layer on top, asking questions like:

- When should we checkpoint? Predict imminent failures so checkpoints happen just in time.
- How often should we checkpoint? Adapt to current cluster failure rates.
- Where should we recover? Pick the healthiest available node.
- What should we checkpoint? Predict which state matters most.

**Important caveats:**

This is one possible direction, not a commitment. There are reasons to consider it and reasons to be careful:

- It aligns very well with the lab's time series and reliability strengths.
- It provides real industry data and infrastructure.
- It has a potential publication path through the CRIU collaboration.
- It also creates dependency on a specific industry collaboration that could change.
- It is narrower than the original domain, which has tradeoffs.

This direction will be one of several options to discuss with Prof. Sengupta.

---

### Entry 7: Additional Literature on Checkpointing
**Date:** May 20, 2026

Since the LinkedIn project surfaced checkpointing as a relevant area, added more papers in this sub-area for completeness. This does not commit the research to checkpointing but builds out the option.

- Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints (SOSP 2023). State of the art in fast recovery, proves optimal static placement, reduces lost time by 92 percent.
- CheckFreq: Frequent, Fine-Grained DNN Checkpointing (USENIX FAST 2021). Foundational fine-grained checkpointing.
- Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures (EuroSys 2024). Predictive checkpointing using rule based prediction. Closest paper to a possible ML extension.
- LowDiff: Optimizing Frequent Checkpointing via Low-Cost Differential (arXiv 2025). Differential checkpointing reduces cost by up to 89.2 percent.
- Convergence-Aware Optimal Checkpointing for Exploratory Deep Learning Training Jobs (2024). Convergence dynamics informing checkpoint decisions.

**Observation:** Most state of the art checkpointing systems use static or rule based policies. ML driven adaptive policies are an open extension across all of these.

---

## Possible Research Directions (Current Working Set)

These are options under consideration. None are committed. The choice will be made after more reading and discussion with Prof. Sengupta.

### Option 1: ML Driven Detection of Silent Data Corruption in LLM Training
Multivariate time series anomaly detection on loss, gradients, attention statistics, and hardware metrics to catch silent corruption that simple NaN checks miss. Strong fit with lab time series work. Independent of any specific industry collaboration.

### Option 2: Graph Neural Networks for Cascading Failure Prediction in GPU Clusters
Model GPU clusters as graphs and predict how failures propagate. Strong fit with lab's recent graph machine learning work. Useful for both training and serving infrastructure.

### Option 3: Adversarial Robustness of AI Infrastructure Monitoring
Study attacks on ML based failure prediction and anomaly detection systems used for AI infrastructure. Develop defenses using lab moving target defense techniques. Very novel. Strong lab fit.

### Option 4: ML Driven Policies for Intelligent Checkpointing (LinkedIn aligned)
Add an ML policy layer on top of static checkpointing systems. Predict failure timing, adapt checkpoint frequency, choose recovery nodes intelligently. Aligned with LinkedIn internship project. Potential publication path through CRIU collaboration.

### Option 5: Reinforcement Learning for Failure Aware Inference Scheduling
RL based inference scheduler that handles node failures gracefully. Aligns with my RL coursework next semester. Less aligned with current lab work but covers an open area.

---

## Open Questions to Discuss With Prof. Sengupta

1. Which of the five options aligns best with where the lab wants to grow?
2. How much should industry alignment (Option 4) weigh against research independence (Options 1, 2, 3)?
3. Is there an existing lab project I could join rather than starting fresh?
4. What is the realistic timeline given LinkedIn internship in summer and full coursework in fall?
5. Are there specific datasets the lab already has access to that would shape the choice?

---

## Next Steps

- Continue lit review during summer, focused on the top one or two options after discussion with Prof. Sengupta.
- Reproduce key results from the anchor papers of the chosen direction on lab compute (Vast AI, Lambda Tensorbook).
- Have weekly check-ins with Prof. Sengupta to refine the direction.
- Plan to commit to one specific project by the end of June.

---

## References

**Industry Reports and Data Sources**

- Meta. (2024). Llama 3 Training Run Report.
- Stanford AI Index Report 2025.
- IDC Worldwide Quarterly AI Infrastructure Tracker.
- Gartner Worldwide AI Spending Forecast 2026.
- Multiple market research firms (Fortune Business Insights, Mordor Intelligence, Coherent Market Insights, MarketsAndMarkets, Technavio).
- Epoch AI Compute Growth Analysis.
- MLCommons MLPerf Storage v2.0 (2025).

**Academic Papers (Working Reading List)**

*Checkpointing in Distributed Training*
1. Wang, Z., Jia, Z., Zheng, S., et al. (2023). Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints. *SOSP 2023*.
2. Mohan, J., Phanishayee, A., Chidambaram, V. (2021). CheckFreq: Frequent, Fine-Grained DNN Checkpointing. *USENIX FAST 2021*.
3. (2024). Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures. *EuroSys 2024*.
4. Yao, C., Hu, Y., Liu, F., et al. (2025). Optimizing Frequent Checkpointing via Low-Cost Differential for Distributed Training Systems. arXiv:2509.04084.
5. (2024). Convergence-Aware Optimal Checkpointing for Exploratory Deep Learning Training Jobs. *Future Generation Computer Systems*.

*Failure Prediction and Silent Data Corruption*
6. Liu, H., Li, Z., Tan, C., et al. (2023). Predicting GPU Failures With High Precision Under Deep Learning Workloads. *SYSTOR 2023*.
7. (2026). LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training. arXiv preprint.
8. (2026). Exploring Silent Data Corruption as a Reliability Challenge in LLM Training. arXiv preprint.

*Localization and Pipeline Integrity*
9. Yao et al. (2025). Localizing Irregularities in LLM Training with Mega-scale Telemetry. *NSDI 2025*.
10. (2026). SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training. arXiv preprint.

*LLM Inference Serving*
11. Wu et al. (2023). Fast Distributed Inference Serving for Large Language Models. arXiv:2305.05920.
12. (2026). PARS: Prompt-Aware Scheduling for Low-Latency LLM Serving. arXiv:2510.03243.

*Reinforcement Learning for Cluster Scheduling*
13. Dongare, S., Khan, R., Albahar, H., et al. (2025). Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads. *SoCC 2025*.
14. Zhao, X. and Wu, C. Large-scale Machine Learning Cluster Scheduling via Multi-agent Graph Reinforcement Learning. arXiv:2112.13354.

*Adversarial Robustness of Monitoring*
15. (2022). Adversarial Analysis of ML-Based Anomaly Detection in Multi-Layer Network Automation.
16. (2022). Robustness Testing of Data and Knowledge Driven Anomaly Detection in Cyber-Physical Systems. arXiv:2204.09183.

*Other*
17. (2026). When Scaling Fails: Network and Fabric Effects on Distributed GPU Training Performance. arXiv:2603.04424.
18. Ranganathan, B., Zhang, M., Wu, K. (2025). Enhancing Reliability in AI Inference Services: An Empirical Study on Real Production Incidents. arXiv:2511.07424.
19. MLCommons (2025). MLPerf Storage v2.0 Checkpointing Workload Specification.