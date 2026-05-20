# Research Domain: Machine Learning for Reliable and Efficient AI Infrastructure

**Student:** Kushagra Bainsla
**Program:** MSCS, San Jose State University
**Lab:** MICoSys Lab (Sengupta Lab)
**Date:** May 2026

---

## 1. Domain

**Machine Learning for Reliable and Efficient AI Infrastructure**

This domain uses machine learning to make large scale AI systems both more reliable and more efficient. The two goals are tightly linked. A failure prediction system that adds too much overhead is useless. A scheduler that ignores reliability wastes compute on jobs that will crash anyway. Research at this intersection produces work that matters to every organization running AI at scale.

---

## 2. Abstract

Modern AI infrastructure runs at a scale where reliability and efficiency cannot be treated as separate problems. Training a frontier model uses tens of thousands of accelerators over weeks. Inference systems serve millions of users with strict latency targets. At this scale, hardware fails, jobs crash silently, and inefficiencies compound into millions of dollars of wasted compute.

This research applies machine learning techniques such as time series anomaly detection, multivariate forecasting, graph neural networks, adversarial robustness, and reinforcement learning to AI infrastructure. The work targets problems where reliability and efficiency reinforce each other: predicting failures to avoid wasted compute, scheduling jobs under uncertainty, detecting silent data corruption before it ruins training runs, and defending monitoring systems against adversarial inputs.

The methods are an extension of the MICoSys Lab's existing strengths in time series reliability, adversarial robustness, and graph based learning, now applied to a new and rapidly growing application domain.

---

## 3. Why This Domain

### Reliability and efficiency are coupled at scale

Every reliability problem is also an efficiency problem.

- A GPU failure during distributed training wastes the compute already spent on that run.
- A silent data corruption event that goes undetected for hours wastes every training step after the corruption.
- A reactive scheduler that does not anticipate failures keeps assigning work to failing nodes.
- An overly cautious monitoring system that triggers false alarms wastes engineering time.

Research that treats these problems together produces solutions that industry can actually use.

### The field is open and active

Key papers on this topic were published in 2025 and 2026 at top venues like NSDI and SoCC. The first dedicated paper on GPU failure prediction in production AI clusters appeared only in 2023. Silent data corruption in LLM training was first systematically characterized in 2026. Adversarial robustness of AI infrastructure monitoring systems is essentially unstudied. A master's project can make a real contribution here.

### Strong fit with the lab

The MICoSys Lab has published on time series forecasting for system health, adversarial robustness in time series, moving target defense, and graph based machine learning. Every one of these techniques transfers directly to AI infrastructure. The lab does not need new methods. It needs a new application domain, which is exactly what this work provides.

### Personal alignment

I will join LinkedIn's AI Infrastructure team in summer 2026, which provides direct exposure to the problems that drive this research. The fall semester courses in Distributed Systems and Reinforcement Learning support this domain technically. The combination gives me an unusually strong foundation for serious work over the next year.

---

## 4. How Reliability and Efficiency Connect in This Work

| Problem | Reliability Angle | Efficiency Angle |
|---|---|---|
| Predicting GPU failures | Avoid silent corruption and crashes | Save compute that would otherwise be wasted |
| Smart inference scheduling | Route around degraded or failing nodes | Reduce tail latency and improve throughput |
| Silent data corruption detection | Catch faults before they propagate | Skip expensive checkpoint rollback |
| Adversarial robust monitoring | Prevent attackers from hiding failures | Reduce false alarms and engineer overhead |
| RL based resource allocation | Anticipate failures probabilistically | Maximize cluster utilization |

Each cell shows that the same technical work serves both goals at once.

---

## 5. Connection to Existing Lab Work

| Existing Lab Strength | Application in AI Infrastructure |
|---|---|
| Time series forecasting for turbofan engines and batteries | Predicting failures in GPU clusters and training runs |
| Adversarial robustness in time series | Studying attacks on AI monitoring and defending against them |
| Moving target defense | Defending ML based monitors used in AI infrastructure |
| Graph aware machine learning (NeurIPS 2025 workshop) | Modeling GPU cluster topology for cascading failure prediction |
| Hardware reliability research | Extending datacenter reliability into AI training and serving |

The mathematics is the same. Only the application changes.

---

## 6. Literature Review

The following papers establish the state of the art and define open problems in this domain.

### 6.1 GPU Failure Prediction

**Paper 1: Predicting GPU Failures With High Precision Under Deep Learning Workloads**
*Liu et al. SYSTOR 2023.*

First paper to study GPU failure prediction at production scale. Trained on a four month dataset from ByteDance with 350 million entries. Found that classic models (GBDT, MLP, LSTM, 1D-CNN) are inaccurate and unstable. Their ensemble and sliding training approach improved precision from 46.3 percent to 85.4 percent.

**Relevance:** Establishes the baseline and confirms the problem is unsolved. Modern methods (transformers, graph networks, robust models) have not been systematically tried.

### 6.2 Silent Data Corruption in LLM Training

**Paper 2: LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training**
*arXiv 2604.10390, 2026.*

Characterizes how silent data corruption affects LLM training across FP8, FP16, and BF16 formats. Some corruptions crash training. Others silently degrade model quality with perplexity deviations up to 600 percent above baseline. Simple NaN checks catch some but not all faults.

**Relevance:** Maps the failure modes. Motivates the need for better detection methods.

**Paper 3: Exploring Silent Data Corruption as a Reliability Challenge in LLM Training**
*arXiv 2604.00726, 2026.*

Uses targeted fault injection at the GPU matrix multiply level to characterize how silent data corruption manifests during LLM training. Shows local faults produce NaN propagation, loss spikes, gradient norm anomalies, and parameter divergence. Proposes a lightweight detection method validated on LLaMA models up to 1.3B parameters.

**Relevance:** Direct template for a project. The detection method is described as lightweight, leaving room for better approaches.

### 6.3 Localizing Failures in Large Scale Training

**Paper 4: Localizing Irregularities in LLM Training with Mega-scale Telemetry**
*NSDI 2025.*

Uses random forest classifiers on communication tracing logs to detect abnormal communication operators across GPUs during pipeline parallel training.

**Relevance:** Shows top systems venues consider this important. Random forests work as baselines, leaving room for more sophisticated methods.

### 6.4 Decentralized Training Integrity

**Paper 5: SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training**
*arXiv 2603.03592, 2026.*

Uses momentum based monitoring with exponential moving averages to detect corrupted inter stage communication. Provides convergence guarantees and demonstrates training of LLMs up to 4 billion parameters across 176 workers in untrusted environments.

**Relevance:** Defense oriented framing. Can be extended to adversarial settings.

### 6.5 LLM Inference Serving

**Paper 6: Fast Distributed Inference Serving for Large Language Models (FastServe)**
*Wu et al. arXiv 2305.05920.*

Uses iteration level scheduling and preemption at the token granularity. Improves throughput by up to 31.4 times over vLLM under the same latency targets.

**Paper 7: PARS: Prompt-Aware Scheduling for Low-Latency LLM Serving**
*arXiv 2510.03243, 2026.*

Integrates response length prediction into vLLM for smarter request scheduling. Shows machine learning based scheduling on top of state of the art serving systems delivers measurable wins.

**Relevance:** LLM serving is an active area where ML based scheduling matters. Open ground for failure aware extensions.

### 6.6 Reinforcement Learning for Cluster Scheduling

**Paper 8: RLTune: Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters**
*Dongare et al. SoCC 2025.*

Combines RL prioritization with MILP based job placement. Trained on production traces from Microsoft Philly, Helios, and Alibaba. Improves GPU utilization by 20 percent, reduces queueing delay by 81 percent, and shortens job completion time by 70 percent.

**Relevance:** Top venue, recent. Aligns with my upcoming RL coursework.

**Paper 9: Large-scale Machine Learning Cluster Scheduling via Multi-agent Graph Reinforcement Learning**
*Zhao and Wu. arXiv 2112.13354.*

Combines graph neural networks with multi agent RL to schedule distributed training jobs while accounting for interference among co located jobs.

### 6.7 Adversarial Robustness of Monitoring

**Paper 10: Adversarial Analysis of ML-Based Anomaly Detection in Multi-Layer Network Automation**
*Researchgate 2022.*

Demonstrates a black box adversarial attack that interacts with ML based monitoring systems and generates adversarial samples to mislead anomaly detectors. The attack operates in a hard to detect manner.

**Paper 11: Robustness Testing of Data and Knowledge Driven Anomaly Detection in Cyber-Physical Systems**
*arXiv 2204.09183.*

Tests robustness of ML based safety monitors against Gaussian noise and FGSM attacks. Shows that integrating domain knowledge reduces robustness error by 54.2 percent.

**Relevance:** Establishes that monitoring systems can be attacked. The same gap exists for AI infrastructure monitoring, which is unstudied.

### 6.8 Network and Scaling Effects

**Paper 12: When Scaling Fails: Network and Fabric Effects on Distributed GPU Training Performance**
*arXiv 2603.04424, 2026.*

Analyzes why distributed training fails to scale linearly. Identifies synchronization amplification, collective communication patterns, and topology contention as root causes.

**Relevance:** Provides the systems context for why reliability and efficiency are coupled at scale.

---

## 7. Open Research Gaps

The literature points to several clear gaps that combine reliability and efficiency:

1. **Silent data corruption detection is still primitive.** NaN checks miss many failures. Multivariate time series methods using loss, gradient norms, and hardware telemetry have not been systematically applied.

2. **Cascading failures across GPU clusters are not well modeled.** Most work focuses on single node prediction. Graph based methods that capture cluster topology are underexplored.

3. **Inference scheduling does not handle failures.** Existing schedulers optimize throughput and latency but assume hardware works correctly. Failure aware scheduling is missing.

4. **Monitoring systems can be attacked.** Adversarial robustness of the ML models used to monitor AI infrastructure is essentially unstudied.

5. **Checkpointing strategies are static.** Smarter, prediction driven checkpointing that balances overhead against expected failure rate is an open area.

---

## 8. Proposed Research Directions

Three concrete directions for discussion with Prof. Sengupta. Each combines reliability and efficiency.

### Direction A: Time Series Detection of Silent Data Corruption with Lightweight Overhead

Apply the lab's time series and anomaly detection expertise to detect silent data corruption during LLM training. The system uses multivariate signals (loss, gradient norms, attention statistics, hardware metrics) to catch corruption that simple NaN checks miss. A key constraint is that the detection itself must be cheap enough to run continuously without slowing training.

- **Reliability angle:** Catch silent failures that ruin training quality.
- **Efficiency angle:** Avoid expensive checkpoint rollback by detecting corruption early. Minimize detection overhead.
- **Lab fit:** Very high. Direct extension of existing time series work.
- **Novelty:** High. Current methods are basic.
- **Tractability:** Good. Lab has compute. Small LLaMA models work as testbeds.

### Direction B: Failure Aware Inference Scheduling with Hybrid Heuristic-RL Policies

Build a hybrid, failure-aware inference scheduler that optimizes for latency and throughput while dynamically routing around degraded or failing nodes. Recognizing that production environments (like Anthropic or LinkedIn) require sub-millisecond decision times where pure RL policies are too slow, this approach uses a lightweight, supervisory reinforcement learning/ML model that updates dynamic heuristic parameters in the critical path (such as in vLLM or continuous batching systems) rather than executing a heavy RL step for every token or request.

- **Reliability angle:** Graceful degradation, proactive routing around unstable nodes, and smart traffic rebalancing.
- **Efficiency angle:** Minimizes tail latency ($p99$) and maximizes cluster-wide GPU utilization under hardware degradation.
- **Lab fit:** Moderate to high. Combines reinforcement learning (new for the lab) with systems reliability (core lab area).
- **Novelty:** High. Schedulers like vLLM optimize for speed but assume hardware is healthy; failure-aware hybrid scheduling is highly novel.
- **Tractability:** Moderate. Leverages existing schedulers (vLLM) and focuses on telemetry-driven heuristic routing rather than reinventing the serving engine.

### Direction C: Graph Neural Networks for Cascading Failure Prediction in GPU Clusters

Model GPU clusters as graphs and use graph neural networks to predict how network fabric congestion or physical hardware degradation propagates across neighboring nodes. These predictions can drive preemptive checkpointing and intelligent job placement decisions.

- **Reliability angle:** Predict cascading fabric/collective communication failures before they stall the entire training run.
- **Efficiency angle:** Saves massive amounts of compute by predicting bottlenecks and optimizing cluster network topologies.
- **Lab fit:** High. Directly builds on the lab's recent NeurIPS 2025 graph ML work.
- **Novelty:** High. Network-aware cascading failure modeling at the graph topology level is currently underexplored.
- **Gotcha (Telemetry Data Scarcity):** Production-grade GPU topologies, InfiniBand/RoCE traffic traces, and detailed failure logs are highly proprietary trade secrets of hyperscalers. Validating a GNN model in a university lab setting relies heavily on synthetic cluster simulation, making practical real-world validation extremely difficult.
- **Tractability:** Low to Moderate. Highly dependent on simulator fidelity or successfully securing an open-source anonymized cluster trace.

---

## 9. Summer and Fall Plan

| Period | Activity |
|---|---|
| Late May to mid June | Deep literature review of the 12 papers and their key citations. Commit to one direction. |
| Mid June to end of July | Set up baseline experiments on lab compute (Vast AI, Lambda Tensorbook). Reproduce key results from the anchor paper. |
| August | Run initial experiments with novel approach. Generate preliminary results. Draft introduction and related work. |
| Fall semester | Full experimental campaign. Target a workshop or conference submission. Integrate findings into the master's project. |

The summer plan is compatible with the LinkedIn AI Infrastructure internship, since the summer work is mostly literature review and infrastructure setup.

---

## 10. Summary

This research domain combines reliability and efficiency in AI infrastructure. The two goals are coupled and cannot be cleanly separated at scale. The lab has the right techniques for this work. The literature shows the field is active, recent, and not yet saturated. Three concrete research directions are ready for discussion. Each produces work that is publishable, useful to industry, and aligned with the lab's long term trajectory.

---

## References

1. Liu, H., Li, Z., Tan, C., et al. (2023). Predicting GPU Failures With High Precision Under Deep Learning Workloads. *Proceedings of SYSTOR 2023*.
2. (2026). LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training. arXiv:2604.10390.
3. (2026). Exploring Silent Data Corruption as a Reliability Challenge in LLM Training. arXiv:2604.00726.
4. Yao et al. (2025). Localizing Irregularities in LLM Training with Mega-scale Telemetry. *NSDI 2025*.
5. (2026). SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training. arXiv:2603.03592.
6. Wu et al. (2023). Fast Distributed Inference Serving for Large Language Models. arXiv:2305.05920.
7. (2026). PARS: Prompt-Aware Scheduling for Low-Latency LLM Serving. arXiv:2510.03243.
8. Dongare, S., Khan, R., Albahar, H., et al. (2025). Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads. *SoCC 2025*.
9. Zhao, X. and Wu, C. Large-scale Machine Learning Cluster Scheduling via Multi-agent Graph Reinforcement Learning. arXiv:2112.13354.
10. (2022). Adversarial Analysis of ML-Based Anomaly Detection in Multi-Layer Network Automation.
11. (2022). Robustness Testing of Data and Knowledge Driven Anomaly Detection in Cyber-Physical Systems. arXiv:2204.09183.
12. (2026). When Scaling Fails: Network and Fabric Effects on Distributed GPU Training Performance. arXiv:2603.04424.
