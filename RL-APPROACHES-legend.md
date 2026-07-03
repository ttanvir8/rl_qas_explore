# Reinforcement Learning-Based QAS: A Sub-Categorized Breakdown

Reinforcement learning is the single largest methodological family in the [Papers](README.md#papers) section of the main [README](README.md) — 24 of the ~68 listed papers frame quantum architecture search as an RL problem. This document pulls those 24 out and organizes them into sub-categories based on *what part of the RL pipeline each paper's contribution targets*: the core policy-optimization algorithm, the curriculum/training regime, the action space, the function approximator, or an orthogonal efficiency lever like the replay buffer.

Some papers touch more than one axis; each is placed under its primary contribution, with a cross-reference (↳) noted where relevant.

## Table of Contents
- [Summary](#summary)
- [Publication Status Legend](#publication-status-legend)
- [1. Foundational Policy-Gradient / Actor-Critic QAS](#1-foundational-policy-gradient--actor-critic-qas)
- [2. Curriculum & Continual RL](#2-curriculum--continual-rl)
- [3. Multi-Agent & Distributed RL](#3-multi-agent--distributed-rl)
- [4. RL with Composite Actions / Program Synthesis (Gadgets)](#4-rl-with-composite-actions--program-synthesis-gadgets)
- [5. Policy-Network Architecture Variants](#5-policy-network-architecture-variants)
- [6. Model-Based & Sample-Efficient RL](#6-model-based--sample-efficient-rl)
- [7. Tensor-Network-Assisted RL](#7-tensor-network-assisted-rl)
- [8. RL for Circuit Compilation & Optimization](#8-rl-for-circuit-compilation--optimization)
- [9. Hybrid-Action & Reward-Engineering RL](#9-hybrid-action--reward-engineering-rl)
- [10. Sample Efficiency: Replay-Buffer Engineering](#10-sample-efficiency-replay-buffer-engineering)
- [11. RL-QAS Analysis, Benchmarks & Frameworks](#11-rl-qas-analysis-benchmarks--frameworks)
- [Appendix: RL Design-Space Comparison Tables](#appendix-rl-design-space-comparison-tables)
  - [A. State Representation](#a-state-representation)
  - [B. Reward Engineering](#b-reward-engineering)
  - [C. Action Space](#c-action-space)
  - [D. Agent & Network Architecture](#d-agent--network-architecture)
- [Appendix 2: Additional RL-QCO Papers (Reward, Action Space & Agent Design)](#appendix-2-additional-rl-qco-papers-reward-action-space--agent-design)
- [Appendix 3: A*-Conference Deep-Dive](#appendix-3-a-conference-deep-dive)
- [Appendix 4: Q1-Journal Deep-Dive](#appendix-4-q1-journal-deep-dive)

## Summary

| # | Sub-category | Count | Defining question |
|---|--------------|-------|--------------------|
| 1 | Foundational Policy-Gradient / Actor-Critic | 6 | What's the base RL algorithm (A2C/PPO/A3C) applied directly to circuit construction? |
| 2 | Curriculum & Continual RL | 4 | How is the training *regime* staged over time or across changing conditions? |
| 3 | Multi-Agent & Distributed RL | 1 | Is circuit construction split across multiple cooperating agents? |
| 4 | Composite Actions / Program Synthesis (Gadgets) | 2 | Does the action space include higher-level composite gates? |
| 5 | Policy-Network Architecture Variants | 2 | What function approximator represents the policy/value function? |
| 6 | Model-Based & Sample-Efficient RL | 1 | Does the agent learn/use a model of the environment to cut evaluations? |
| 7 | Tensor-Network-Assisted RL | 1 | Is the search warm-started from a tensor-network approximation? |
| 8 | RL for Circuit Compilation & Optimization | 2 | Is RL optimizing an *existing* circuit rather than designing from scratch? |
| 9 | Hybrid-Action & Reward-Engineering RL | 2 | Does the contribution lie in the action representation or reward shaping? |
| 10 | Replay-Buffer Engineering | 1 | Is the lever the experience-replay mechanism itself? |
| 11 | Analysis, Benchmarks & Frameworks | 2 | Does the paper study/benchmark RL-QAS rather than propose a new algorithm? |

## Publication Status Legend

Each paper title below is tagged with a text-formatted tag showing where it actually landed peer review, sourced from [PUBLICATION-STATUS.md](PUBLICATION-STATUS.md) (checked as of 2026-07-01). This is a text-only variant of the image-badge legend in [RL-APPROACHES.md](RL-APPROACHES.md): tag *weight* (bold+code → italic+code → code → italic) stands in for badge *color*, so the strongest venues still jump out at a glance without loading external images:

| Tag | Meaning |
|---|---|
| **`Q1 · Journal`** | Published in a Scimago **Q1** journal (top quartile) |
| **`A* · Conference`** | Published at a CORE **A\*** conference (NeurIPS, ICML, ICLR, AAAI, HPCA) |
| *`Q2 · Journal`* | Published in a Scimago **Q2** journal |
| `Peer-Reviewed · Other` | Peer-reviewed, but a workshop/companion track or a non-CORE-tiered venue (e.g. IEEE QCE) |
| _Preprint · Only_ | arXiv only — no journal/conference acceptance found |

Of the 24 RL papers below: **7 Q1**, **3 A\***, **1 Q2**, **4 other peer-reviewed**, **9 preprint-only**. See [PUBLICATION-STATUS.md](PUBLICATION-STATUS.md) for full methodology and caveats.

---

## 1. Foundational Policy-Gradient / Actor-Critic QAS
Papers whose core contribution is applying (or algorithmically refining) a standard policy-gradient / actor-critic method — A2C, A3C, PPO — directly to the sequential gate-placement problem, typically for state preparation or task-specific circuit design, without a specialized curriculum, action space, or architecture twist.

- [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) _Preprint · Only_ — the foundational DRL-QAS paper; A2C and PPO agents build GHZ-state circuits from Pauli-X/Y/Z expectation values alone.

  Automates quantum circuit design using deep reinforcement learning (DRL), addressing the challenge of creating efficient gate sequences for target states (e.g., multi-qubit GHZ states) without expert knowledge. The DRL agent, trained via A2C and PPO algorithms, accesses only Pauli-X/Y/Z expectation values and predefined quantum operations to optimize gate synthesis. This framework eliminates the need for embedded quantum physics principles in the agent, demonstrating generality for diverse DRL architectures and gate compilation studies, enabling resource-efficient quantum circuit development.
- [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) **`Q1 · Journal`** *Scientific Reports* — refines PPO itself with a rollback clipping function and KL-divergence trust-region constraints for more stable, monotonic policy improvement.

  Enhances deep reinforcement learning-based QAS by addressing limitations in policy ratio constraints and trust region enforcement from prior Proximal Policy Optimization (PPO) methods. It introduces a rollback clipping function to strictly bound the probability ratio between old and new policies, preventing destabilizing deviations during training. Simultaneously, trust region constraints (via KL-divergence) ensure policies remain within optimization-safe zones, guaranteeing monotonic performance improvements.
- [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) `Peer-Reviewed · Other` *QCCC'23 workshop* — asynchronous advantage actor-critic (A3C) for multi-qubit GHZ-state preparation.

  Automates multi-qubit GHZ state preparation using the asynchronous advantage actor-critic (A3C) algorithm, building on prior DRL-based QAS methods. The agent, devoid of pre-encoded quantum knowledge, interacts with a predefined gate set and observes only Pauli-X/Y/Z expectation values to iteratively construct circuits.
- [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) `Peer-Reviewed · Other` *IEEE QCE 2024* — applies deep RL to discover QML model architectures balancing complexity and NISQ feasibility.

  A novel approach to designing effective quantum machine learning (QML) models using deep reinforcement learning (RL). The authors address the challenge of balancing model complexity and feasibility on Noisy Intermediate-Scale Quantum (NISQ) devices by employing RL to discover proficient QML model architectures tailored for specific supervised learning tasks.
- [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) **`Q1 · Journal`** *Machine Learning: Science and Technology* — RL with a convolutional-network agent and a composite entropy/energy reward, applied to a new target task (SYK thermal states).

  Addresses the significant challenge of preparing thermal states for large systems of the Sachdev-Ye-Kitaev (SYK) model on near-term quantum processors, which is complicated by the rapid increase in complexity of parameterized quantum circuits as system size grows. The authors propose a scalable framework that integrates reinforcement learning with convolutional neural networks, iteratively optimizing quantum circuits and their parameters using a composite reward signal based on entropy and the expectation values of the SYK Hamiltonian.
- [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) _Preprint · Only_ — RL agent discovers low-complexity, high-accuracy circuits for Iris and binary-MNIST classification.

  Introduces RL for quantum architecture search to discover effective circuit architectures for classification tasks. The evaluation uses the Iris and binary MNIST datasets. The agent autonomously discovers low-complexity circuit designs that achieve high test accuracy. Results show that RL is a viable approach for automated architecture search in quantum machine learning.

## 2. Curriculum & Continual RL
Papers where the central idea is *staging* the training process — progressively increasing task difficulty (curriculum) or transferring a policy across shifting environments/noise conditions (continual learning) — rather than the base algorithm itself.

- [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) _Preprint · Only_ — PPR-DQL reuses policies learned under one noise environment to accelerate learning under new ones, avoiding full retraining.

  Enhances quantum circuit design by integrating continual learning to address changing device noise, unlike prior DRL methods requiring frequent retraining. By reusing learned policies across noise environments, PPR-DQL accelerates the discovery of quantum gate sequences (e.g., generating two-qubit Bell states faster than training from scratch) while reducing resource overhead.
- [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) **`A* · Conference`** *NeurIPS 2021* — feedback-driven curriculum learning dynamically adjusts task complexity to balance expressivity against circuit depth.

  An autonomous algorithm that balances circuit expressivity and depth constraints for near-term quantum devices. Using feedback-driven curriculum learning, the method dynamically adjusts task complexity based on real-time performance, incrementally refining ground-state energy estimates while minimizing circuit depth.
- [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) **`A* · Conference`** *ICLR 2024* — combines a 3D tensor circuit encoding with an episode-halting curriculum that prioritizes shorter circuits under hardware noise.

  Addresses noise-aware circuit design by integrating a 3D tensor-based encoding (capturing gate types, qubits, and circuit depth) with reinforcement learning to efficiently explore the architecture space. The method employs an episode halting scheme to prioritize shorter circuits and a modified simultaneous perturbation stochastic approximation (SPSA) optimizer for noise-resilient parameter tuning, accelerating convergence. A Pauli-transfer matrix (PTM) simulator in the Pauli-Liouville basis enables efficient noisy circuit simulations, combining gate operations with hardware error models.
- [QAS-QTNs: Curriculum Reinforcement Learning-Driven Quantum Architecture Search for Quantum Tensor Networks (2025)](https://arxiv.org/abs/2507.12013) `Peer-Reviewed · Other` *IEEE QCE 2025* — extends the CRLQAS curriculum approach and benchmarks it across multiple classical and quantum-enhanced RL agents.

  Follows the developments of CRLQAS, applying curriculum reinforcement learning-driven quantum architecture search and additionally extends performance assessment across multiple RL agents and strategies. This includes benchmarking both classical and quantum-enhanced agents for robust circuit optimization as complexity increases.

## 3. Multi-Agent & Distributed RL
Papers that decompose the circuit-construction problem across multiple cooperating agents instead of a single global policy.

- [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) _Preprint · Only_ — each agent controls a local block of the circuit, shrinking per-agent action-space dimensionality and speeding convergence.

  Introduces a multi-agent reinforcement learning framework for quantum architecture search, where each agent controls a local block of a parameterized quantum circuit rather than using a single global agent. This distributed design reduces the dimensionality of each agent's action space, leading to faster convergence and lower computational cost when searching for problem-specific quantum circuits.

## 4. RL with Composite Actions / Program Synthesis (Gadgets)
Papers that enrich the RL action space with higher-level, composite "gadget" gates (synthesized sub-programs) rather than only elementary gates, to tackle harder problems or scale to more qubits.

- [Reinforcement learning with learned gadgets to tackle hard quantum problems on real hardware (2025)](https://arxiv.org/abs/2411.00230) (GRL) **`Q1 · Journal`** *Communications Physics* — integrates RL with program synthesis, adding composite gates ("gadgets") to the action space for ground-state approximation tasks.

  Integrates reinforcement learning with program synthesis to design quantum circuits for complex tasks. By incorporating composite gates (gadgets) into the action space, GRL enhances the exploration of parameterized quantum circuits (PQCs) for tasks such as approximating ground states of quantum Hamiltonians.
- [Scaling the Automated Discovery of Quantum Circuits via Reinforcement Learning with Gadgets (2025)](https://arxiv.org/abs/2503.11638) _Preprint · Only_ — extends the gadget concept specifically to address RL's poor scaling with circuit complexity, enabling discovery of more complex circuits/codes.

  Presents a novel approach to enhancing the scalability of reinforcement learning in quantum circuit design. The main objective is to address the challenges associated with RL's limited scalability due to the exponential increase in computation times with growing circuit complexity. The authors utilize the concept of "gadgets" which are composite gates that can be incorporated into RL frameworks to improve efficiency and enable the discovery of highly complex quantum codes.

## 5. Policy-Network Architecture Variants
Papers whose contribution is primarily *which function approximator* implements the policy or value network, rather than the RL algorithm or training regime.

- [KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search (2024)](https://arxiv.org/abs/2406.17630) **`Q1 · Journal`** *EPJ Quantum Technology* — replaces the MLP inside the RL agent with a Kolmogorov-Arnold Network, using spline-based learnable activations for fewer parameters and better interpretability.

  Addresses limitations of traditional MLP-based QAS methods by leveraging the Kolmogorov-Arnold theorem to design quantum circuits with improved interpretability and efficiency. KANs replace MLPs' fixed activation functions with spline-based learnable univariate functions, reducing the number of parameters while enhancing adaptability.
- [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) *`Q2 · Journal`* *Quantum Information Processing* — uses a recurrent neural network as the policy-gradient controller, with a layer-based search to speed up computation.

  Uses deep reinforcement learning to automatically design quantum circuit architectures. The main objective is to find the optimal quantum circuit composition architecture for a given task, enhancing the performance capability of quantum algorithms. The approach involves learning the sampling of the circuit architecture through a reinforcement learning-based controller and applying layer-based search to accelerate computational efficiency.

## 6. Model-Based & Sample-Efficient RL
Papers where the agent explicitly learns or exploits a model of the environment/task to reduce the number of expensive circuit evaluations, as opposed to purely model-free learning.

- [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) **`Q1 · Journal`** *Machine Learning: Science and Technology* — a model-based RL algorithm reduces required circuit evaluations, using a layered structure to shrink the search space and weigh solution quality, hardware constraints, and depth jointly.

  Addresses the challenge of heuristic architecture selection by employing a model-based reinforcement learning algorithm, which reduces the number of necessary circuit evaluations and provides a sample-efficient framework. The researchers utilize a layered circuit structure to significantly reduce the search space and account for multiple objectives, such as solution quality, hardware restrictions, and circuit depth.

## 7. Tensor-Network-Assisted RL
Papers that warm-start or otherwise couple the RL search process with a tensor-network approximation of the solution, for scalability to larger qubit counts.

- [TensorRL-QAS: Reinforcement learning with tensor networks for scalable quantum architecture search (2025)](https://arxiv.org/abs/2505.09371) **`A* · Conference`** *NeurIPS 2025* — warm-starts the search with a matrix-product-state approximation of the target solution, narrowing the search space and accelerating convergence up to 20 qubits.

  Addresses the scalability issues faced by RL-based quantum architecture search (QAS) methods, which encounter significant computational and training costs as the number of qubits, circuit depth, and noise increase. By warm-starting the architecture search with a matrix product state approximation of the target solution, TensorRL-QAS effectively narrows the search space and accelerates convergence to the desired solution. The framework is tested on several quantum optimization problems of up to 20-qubit.

## 8. RL for Circuit Compilation & Optimization
Papers applying RL to *optimize or compile* an already-specified circuit or operation (e.g., minimizing gate/T-count on given hardware) rather than designing a novel architecture from scratch for a learning task.

- [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) _Preprint · Only_ — a CNN-based DRL agent autonomously learns generic, hardware-specific strategies to optimize arbitrary circuits on a target architecture.

  The key objective is to address the gap in current optimization methods that overlook hardware-specific details, an essential consideration for near-term quantum devices. The authors employ a deep convolutional neural network as an agent to autonomously learn and apply generic strategies to optimize arbitrary circuits on a specific architecture.
- [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) **`Q1 · Journal`** *Nature Machine Intelligence* — deep RL minimizes T-gate count for fault-tolerant computation by exploiting the connection between T-count optimization and tensor decomposition.

  A deep reinforcement learning-based method designed to optimize quantum circuits by minimizing the number of costly T gates required for fault-tolerant quantum computation. By leveraging the connection between T-count optimization and tensor decomposition, this approach can incorporate domain-specific quantum knowledge and utilize specialized gadgets, leading to significant reductions in T-count compared to traditional methods.

## 9. Hybrid-Action & Reward-Engineering RL
Papers whose central contribution is *how actions are represented* (mixing discrete structure choices with continuous parameters in one policy) or *how the reward is engineered* to jointly balance competing objectives.

- [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) _Preprint · Only_ — a single policy jointly learns discrete gate placement and continuous parameter initialization for VQE, reusing optimization experience as a warm start instead of external restarts.

  Introduces HyRLQAS, a hybrid-action reinforcement learning framework that simultaneously learns quantum gate placement and parameter initialization for variational quantum circuits used in VQE. By treating structure (discrete gate choices) and parameters (continuous values) within a single policy, the agent reuses optimization experience and provides informed warm-starts instead of relying solely on external optimizers that restart from scratch.
- [QASER: Breaking the Depth vs. Accuracy Trade-Off for Quantum Architecture Search (2025)](https://arxiv.org/abs/2511.16272) _Preprint · Only_ — an engineered exponential reward function jointly encodes energy error and resource costs (depth, two-qubit gate count) to steer the agent toward shallow, accurate circuits.

  Introduces QASER, a reinforcement learning–based quantum architecture search method that uses a carefully engineered exponential reward function to jointly optimize circuit depth and accuracy. By explicitly encoding both energy error and resource costs such as depth and two-qubit gate count into the reward, QASER steers the RL agent toward circuits that are simultaneously shallow and precise for quantum chemistry state preparation.

## 10. Sample Efficiency: Replay-Buffer Engineering
Papers that treat the experience-replay mechanism itself — rather than the policy, action space, or reward — as the primary lever for improving RL-QAS efficiency and robustness.

- [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) _Preprint · Only_ — introduces ReaPER+ (annealed prioritization), OptCRLQAS (curriculum RL amortizing evaluations across edits), and a noiseless-to-noisy replay-transfer strategy for data efficiency and noise robustness.

  Treats the replay buffer as a primary algorithmic lever for RL-based quantum circuit optimization and introduces three components: ReaPER+, an annealed prioritization scheme that improves sample efficiency and circuit compactness; OptCRLQAS, a curriculum RL method that amortizes expensive quantum-classical evaluations across multiple architectural edits; and a replay-buffer transfer strategy that reuses noiseless trajectories to warm-start noisy-hardware training, together yielding substantial gains in data efficiency, wall-clock time, and noise robustness on compilation, QAS, and molecular benchmarks.

## 11. RL-QAS Analysis, Benchmarks & Frameworks
Papers that study, analyze, or benchmark existing RL-QAS methods and algorithms rather than introducing a new search algorithm of their own.

- [A quantum information theoretic analysis of reinforcement learning-assisted quantum architecture search (2024)](https://arxiv.org/abs/2404.06174) **`Q1 · Journal`** *Quantum Machine Intelligence* — analyzes RL-QAS through entanglement thresholds, initial-condition sensitivity, correlation phase transitions, and per-qubit contributions to eigenvalue estimation.

  Investigates RL-QAS, specifically for variational quantum state diagonalisation problem. The study analyzes various dimensions such as entanglement thresholds, the impact of initial conditions on RL-agent performance, phase transition behavior of correlations, and the discrete contributions of qubits in deducing eigenvalues through conditional entropy.
- [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) `Peer-Reviewed · Other` *IEEE QCE 2024* — introduces the Quantum Circuit Designer (QCD) benchmarking environment and evaluates PPO, A2C, TD3, and SAC, exposing limitations in sparse-reward, high-dimensional continuous action spaces.

  Proposes a generic RL framework, formalized as a Markov decision process, that enables agents to learn low-level quantum control by selecting and parameterizing universal quantum gates to solve two main tasks: state preparation and unitary composition. Introduces the Quantum Circuit Designer (QCD), a customizable RL environment built to benchmark various RL algorithms, including PPO, A2C, TD3, and SAC, across a range of quantum objectives. Through comprehensive evaluations, the study highlights the limitations of current RL methods in exploring sparse reward landscapes and high-dimensional continuous action spaces, while showing SAC's superior performance due to its effective exploration strategies.

---

## Appendix: RL Design-Space Comparison Tables

The sections above group papers by *what part of the RL pipeline the contribution targets*. The four tables below cut across that grouping and instead compare all 24 papers along the four MDP design axes used by [Reinforcement Learning for Quantum Circuit Optimization: A Review](https://openreview.net/pdf?id=h6w1j1fjeZ) (already cited in the main [README](README.md#reviews--surveys)): **state representation**, **reward engineering**, **action space**, and **agent/network architecture**. Rows are grouped by the same §1–§11 sub-categories as above, in the same order, so the two views stay easy to cross-reference.

Data sources: for the 7 papers the review paper analyzes directly (its Tables 2/3/4/6/7 and body text), figures/formulas are quoted or paraphrased from there. For the remaining 17 papers the review doesn't cover, details were pulled from the papers' own abstracts/methods sections. Where a paper doesn't clearly specify a field (e.g. an analysis paper with no reward of its own), the cell says so rather than guessing.

### A. State Representation

| Paper | State representation | Hardware info |
| :--- | :--- | :--- |
| **§1 Foundational Policy-Gradient / Actor-Critic** | | |
| [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) | Length-$3n$ vector of single-qubit Pauli-$X/Y/Z$ expectation values, recomputed after each gate | None (hardware-agnostic) |
| [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) | Same Pauli-expectation vector convention as the base DRL-QAS setup above | None |
| [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) | Pauli-$X/Y/Z$ expectation values per qubit (same lineage as above, target: GHZ state) | None |
| [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) | $4\times L$ matrix (control/target gate location; rotation gate location+axis), with running test accuracy appended | None |
| [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) | 3D binary tensor: gate position × gate type × circuit depth | Transfers zero-shot to noisy IBM Eagle r3 hardware |
| [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) | 3D binary tensor $[Q \times (G+Q-1) \times D]$ (qubits × gate-set-size × depth) | None |
| **§2 Curriculum & Continual RL** | | |
| [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) | Pauli-$X/Y/Z$ expectation vector, dimension $3n$ | Noise environment is what the policy is reused/adapted across |
| [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) | Fixed-length gate sequence with "no-gate" padding; angles excluded, energy appended as a scalar | None |
| [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) | Circuit tensor (qubits × moments × gate types) | Noise injected into the environment, not encoded in the state |
| [QAS-QTNs (2025)](https://arxiv.org/abs/2507.12013) | Tensor-network encoding (MPS/TTN representation) of the current quantum state | None |
| **§3 Multi-Agent & Distributed RL** | | |
| [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) | Per-agent local observation/action pair $(o_t^i, a_{t-1}^i)$; each agent owns one circuit block | None |
| **§4 Composite Actions / Program Synthesis (Gadgets)** | | |
| [Reinforcement learning with learned gadgets... (2025)](https://arxiv.org/abs/2411.00230) (GRL) | Refined binary tensor $T_{max}\times((N+N_{1q})\times N)$ | Gadgets are designed to transfer via device-compatible decompositions |
| [Scaling the Automated Discovery of Quantum Circuits via RL with Gadgets (2025)](https://arxiv.org/abs/2503.11638) | Binary stabilizer tableau, $(n-k)\times 2n$ matrix of code stabilizer generators | None |
| **§5 Policy-Network Architecture Variants** | | |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) | Tensor $D_{max}\times N\times(N+N_{1q})$ (2-qubit gate block + 1-qubit gate block) | None |
| [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) | Sequence of gate tuples {begin-qubit, end-qubit, gate type} built so far, fed via an embedding layer | None |
| **§6 Model-Based & Sample-Efficient RL** | | |
| [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) | Layered encoding circuit $U_t$ (each layer applies a chosen unitary to all accessible nearest-neighbor qubits) | Topology fixed (implicit in layer structure) |
| **§7 Tensor-Network-Assisted RL** | | |
| [TensorRL-QAS (2025)](https://arxiv.org/abs/2505.09371) | Binary tensor $(D_{MPS}+D)\times N\times(N+N_{1q})$; MPS warm-start circuit occupies the first block (trainable-init) or seeds only the initial statevector (fixed-init) | None |
| **§8 RL for Circuit Compilation & Optimization** | | |
| [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) | Circuit tensor (qubits × moments × gate classes) | None |
| [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) | Residual signature tensor $\mathcal{T}^{(s)}$ plus history of previously chosen rank-1 factors | None |
| **§9 Hybrid-Action & Reward-Engineering RL** | | |
| [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) | CNOT adjacency block ($N\times N$) + rotation-gate one-hot block ($N\times 3$) + rotation-angle values, indexed by circuit "moment" | None |
| [QASER (2025)](https://arxiv.org/abs/2511.16272) | 3D binary tensor $D_{max}\times((N\times N_{1q})\times N)$ | None |
| **§10 Sample Efficiency: Replay-Buffer Engineering** | | |
| [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) | Task-dependent: $2\times2^N$ unitary-matrix encoding (compilation) or circuit tensor extending CRLQAS's (QAS) | Noiseless-to-noisy replay transfer is the paper's core hardware-facing lever |
| **§11 RL-QAS Analysis, Benchmarks & Frameworks** | | |
| [A quantum information theoretic analysis of RL-assisted QAS (2024)](https://arxiv.org/abs/2404.06174) | Not a new design — analyzes the existing RL-VQSD tensor-based ansatz encoding (Patel et al.-style) | N/A (analysis paper) |
| [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) (qcd-gym) | Full state vector or unitary plus target state | Simulator-only |

### B. Reward Engineering

| Paper | Reward formulation | Objective(s) |
| :--- | :--- | :--- |
| **§1 Foundational Policy-Gradient / Actor-Critic** | | |
| [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) | Dense: $r=-0.01$ per step; terminal $r=F-0.01$ on reaching the goal | Fidelity, path length |
| [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) | Dense: $-0.01$/step; terminal fidelity $F$ on success | Fidelity, path length |
| [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) | Fidelity-to-GHZ-target based, likely dense/step-penalized (exact formula not accessible — paper paywalled) | Fidelity |
| [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) | Dense, 3-case piecewise: success $\propto (y_l/y_{target})(L-l)$; failure $\propto -(y_{target}-y_l)/y_{target}\cdot l$; else clipped accuracy-delta | Accuracy vs. gate count |
| [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) | Terminal $\pm5$ (free energy vs. threshold) else dense energy-improvement term; extended to $0.6\,E_{term}+0.4\,F_{term}$ | Free energy, thermal fidelity |
| [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) | Dense: accuracy-delta + complexity (depth/gate-count) terms; $+100$ bonus on target hit; $-0.01$ illegal-action penalty | Accuracy, circuit complexity |
| **§2 Curriculum & Continual RL** | | |
| [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) | Dense: Fidelity $-\,0.01\times$steps; episode ends at $F>0.95$ or 20-step cap | Fidelity, step count |
| [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) | $+5$ if $E_t<\xi$; $-5$ if $t\ge L$ and $E_t\ge\xi$; else $\max\!\big(\tfrac{E_{t-1}-E_t}{E_{t-1}-E_{min}},-1\big)$ | Energy, episode length |
| [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) | Same threshold structure as above, extended with a random-halting curriculum (negative-binomial episode length) and a feedback-driven adaptive $\xi$ | Energy, gate efficiency |
| [QAS-QTNs (2025)](https://arxiv.org/abs/2507.12013) | $+5$ success / $-5$ fail, else shaped $\max\!\big(\tfrac{C_{t-1}-C_t}{C_{t-1}-C_{min}},-1\big)$; curriculum raises task difficulty via density-ratio-weighted sampling | Cost function, curriculum progress |
| **§3 Multi-Agent & Distributed RL** | | |
| [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) | Dense shared team reward $r=2\eta-\rho_t$ ($\eta$ = normalized approx. ratio to ground energy, $\rho_t$ = circuit-length penalty) | Ground-state approximation, circuit length |
| **§4 Composite Actions / Program Synthesis (Gadgets)** | | |
| [Reinforcement learning with learned gadgets... (2025)](https://arxiv.org/abs/2411.00230) (GRL) | Terminal+dense hybrid: $+r$ if $E_t<\zeta$; $-r$ on timeout; else $\max(\Delta/|C_{t-1}-C_{min}|,-1)$ | VQE ground-state energy |
| [Scaling the Automated Discovery of Quantum Circuits via RL with Gadgets (2025)](https://arxiv.org/abs/2503.11638) | Dense, Knill-Laflamme-based: $r_t=-[\Sigma KL(t)-\Sigma KL(t-1)]$; success at $\Sigma KL=0$ | Error-correcting-code validity |
| **§5 Policy-Network Architecture Variants** | | |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) | State-prep: bonus at fidelity $\ge0.98$, else dense fidelity value. Chemistry: terminal $\pm5$ ($\xi=0.0016$ Ha) + dense shaping | Fidelity or VQE energy |
| [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) | Dense: per-episode validation accuracy of the trained circuit, given after each added layer | Task accuracy |
| **§6 Model-Based & Sample-Efficient RL** | | |
| [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) | Predicted via MuZero's learned reward/latent-dynamics model rather than a hand-specified formula; grounded in QML task performance | QML task performance, simulator-call budget |
| **§7 Tensor-Network-Assisted RL** | | |
| [TensorRL-QAS (2025)](https://arxiv.org/abs/2505.09371) | Paper states it reuses the same terminal $\pm5$ + dense-shaping reward family as Ostaszewski et al. / CRLQAS (chemical-accuracy threshold) | Energy |
| **§8 RL for Circuit Compilation & Optimization** | | |
| [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) | Dense: $r_t=q(s_{t+1})-q(s_t)$, $q=d+\eta n,\ \eta=0.2$ | Depth, gate count |
| [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) | TensorGame: $-1$ per move; gadget-aware bonuses (e.g. a completed 7-factor Toffoli gadget nets $-2$ instead of $-7$); further penalty for unsuccessful games | T-count / tensor rank |
| **§9 Hybrid-Action & Reward-Engineering RL** | | |
| [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) | Same family as Ostaszewski/KANQAS: $+5/-5$ terminal + dense shaping, recomputed after each incremental extension and angle refit | Energy |
| [QASER (2025)](https://arxiv.org/abs/2511.16272) | Dense exponential: $r_t=\big(\tfrac{M_{D_t}}{D_t+1}+\tfrac{M_{C_t}}{C_t+1}\big)\tfrac{C_t}{C_{min}}$ | Depth, gate cost, energy error |
| **§10 Sample Efficiency: Replay-Buffer Engineering** | | |
| [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) | Sparse fidelity-threshold (compilation) or CRLQAS-style energy threshold (QAS); reshaped via the paper's ReaPER+ prioritized replay | Fidelity or energy, noise robustness |
| **§11 RL-QAS Analysis, Benchmarks & Frameworks** | | |
| [A quantum information theoretic analysis of RL-assisted QAS (2024)](https://arxiv.org/abs/2404.06174) | Analyzes RL-VQSD's reward: terminal bonus if purity-cost $C_t(\theta)<\zeta$, else dense $-\log(C_t(\theta)-\zeta)$ | State purity |
| [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) (qcd-gym) | Task/environment-configurable — the paper's contribution is exposing sparse-reward, high-dimensional continuous-action settings as a shared failure mode, not one fixed reward | Configurable (fidelity/energy per task) |

### C. Action Space

| Paper | Action granularity | D/H |
| :--- | :--- | :--- |
| **§1 Foundational Policy-Gradient / Actor-Critic** | | |
| [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) | Single gate (X, Y, Z, H, rotation, or CNOT) applied to a chosen qubit | D |
| [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) | Single gate ($U(\pi/4)$, X, Y, Z, H, or CNOT) on a specific qubit | D |
| [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) | Single/two-qubit gate applied to a qubit | D |
| [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) | Rotation-gate axis (X/Y/Z) + target qubit; angle handled classically | D |
| [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) | Appends one gate (CNOT, RX/RY/RZ) per step; angles excluded from the action | D |
| [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) | (gate, qubit-index) pair from {Rx, Ry, Rz, CNOT} | D |
| **§2 Curriculum & Continual RL** | | |
| [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) | Single gate (X, Y, Z, H, $\pi/4$-rotation, or CNOT) on a specific qubit | D |
| [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) | Single gate insertion; flat space $\mathcal{A}=(\mathcal{G}_1\times[n])\cup(\mathcal{G}_2\times[n]\times[n])$, $\mathcal{G}_1=\{R_x,R_y,R_z\}$, $\mathcal{G}_2=\{\text{CNOT}\}$ | D |
| [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) | Single gate/block, topology-filtered; illegal placements masked with $Q(s,a)=-\infty$ | D |
| [QAS-QTNs (2025)](https://arxiv.org/abs/2507.12013) | Single-qubit rotation or CNOT selection; $3n+2n^2$ actions | D |
| **§3 Multi-Agent & Distributed RL** | | |
| [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) | Per-agent gate type {Rx, Ry, CNOT to neighbor ±1} × target qubit, plus a skip/no-op token | D |
| **§4 Composite Actions / Program Synthesis (Gadgets)** | | |
| [Reinforcement learning with learned gadgets... (2025)](https://arxiv.org/abs/2411.00230) (GRL) | Learned/synthesized composite gadget (macro of multiple gates) | D |
| [Scaling the Automated Discovery of Quantum Circuits via RL with Gadgets (2025)](https://arxiv.org/abs/2503.11638) | Base CNOTs (connectivity-constrained) plus composite Clifford gadgets (DCX and higher-order variants) | D |
| **§5 Policy-Network Architecture Variants** | | |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) | State-prep: {CX, X, Y, Z, H, T}. Chemistry: {CX, RX, RY, RZ}, $2+3N$ actions | D |
| [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) | One action adds a full layer (up to 8 gates); action space size $4q+q(q-1)$ | D |
| **§6 Model-Based & Sample-Efficient RL** | | |
| [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) | Choice of a unitary layer from a fixed gate set, applied to all accessible nearest-neighbor qubits | D |
| **§7 Tensor-Network-Assisted RL** | | |
| [TensorRL-QAS (2025)](https://arxiv.org/abs/2505.09371) | Discrete gate pool {RX, RY, RZ, CNOT} | D |
| **§8 RL for Circuit Compilation & Optimization** | | |
| [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) | Single gate insertion or deletion | D |
| [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) | Rank-1 binary factor $u^{(s)}\in\{0,1\}^N$ (tensor-decomposition step, corresponds 1:1 to a T-gate) | D |
| **§9 Hybrid-Action & Reward-Engineering RL** | | |
| [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) | Discrete gate + placement head, jointly with continuous rotation-angle init/refinement heads (Parameterized-Action MDP) | H |
| [QASER (2025)](https://arxiv.org/abs/2511.16272) | {CNOT, RX, RY, RZ}, $3N+2\binom{N}{2}$ actions, with illegal-action masking | D |
| **§10 Sample Efficiency: Replay-Buffer Engineering** | | |
| [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) | Sequential gate append from fixed sets (small-angle RX/RY/RZ + HRC for compilation; RXX/RYY/RZZ-type for QAS) | D |
| **§11 RL-QAS Analysis, Benchmarks & Frameworks** | | |
| [A quantum information theoretic analysis of RL-assisted QAS (2024)](https://arxiv.org/abs/2404.06174) | Not specified beyond "selecting quantum gate configurations" (analysis paper) | D |
| [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) (qcd-gym) | Task/environment-configurable continuous gate parameters; paper's finding is that high-dimensional continuous spaces are a key bottleneck | H |

### D. Agent & Network Architecture

| Paper | RL algorithm(s) | Network architecture |
| :--- | :--- | :--- |
| **§1 Foundational Policy-Gradient / Actor-Critic** | | |
| [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) | A2C, PPO | Not stated explicitly in the review; consistent with the MLP used across this paper's derivatives |
| [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) | QAS-TR-PPO-RB (trust-region PPO with rollback clipping + KL constraints) | Standard actor-critic net (exact backbone not stated) |
| [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) | A3C | Parameterized quantum circuit (PQC) as the policy network — quantum-native, not a classical NN |
| [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) | $n$-step Double DQN (DDQN) | MLP (Linear + LeakyReLU + dropout) |
| [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) | DQN-style, $\epsilon$-greedy, Adam optimizer | 3D CNN (depth × qubit × gate type) |
| [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) | PPO | MLP, two 64-unit hidden layers |
| **§2 Curriculum & Continual RL** | | |
| [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) | PPR-DQL (Probabilistic Policy Reuse + Deep Q-Learning) | MLP, 2 hidden layers + linear output |
| [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) | DDQN | MLP (fixed-length feature vector) |
| [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) | DDQN, curriculum-driven episode halting | MLP |
| [QAS-QTNs (2025)](https://arxiv.org/abs/2507.12013) | Benchmarks A2C/PPO/DDQN/TD3 and quantum counterparts (QA2C/QPPO/QDDQN/QTD3) with prioritized experience replay | MLP, 3×30; quantum variants use a 6-qubit PQC |
| **§3 Multi-Agent & Distributed RL** | | |
| [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) | QMIX (multi-agent value factorization) | Per-agent: 2 linear layers + GRU; monotonic mixing network: 2 linear layers + ReLU |
| **§4 Composite Actions / Program Synthesis (Gadgets)** | | |
| [Reinforcement learning with learned gadgets... (2025)](https://arxiv.org/abs/2411.00230) (GRL) | DDQN | Not detailed beyond DDQN's standard Q-network |
| [Scaling the Automated Discovery of Quantum Circuits via RL with Gadgets (2025)](https://arxiv.org/abs/2503.11638) | MaxPPO (PPO variant crediting the best intermediate state in a trajectory) | Not stated explicitly |
| **§5 Policy-Network Architecture Variants** | | |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) | DDQN | Kolmogorov-Arnold Network (KAN) replacing the MLP inside DDQN — spline-based learnable activations |
| [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) | REINFORCE (Monte Carlo policy gradient) | LSTM controller (embedding → BatchNorm → Linear → ReLU → LSTM, 100 hidden units → softmax) |
| **§6 Model-Based & Sample-Efficient RL** | | |
| [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) | MuZero (latent dynamics model + Monte Carlo Tree Search) | MuZero's representation/dynamics/prediction networks |
| **§7 Tensor-Network-Assisted RL** | | |
| [TensorRL-QAS (2025)](https://arxiv.org/abs/2505.09371) | DDQN | Not detailed beyond DDQN's standard Q-network |
| **§8 RL for Circuit Compilation & Optimization** | | |
| [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) | A2C, PPO | CNN (2D gate-qubit grid); 27% depth / 15% gate-count reduction, zero-shot size generalization |
| [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) | AlphaZero-style sample-based MCTS (800 simulated trajectories per move) | Policy/value network with symmetrization layers, attention/transformer-based (inherited from AlphaTensor) |
| **§9 Hybrid-Action & Reward-Engineering RL** | | |
| [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) | REINFORCE | Shared MLP backbone (hidden sizes scale $[1000\text{–}5000]\times5$ with qubit count) branching into discrete, continuous, and refinement heads |
| [QASER (2025)](https://arxiv.org/abs/2511.16272) | DDQN, $\epsilon$-greedy (decay 0.99995), replay buffer 20,000, target-net update every 500 steps | Not stated explicitly |
| **§10 Sample Efficiency: Replay-Buffer Engineering** | | |
| [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) | DQN (compilation) / DDQN (QAS), off-policy, engineered with ReaPER+ prioritized replay + OptCRLQAS amortized curriculum | Not stated explicitly |
| **§11 RL-QAS Analysis, Benchmarks & Frameworks** | | |
| [A quantum information theoretic analysis of RL-assisted QAS (2024)](https://arxiv.org/abs/2404.06174) | DDQN, $\epsilon$-greedy, Adam optimizer (analyzing the RL-VQSD framework) | MLP, 5 hidden layers × 1000 neurons |
| [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) (qcd-gym) | PPO, A2C, SAC, TD3 (systematically benchmarked) | Not detailed — the paper's contribution is the benchmark suite, not a fixed architecture |

---

## Appendix 2: Additional RL-QCO Papers (Reward, Action Space & Agent Design)

The 24 papers above are drawn from the main [README](README.md)'s Papers list. Cross-referencing the reward-engineering (§5), action-space (§6), and agent/network-architecture (§7) sections and tables of [Reinforcement Learning for Quantum Circuit Optimization: A Review](https://openreview.net/pdf?id=h6w1j1fjeZ) turned up 26 further RL-QCO papers discussed there that are not yet part of this repo's paper list. They're recorded here for reference, grouped the same way the review groups them, with a short description each — not yet sub-categorized into §1–§11 above or added to the Appendix tables.

Publication status uses the same text-tag legend as [above](#publication-status-legend), checked 2026-07-02. Of the 26 papers below: **12 Q1**, **0 A\***, **0 Q2**, **5 other peer-reviewed**, **9 preprint-only**.

### Reward Engineering

- [Reinforcement Learning assisted Quantum Optimization (2020)](https://arxiv.org/abs/2004.12323) **`Q1 · Journal`** *Physical Review Research* — RL agent chooses QAOA control-unitary parameters from partial system information; reward is the negative energy, and the learned policy transfers across system sizes and disorder.
- [Reinforcement Learning Assisted Recursive QAOA (2022)](https://arxiv.org/abs/2207.06294) **`Q1 · Journal`** *EPJ Quantum Technology* — RL learns the variable-elimination sequence for recursive QAOA, with reward tied to the final Ising/MaxCut cost rather than circuit structure.
- [Quantum Circuit Discovery for Fault-Tolerant Logical State Preparation with Reinforcement Learning (2024)](https://arxiv.org/abs/2402.17761) **`Q1 · Journal`** *Physical Review X* — reward is defined via distance between the current and target stabilizer tableaux, letting the agent discover shallow fault-tolerant logical state-prep circuits.
- [Quantum Compiling by Deep Reinforcement Learning (2021)](https://arxiv.org/abs/2105.15048) **`Q1 · Journal`** *Communications Physics* — state is defined via probe-state (Bloch-sphere) images under the partial circuit rather than an explicit gate list; reward jointly rewards target-unitary accuracy and short gate sequences for single-qubit compiling.

### Action Space Design

- [A Reinforcement Learning Environment for Directed Quantum Circuit Synthesis (2024)](https://arxiv.org/abs/2401.07054) _Preprint · Only_ — PPO agent over a flat Clifford+T action space, used as a controlled testbed for how action-space size scales with qubit count on Bell/GHZ synthesis.
- [Reinforcement Learning Based Quantum Circuit Optimization via ZX-Calculus (2023)](https://arxiv.org/abs/2312.11597) **`Q1 · Journal`** *Quantum* — action space is (ZX rewrite rule, diagram node) pairs; a PPO+GNN agent trained on 5-qubit circuits generalizes to circuits up to 80 qubits and 2100 gates.
- [Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks (2025)](https://arxiv.org/abs/2504.03429) _Preprint · Only_ — follows the same ZX-rewrite-as-action paradigm as Riu et al. above, pairing a GNN policy with ZX-diagram rewrite locations.
- [Towards Faster Quantum Circuit Simulation Using Graph Decompositions, GNNs and Reinforcement Learning (2024)](https://openreview.net/forum?id=54060pbCKY) `Peer-Reviewed · Other` *NeurIPS'24 Math-AI workshop* — action is the choice of ZX-calculus sub-circuit decomposition; the RL+GNN agent speeds up classical circuit simulation rather than constructing a circuit.
- [Reinforcement Learning for Many-Body Ground-State Preparation Inspired by Counterdiabatic Driving (2020)](https://arxiv.org/abs/2010.03655) **`Q1 · Journal`** *Physical Review X* — each action selects a physically-motivated control generator (CD-QAOA) from a fixed set, learning solely from the final energy.
- [Quantum Many-body Simulations from a Reinforcement-Learned Exponential Ansatz (2025)](https://arxiv.org/abs/2505.01935) **`Q1 · Journal`** *Physical Review A* — action is a two-body exponential operator chosen via the contracted-Schrödinger-equation residual, yielding compact ansatz circuits for strongly correlated molecules.
- [Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes (2026)](https://arxiv.org/abs/2601.21629) _Preprint · Only_ — PPO agent selects among 7 PyTKET compiler passes plus a DoNothing action, beating the best fixed pass-sequence default on 40,000+ circuits.
- [Hybrid Discrete-Continuous Compilation of Trapped-Ion Quantum Circuits with Deep Reinforcement Learning (2023)](https://arxiv.org/abs/2307.05744) **`Q1 · Journal`** *Quantum* — a projective-simulation RL agent learns discrete gate ordering while a gradient-based optimizer handles continuous rotation angles, alternating within each episode.
- [Practical and Efficient Quantum Circuit Synthesis and Transpiling with Reinforcement Learning (2024)](https://arxiv.org/abs/2405.13196) _Preprint · Only_ — action space is defined directly over IBM's native basis gates (ECR, √X), so RL output circuits need no further transpilation.
- [qgym: A Gym for Training and Benchmarking RL-Based Quantum Compilation (2023)](https://arxiv.org/abs/2308.02536) `Peer-Reviewed · Other` *IEEE QCE 2023* — Gymnasium-style environment exposing the device coupling graph and SWAP actions for topology-aware routing/mapping research.
- [Quantum Compiling with Reinforcement Learning on a Superconducting Processor (2024)](https://arxiv.org/abs/2406.12195) _Preprint · Only_ — actions ("percepts") are generated directly from the processor's native gate set on the real coupling graph, so every action is hardware-realizable by construction.
- [Rethinking How to Act: Action-Space Engineering for Reinforcement Learning-Based Circuit Routing in Distributed Quantum Systems (2026)](https://arxiv.org/abs/2605.02389) _Preprint · Only_ — introduces direct inter-module qubit-pair actions plus masking for distributed multi-chip routing, cutting modeled execution time up to 35% versus the prior RL baseline.
- [Noise-Adaptive Quantum Circuit Mapping for Multi-Chip NISQ Systems via Deep Reinforcement Learning (2025)](https://arxiv.org/abs/2511.18079) _Preprint · Only_ (DeepQMap) — augments the action space with noise-weighted link priorities learned via a BiLSTM + attention model over multi-chip noise telemetry.

### Agent & Network Architecture

- [Enhancing Variational Quantum State Diagonalization Using Reinforcement Learning Techniques (2023)](https://arxiv.org/abs/2306.11086) **`Q1 · Journal`** *New Journal of Physics* (RL-VQSD) — the original DDQN-based algorithm paper for RL-assisted quantum state diagonalization; distinct from the RL-VQSD *analysis* paper already covered in [§11](#11-rl-qas-analysis-benchmarks--frameworks).
- [Reinforcement Learning for Variational Quantum Circuits Design (2024)](https://arxiv.org/abs/2409.05475) `Peer-Reviewed · Other` *CEUR-WS workshop proceedings* — PPO agent for VQC ansatz design.
- [Optimizing Quantum Variational Circuits with Deep Reinforcement Learning (2021)](https://arxiv.org/abs/2109.03188) _Preprint · Only_ — benchmarks TD3, SAC, and PPO for variational circuit optimization.
- [Reinforcement Learning for Parameterized Quantum State Preparation: A Comparative Study (2026)](https://arxiv.org/abs/2602.16523) _Preprint · Only_ — comparative PPO/A2C study for parameterized state-preparation circuits.
- [BenchRL-QAS: Benchmarking Reinforcement Learning Algorithms for Quantum Architecture Search (2025)](https://arxiv.org/abs/2507.12189) `Peer-Reviewed · Other` *AAAI Symposium Series 2025* — evaluates 9 RL agents (A2C/A3C, DQN-family, PPO/TPPO) across VQE, VQSD, VQC, and GHZ tasks, providing the field's first large-scale confirmation of a no-free-lunch result for RL-QAS.
- [Hybrid Reward-Driven Reinforcement Learning for Efficient Quantum Circuit Synthesis (2025)](https://arxiv.org/abs/2507.16641) **`Q1 · Journal`** *Quantum Machine Intelligence* — tabular Q-learning with a static fidelity term plus dynamic gate-congestion/revisit penalties finds minimal-depth graph-state circuits up to 7 qubits.
- [AlphaRouter: Quantum Circuit Routing with Reinforcement Learning and Tree Search (2024)](https://arxiv.org/abs/2410.05115) `Peer-Reviewed · Other` *IEEE QCE 2024* — AlphaZero-style neural-guided Monte Carlo tree search for SWAP-based routing, up to 20% lower routing overhead than prior RL baselines.
- [Automated Quantum Circuit Design with Nested Monte Carlo Tree Search (2022)](https://arxiv.org/abs/2207.00132) **`Q1 · Journal`** *IEEE Transactions on Quantum Engineering* — a search-only nested-MCTS baseline with no neural prior, competitive for circuit design up to roughly 6-7 qubits.
- [Learning-Optimized Qubit Mapping and Reuse to Minimize Inter-Core Communication in Modular Quantum Architectures (2025)](https://arxiv.org/abs/2506.09323) **`Q1 · Journal`** *Physical Review Applied* (QARMA) — an attention + GNN RL agent for modular/multi-core qubit mapping and reuse, reducing inter-core communication by up to 100%.

---

## Appendix 3: A*-Conference Deep-Dive

Of the 24 RL-QAS papers in this repo, exactly **3** cleared a CORE **A\*** venue: NeurIPS 2021, ICLR 2024, and NeurIPS 2025 (see the [publication-status legend](#publication-status-legend)). All three sit in different sub-categories above (§2 Curriculum RL, §2 Curriculum RL, §7 Tensor-Network-Assisted RL respectively) but, notably, **all three attack the same task**: RL-driven **ansatz construction for VQE ground-state estimation**. None of the 24 papers that cleared A* venues target unitary/state compilation, qubit mapping and routing, circuit simplification of an existing circuit, or fault-tolerant/T-count resource reduction — those tasks are covered by A*-adjacent-but-not-A* work instead (e.g. AlphaTensor for T-count is Nature Machine Intelligence/Q1, ZX-rewrite compilation work is *Quantum*/Q1; see [§8](#8-rl-for-circuit-compilation--optimization) and [Appendix 2](#appendix-2-additional-rl-qco-papers-reward-action-space--agent-design)). This is a concrete gap: the field's most rigorously peer-reviewed RL results all validate on the *same* problem class (molecular-Hamiltonian VQE), leaving RL for compilation/routing/FT-reduction comparatively under-vetted at top ML venues.

The three are also a genealogical chain, not independent results: CRLQAS (ICLR 2024) directly extends the Ostaszewski et al. (NeurIPS 2021) reward/curriculum design to noisy hardware, and TensorRL-QAS (NeurIPS 2025) explicitly reuses CRLQAS's reward family and DDQN backbone, adding an MPS warm start on top. So the "3 A* papers" are best read as one lineage — reward shaping → noise-aware curriculum → tensor-network warm-start — applied to a fixed task (VQE ansatz search) and a fixed benchmark family (molecular/spin Hamiltonians), progressively scaling qubit count and hardware realism.

| Axis | [Ostaszewski et al. (2021)](https://arxiv.org/abs/2103.16089) — NeurIPS 2021 | [CRLQAS (2024)](https://arxiv.org/abs/2402.03500) — ICLR 2024 | [TensorRL-QAS (2025)](https://arxiv.org/abs/2505.09371) — NeurIPS 2025 |
| :--- | :--- | :--- | :--- |
| **Task (taxonomy)** | Ansatz construction | Ansatz construction (noise-aware) | Ansatz construction (tensor-network-warm-started) |
| **Problem type** | VQE ground-state energy estimation, molecular Hamiltonian | VQE ground-state energy estimation under hardware noise | VQE ground-state energy estimation, scalability to larger qubit counts |
| **State representation** | Fixed-length gate-sequence vector with "no-gate" padding; rotation angles excluded from state, current energy estimate appended as a scalar | 3D binary tensor (qubits × moments × gate types); noise/coupling-map info injected into the *environment*, not the state | Binary tensor $(D_{MPS}+D)\times N\times(N+N_{1q})$ — the MPS-derived warm-start circuit occupies the first $D_{MPS}$ layers, the RL agent extends/edits the remaining $D$ |
| **Reward engineering** | Threshold-triggered: $+5$ if $E_t<\xi$ (chemical accuracy); $-5$ if episode ends without reaching $\xi$; else dense shaping $\max\!\big(\tfrac{E_{t-1}-E_t}{E_{t-1}-E_{min}},-1\big)$. $\xi$ tightened via intrinsically-motivated curriculum (moving threshold, amortization radius $10^{-4}$, updated every 2000 episodes) | Same $\pm5$ terminal + dense-shaping family, extended with (a) random/negative-binomial episode halting to bias toward shorter circuits and (b) a feedback-driven adaptive $\xi$ that reacts to hardware-noise-induced energy floors | Reuses the Ostaszewski/CRLQAS $\pm5$ terminal + dense-shaping reward unchanged; the paper's lever is *where the search starts* (MPS-seeded state), not the reward itself |
| **Action space** | Discrete, flat: $\mathcal{A}=(\{R_x,R_y,R_z\}\times[n])\cup(\{\text{CNOT}\}\times[n]\times[n])$ — single gate insertion per step, $3|Q|+2\binom{|Q|}{2}$ actions | Discrete, topology-filtered single gate/block insertion from {RX, RY, RZ, CNOT}; illegal (non-coupling-map) placements masked with $Q(s,a)=-\infty$ | Discrete gate pool {RX, RY, RZ, CNOT}, same insertion mechanics as CRLQAS, applied only to the non-MPS extension layers |
| **Agent architecture** | Double DQN (DDQN), MLP over the fixed-length feature vector | DDQN with curriculum-driven episode halting, MLP over the 3D tensor state | DDQN, MLP/CNN over the MPS-augmented tensor state; classical optimizer (SPSA-family) refits angles after each structural edit |
| **Benchmark instances (input)** | LiH at 3 bond distances (1.2, 2.2, 3.4 Å); 4-qubit (parity-mapped) and 6-qubit (Jordan-Wigner, symmetry-reduced) Hamiltonians | H₂ (2/3/4-qubit), LiH (4/6-qubit), H₂O (8-qubit), STO-3G basis, Jordan-Wigner/parity mappings; noiseless + depolarizing noise + IBM Mumbai/Ourense device noise profiles + finite-shot noise ($10^3$–$10^8$ shots) | BeH₂ (6-qubit), H₂O/CH₂O (8- and 10-qubit), LiH (12-qubit) for chemistry; Heisenberg (5-qubit) and TFIM (6-qubit, scaled to 20-qubit) for spin models; noiseless + 5–10× amplified depolarizing noise |
| **Output** | Parameterized ansatz circuit (ordered gate sequence + angles) meeting the chemical-accuracy threshold at minimum depth | Noise-robust ansatz circuit meeting chemical accuracy under a given hardware/noise profile, with gate/depth efficiency | Compact ansatz circuit (MPS-seeded + RL-extended) meeting chemical accuracy, scaling to system sizes prior tensor-free RL-QAS could not reach |
| **Evaluation criteria** | Energy error vs. chemical accuracy (~$10^{-3}$ Ha); circuit depth (12 vs. 17 for Hardware-Efficient ansatz, vs. 377 for UCCSD at 6 qubits); gate count; CNOT count | Energy error vs. chemical accuracy ($1.6\times10^{-3}$ Ha, $2.2\times10^{-4}$ for H₂); CNOT + rotation gate count; circuit depth; parameter count; actions-to-first-success and total successful episodes (training efficiency) under each noise profile | Energy error vs. chemical accuracy ($1.6\times10^{-3}$ Ha); success probability; CNOT/rotation/total gate count; circuit depth; classical-optimizer function evaluations (nfev); wall-clock training time; scaling behavior up to 20 qubits |

**Cross-cutting observations:**
- **Same reward family, three generations.** All three reuse the identical $\pm5$ terminal-plus-dense-shaping reward introduced by Ostaszewski et al. — the A* lineage treats reward engineering as a solved primitive and spends its novelty on the curriculum (CRLQAS) or the warm start (TensorRL-QAS) instead.
- **Same benchmark family, escalating scale.** LiH-only (2021, ≤6 qubits) → H₂/LiH/H₂O with hardware noise (2024, ≤8 qubits) → BeH₂/H₂O/CH₂O/LiH/TFIM with noise and scaling studies (2025, up to 20 qubits). Reported metrics are consistent across the lineage (energy error vs. chemical accuracy, CNOT count, depth), which is what makes the progression legible as one continued benchmark rather than three unrelated evaluations.
- **Agent architecture is static.** All three use DDQN with an MLP/CNN-style value network over a tensor/vector encoding of the partial circuit — the field's top-venue results have not yet moved to actor-critic, transformer, or model-based agents for this task (contrast with MuZero in [§6](#6-model-based--sample-efficient-rl) or REINFORCE+LSTM in [§5](#5-policy-network-architecture-variants), neither of which cleared an A* venue).
- **No A*-venue coverage for compilation/routing/simplification/FT tasks.** Every A* paper above is "build a circuit from scratch to estimate an energy." Circuit *optimization* of an already-specified circuit (§8), qubit mapping/routing ([Appendix 2](#appendix-2-additional-rl-qco-papers-reward-action-space--agent-design)), and fault-tolerant T-count/logical-state work all currently top out at Q1 journals (Nature Machine Intelligence, Physical Review X/Applied, *Quantum*) rather than A* conferences.

---

## Appendix 4: Q1-Journal Deep-Dive

Of the 24 RL-QAS papers in this repo, exactly **7** cleared a Scimago **Q1** journal (see the [publication-status legend](#publication-status-legend)): *Scientific Reports*, *EPJ Quantum Technology*, *Machine Learning: Science and Technology* (×2), *Communications Physics*, *Nature Machine Intelligence*, and *Quantum Machine Intelligence*. Unlike the [A* lineage](#appendix-3-a-conference-deep-dive) — one genealogical chain, all solving VQE ansatz construction — the Q1 set is task-diverse: fixed-target state synthesis, molecular/spin-model VQE, thermal-state preparation, QML architecture search, fault-tolerant T-count reduction, and a quantum-information-theoretic *analysis* of an existing RL-VQSD system rather than a new algorithm. Details below were extracted directly from each paper (not abstracts), cross-checked against the [state/reward/action/agent tables](#appendix-rl-design-space-comparison-tables) above.

### E1. Task, Problem & Benchmark Profile

| Paper | Task (taxonomy) | Problem type | Benchmark instances (input) | Output | Evaluation criteria (key results) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Truly proximal PPO (2023)](https://www.nature.com/articles/s41598-023-32349-2) *Sci. Rep.* | State prep & synthesis / ansatz construction | Multi-qubit entangled-state synthesis + Ising spin-glass ground-state approximation | Bell (2q), GHZ (3q), 4-qubit Sherrington–Kirkpatrick Ising ground state (target from Metropolis sampling of $H=\tfrac1n\sum J_{ij}\sigma_i^z\sigma_j^z+\sum h_i\sigma_i^z$); noiseless + depolarizing/measurement noise at $p=0.001,0.005$ | Discrete gate sequence ($U(\pi/4)$, X/Y/Z/H, CNOT) reaching target state/energy | State fidelity, runtime, convergence speed vs. QAS-PPO / QAS-A2C / QAS-PPO-RB / QAS-TR-PPO baselines (QAS-TR-PPO-RB fastest and most stable; reported graphically) |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) *EPJ QT* | Ansatz construction | Fixed-target state synthesis + molecular ground-state VQE | Bell (2q, noiseless + 2 noise levels), GHZ (3q, noiseless), H₂ (4q, STO-3G, Jordan-Wigner), LiH (4q, STO-3G, parity mapping) | Non-parametrized $\{CX,X,Y,Z,H,T\}$ circuit (state prep); parametrized $\{CX,RX,RY,RZ\}$ ansatz (chemistry) | Success prob./fidelity $\ge0.98$; chemical accuracy $1.6\times10^{-3}$ Ha; gate/CNOT count; depth; parameter count vs. MLP-DDQN — KAN +10pp success on GHZ, 73–85% fidelity where MLP fails under high noise, 10–100× fewer trainable parameters at comparable/better error |
| [SYK thermal-state prep (2025)](https://arxiv.org/abs/2501.11454) *ML: Sci. Technol.* | Ansatz construction (thermal/Gibbs state prep) | Variational thermal-state preparation, dense SYK Hamiltonian (spin/model physics) | $N=8,10,12,14$ Majoranas (4–7 qubits), $\beta=1.0/5.2/18/35$, 5 disorder realizations/size; noiseless sim + noisy sim (IBM Eagle r3 model) + real IBM Eagle r3 hardware run ($N=8$, no error mitigation) | RL-extended PQC2 (RX/RY/RZ/CNOT) appended to fixed hardware-efficient PQC1, COBYLA-optimized | Free-energy error vs. exact diagonalization; thermal-state fidelity (0.80–0.98 for $N\le12$); CNOT count $\ge100\times$ below first-order Trotterization at $N\ge12$ (polynomial vs. exponential scaling); free-energy robustness on real hardware |
| [GRL / learned gadgets (2025)](https://arxiv.org/abs/2411.00230) *Commun. Phys.* | Ansatz construction + circuit simplification (gadget-based, hardware-native) | Ground-state energy estimation, TFIM across 3 field regimes + preliminary H₂ | TFIM $N=2$–6 (10 in scalability ablation) at $h=10^{-3},5\times10^{-2},1$; H₂ at 2–3 qubits; IBM Heron-native gates $\{RZ,SX,X,CZ\}$ on fake_torino/kawasaki/quebec noise models | Native-gate circuit with learned composite gadgets (e.g. RZ·CZ, X·√X) inserted as atomic actions | Energy error vs. true ground state ($7.2\times10^{-9}$, 3-qubit TFIM, beats QNEAT/HEA); gate count $\approx3\times$ fewer than gadget-free RL; 2–4× faster training; gadget transfer to H₂ cuts error up to 43.5% |
| [MuZero QML arch. search (2024)](https://arxiv.org/abs/2406.02717) *ML: Sci. Technol.* | Ansatz construction (data-encoding/feature-map circuit) | QML architecture search for kernel-based classification/regression (not a physics Hamiltonian) | California housing regression (8 features/qubits), Quantum-Fashion-MNIST regression (8 qubits, PQK), two-curves classification (10 qubits); fixed layered gate set, max depth 10, LNN connectivity | Layered encoding circuit (one gate type/layer across all qubits, $\le10$ layers) | 5-fold CV MSE/accuracy vs. literature, random-layered/unlayered, and genetic-search baselines — MCS wins/ties all 3 tasks (e.g. MSE 0.0147 vs. 0.0165 lit. on housing); sample-efficiency framed as fewer real circuit evaluations via MuZero's learned model |
| [AlphaTensor-Quantum (2024)](https://arxiv.org/abs/2402.14396) *Nat. Mach. Intell.* | Fault-tolerant resource reduction (T-count/T-depth) | Tensor-decomposition optimization of Clifford+T arithmetic/oracle circuits | 26-circuit Amy et al. benchmark suite (adders, multipliers, Toffoli-gadget circuits, GF($2^m$) mult, QFT4, etc., 5–36 qubits, up to 72 post-compilation); GF($2^m$) mult ($m=2$–7); Cuccaro/Gidney adders; Hamming-weight phase-gradient circuits; unary iteration; FeMoco Select oracle (chemistry, up to 10 spin-orbitals) | Rank-$R$ Waring/tensor decomposition mapped back to an optimized Clifford+T circuit (T-gates + auto-detected Toffoli/CS gadgets) | T-count vs. best-known baselines — halves Cuccaro adders' Toffoli count (matches Gidney); matches classical Karatsuba scaling for GF($2^m$) mult; e.g. GF($2^{10}$)-mult 350→92, 8-bit adder 129→94; provably optimal (SMT solver) for circuits $\le20$ T gates |
| [RL-VQSD info-theoretic analysis (2024)](https://arxiv.org/abs/2404.06174) *Quantum Mach. Intell.* | Ansatz construction (diagonalizing circuit) — analysis of an existing system, not a new algorithm | Variational quantum state diagonalization (VQSD) + information-theoretic study of the resulting ansatzes | 2-qubit random/Haar mixed density matrices (100,000 states for broad analysis; 9 seeds × 5 DDQN inits for the concurrence-bound study; 25/35-state subsets for gate-cost and per-qubit correlation studies) | Diagonalizing unitary $U(\theta)$ + eigenvalue/eigenvector readout; the paper's own output is concurrence bounds, an entanglement phase-transition point, and correlation-regime maps | Purity cost $C_t(\theta)=\mathrm{Tr}(\rho^2)-\mathrm{Tr}(\rho_t^2)$; concurrence upper/lower bounds of admissible ansatzes; Pearson correlation between concurrence change and cost (phase transition at input concurrence $=0.322$); gate count/depth of admissible circuits; entanglement-enhancing (EE) block ablation ($2.2\times$ more successful episodes) |

### E2. RL Design Profile (State / Reward / Action / Agent)

| Paper | State representation | Reward engineering | Action space | Agent & network architecture |
| :--- | :--- | :--- | :--- | :--- |
| [Truly proximal PPO (2023)](https://www.nature.com/articles/s41598-023-32349-2) | Pauli-$X/Y/Z$ expectation vector, dim $3n$ | Dense: $-0.01$/step; terminal reward = achieved fidelity | 12 discrete actions (2-qubit case): $U(\pi/4)$, X, Y, Z, H per qubit + CNOT | QAS-TR-PPO-RB: PPO + rollback clipping ($F_{TR\text{-}RB}=-\alpha q_{t}(\theta_{old})$ when KL $\ge\delta$) + KL trust region ($\delta=0.03,\alpha=-0.3$); Adam, lr $=0.002$, $\gamma=0.99$, clip $=0.2$, $K=4$ epochs |
| [KANQAS (2024)](https://arxiv.org/abs/2406.17630) | Tensor $D_{max}\times N\times(N+N_{1q})$ | State prep: $R=\mathcal{R}$ if $F\ge0.98$ else $F$. Chemistry: $\pm5$ terminal + dense shaping, $\xi=0.0016$ Ha | $\{CX,X,Y,Z,H,T\}$ (state prep) or $\{CX,RX,RY,RZ\}$ (chemistry) | DDQN with a Kolmogorov-Arnold Network (B-spline order $k$, grid $G$) replacing the MLP — e.g. KAN$_{2,2}$/KAN$_{4,30}$ configs; $\epsilon$-greedy, smooth-L1 loss |
| [SYK thermal-state prep (2025)](https://arxiv.org/abs/2501.11454) | 3D binary tensor $D_{max}\times(N+N_{1q})\times N$ | $\pm5$ terminal (free-energy threshold $\zeta_F=10^{-2}$) + dense $E_{term}$; extended to $0.6E_{term}+0.4Fid_{term}$ ($\zeta_{Fid}=0.9$) | $\{RX,RY,RZ,CNOT\}$ | DDQN, 3D-CNN (up to 7 conv layers, 32–2048 channels) + FC head; Adam lr $=10^{-3}$, $\epsilon$-decay $0.99995$ |
| [GRL / learned gadgets (2025)](https://arxiv.org/abs/2411.00230) | Refined binary tensor $T_{max}\times((N+N_{1q})\times N)$ | $\pm r$ terminal (energy threshold $\zeta$, feedback-curriculum updated) + dense shaping | Native gates $\{RZ,SX,X,CZ\}$ + program-synthesized composite gadgets (fragment-grammar synthesis, score $S=L_g(D)-\lambda|g|-k\sum|p|$) | DDQN, MLP $5\times1000$; Adam lr $=10^{-4}$; COBYLA optimizes rotation parameters |
| [MuZero QML arch. search (2024)](https://arxiv.org/abs/2406.02717) | Observation $o_t=U_t$, the current layered circuit (POMDP) | Piecewise CV-score improvement: $r_g=100$ (beats global best), $r_l=50$ (beats episode best), $-1$ otherwise/invalid | 16 discrete layer-types from a fixed gate set $\{X,Y,Z,CX,CY,CZ,H,R_{x/y/z}(n\pi),CR_{x/y/z}(\pi/n),R_{x/y/z}(\arctan x)\}$, applied to all qubits (LNN entangling) | MuZero (learned representation/dynamics/prediction networks + MCTS); architecture/hyperparameters deferred to Schrittwieser et al., not disclosed in-paper |
| [AlphaTensor-Quantum (2024)](https://arxiv.org/abs/2402.14396) | Binary signature tensor $\mathcal{T}^{(s)}\in\{0,1\}^{N\times N\times N}$ (phase-polynomial encoding of the CNOT+T circuit) + action history | $-1$/move; gadget-aware ($+4$ on a completed 7-move Toffoli gadget, $0$ on a completed 3-move CS gadget) | Single binary factor $u^{(s)}\in\{0,1\}^N$ — one rank-1 tensor-decomposition step $\equiv$ one T gate | Sampled AlphaZero-style MCTS (800 sims/move) + symmetrized-axial-attention transformer (learnable symmetrization $\beta$); trained on a 256-core TPU pod, $\sim10^6$ iterations |
| [RL-VQSD info-theoretic analysis (2024)](https://arxiv.org/abs/2404.06174) | Tensor-based ansatz encoding (from the underlying RL-VQSD system) | $R=+\mathcal{R}$ if $C_t(\theta)<\zeta+10^{-5}$ else $-\log(C_t(\theta)-\zeta)$, purity cost $C_t=\mathrm{Tr}(\rho^2)-\mathrm{Tr}(\rho_t^2)$ | Gate configurations added to circuit (same action family as the underlying RL-VQSD system) | DDQN, MLP $5\times1000$; Adam lr $=10^{-4}$, $\epsilon$-decay $0.99995$, replay buffer $15000$; COBYLA optimizes parameters — this paper adds only an "Entanglement-Enhancing" pre-block ablation, not a new agent |

### E3. Benchmark Families Actually Used by the Q1 Set

| Benchmark family | Papers using it | Typical instances (as actually run) | What was reported |
| :--- | :--- | :--- | :--- |
| Fixed-target state synthesis | Truly-PPO, KANQAS | Bell (2q), GHZ (3q); noiseless + depolarizing/measurement noise | Fidelity, success probability, gate count, depth, convergence speed |
| Molecular VQE | KANQAS | H₂ (4q, STO-3G), LiH (4q, STO-3G) | Energy error vs. $1.6\times10^{-3}$ Ha chemical accuracy, CNOT/gate count, depth, parameter count |
| Spin & model Hamiltonians | Truly-PPO (SK Ising), SYK thermal-state prep, GRL (TFIM) | 4q SK spin glass; 4–7q SYK ($N=8$–14 Majoranas); 2–6q TFIM at 3 field regimes | Energy/free-energy error, fidelity, CNOT scaling vs. Trotterization, hardware-noise robustness |
| QML architecture search | MuZero QML arch. search | California housing (8q), Quantum-Fashion-MNIST (8q), two-curves (10q) | Cross-validated MSE/accuracy vs. literature, random, and genetic-search baselines |
| Fault-tolerant arithmetic/oracle circuits | AlphaTensor-Quantum | 26-circuit Amy et al. suite, GF($2^m$) mult, adders, FeMoco Select oracle | T-count/T-depth vs. best-known baseline, gadget decomposition, provable optimality (small cases) |
| State diagonalization & purification | RL-VQSD info-theoretic analysis | 2q Haar-random mixed density matrices (up to 100,000 states) | Purity cost, concurrence bounds, correlation/phase-transition thresholds, gate cost of admissible ansatzes |

**Cross-cutting observations:**
- **Task-diverse, not one lineage.** Where the 3 A* papers form a single genealogical chain on one task (VQE ansatz construction), the 7 Q1 papers cover six materially different problem types with no shared codebase or reward formula between most of them — Q1 breadth trades off against A* depth.
- **Two taxonomy categories are entirely absent from the Q1 set.** None of the 7 target *qubit mapping/routing* or *compilation of an arbitrary fixed unitary* as their primary task — those tasks' Q1 coverage comes from papers outside this repo's 24-paper RL-QAS core list (e.g. the ZX-calculus *Quantum*-journal papers and QARMA in [Appendix 2](#appendix-2-additional-rl-qco-papers-reward-action-space--agent-design)).
- **AlphaTensor-Quantum is the only fault-tolerant-reduction paper**, and the only one of the 7 built on AlphaZero-style MCTS with a transformer rather than DDQN over an MLP/CNN — reflecting that T-count optimization is framed as a single-player tensor-decomposition game, not a sequential circuit-construction MDP like the other six.
- **Agent architecture is more diverse here than in the A* set.** The A* lineage is DDQN end-to-end; the Q1 set spans trust-region PPO (Truly-PPO), DDQN with a KAN in place of an MLP (KANQAS), DDQN with a 3D-CNN (SYK), DDQN with program-synthesized actions (GRL), MuZero (QML search), and AlphaZero-style MCTS (AlphaTensor) — five distinct algorithm/architecture families across seven papers.
- **Reward engineering still converges on the same family for physics tasks.** KANQAS (chemistry), SYK thermal-state prep, and GRL all reuse the Ostaszewski-style $\pm5$-terminal-plus-dense-shaping reward (see [Appendix 3](#appendix-3-a-conference-deep-dive)) even though none of them are in the A* lineage — this reward shape appears to be a de facto standard for any threshold-based physics objective (energy, free energy) regardless of venue.

---

*This file is a derived, RL-focused sub-view of [QAS-APPROACHES.md](QAS-APPROACHES.md) (§2), which itself categorizes the full paper list from [README.md](README.md) by approach. If you add a new RL-based QAS paper to the README, please also file it into the appropriate sub-category here.*
