# QAS Approaches: A Categorized Breakdown

This document reorganizes every paper listed in the [Papers](README.md#papers) section of the main [README](README.md) by **algorithmic approach** rather than by year. Many works combine more than one technique (e.g. a surrogate predictor guiding an evolutionary search); in those cases the paper is placed under its **primary/defining** search mechanism, with a cross-reference note (↳) added to the secondary category it also touches.

Each bullet is also tagged with its peer-review/venue status, cross-checked against [PUBLICATION-STATUS.md](PUBLICATION-STATUS.md): *(Q1: venue)* / *(Q2: venue)* / *(A\*: venue)* for CORE/Scimago-ranked venues, *(Other, ...: venue)* for genuinely peer-reviewed but non-tiered venues (e.g. IEEE QCE/QAI, workshops, companion tracks), or *(Preprint only)* where no acceptance was found as of 2026-07-01. Cross-reference (↳) pointer lines are left untagged since the paper's status is already tagged at its primary listing.

## Table of Contents
- [Summary](#summary)
- [1. Evolutionary & Genetic Algorithm-Based QAS](#1-evolutionary--genetic-algorithm-based-qas)
- [2. Reinforcement Learning-Based QAS](#2-reinforcement-learning-based-qas)
- [3. Differentiable / Gradient-Based QAS](#3-differentiable--gradient-based-qas)
- [4. Predictor & Surrogate-Model-Assisted QAS](#4-predictor--surrogate-model-assisted-qas)
- [5. Training-Free / Zero-Cost Proxy-Based QAS](#5-training-free--zero-cost-proxy-based-qas)
- [6. Bayesian Optimization & Tree-Search-Based QAS](#6-bayesian-optimization--tree-search-based-qas)
- [7. Meta-Learning & Transfer-Learning-Based QAS](#7-meta-learning--transfer-learning-based-qas)
- [8. Weight-Sharing / Supernet-Based QAS](#8-weight-sharing--supernet-based-qas)
- [9. Generative Model-Based QAS](#9-generative-model-based-qas)
- [10. Hybrid, Agent-Based & Other Approaches](#10-hybrid-agent-based--other-approaches)
- [Secondary View: By Application Domain](#secondary-view-by-application-domain)

## Summary

| # | Category | Count |
|---|----------|-------|
| 1 | Evolutionary & Genetic Algorithm-Based | 12 |
| 2 | Reinforcement Learning-Based | 24 |
| 3 | Differentiable / Gradient-Based | 10 |
| 4 | Predictor & Surrogate-Model-Assisted | 5 |
| 5 | Training-Free / Zero-Cost Proxy-Based | 4 |
| 6 | Bayesian Optimization & Tree-Search-Based | 3 |
| 7 | Meta-Learning & Transfer-Learning-Based | 1 |
| 8 | Weight-Sharing / Supernet-Based | 1 |
| 9 | Generative Model-Based | 1 |
| 10 | Hybrid, Agent-Based & Other | 7 |

Reinforcement learning is by far the dominant paradigm in the QAS literature, followed by evolutionary/genetic methods and differentiable (gradient-based) search — reflecting the field's roots in classical NAS, adapted to the discrete, noisy, and expensive-to-evaluate setting of quantum circuits.

---

## 1. Evolutionary & Genetic Algorithm-Based QAS
Approaches that evolve a *population* of circuit candidates via mutation, crossover, selection, or other biologically-inspired operators — no gradient through the architecture itself.

- [Automated Design of Quantum Circuits (1998)](https://link.springer.com/chapter/10.1007/3-540-49208-9_8) — the founding work of the field; evolves circuits with genetic programming (mutation, crossover, transposition). *(Other, pre-CORE: QCQC'98, Springer LNCS 1509)*
- [Evolutionary quantum architecture search for parametrized quantum circuits (2022)](https://arxiv.org/abs/2208.11167) — population-based genetic algorithm evolving PQCs for hybrid quantum-classical RL. *(Other, not CORE-tiered: GECCO '22 Companion)*
- [Search space pruning for quantum architecture search (2022)](https://link.springer.com/article/10.1140/epjp/s13360-022-02714-7) — ↳ see also [§5 Training-Free](#5-training-free--zero-cost-proxy-based-qas); listed there since its rotation-based indicator is a proxy, but the underlying search progressively prunes a population/candidate pool. *(Q2: European Physical Journal Plus)*
- [Evolutionary-based quantum architecture search (2022)](https://arxiv.org/abs/2212.00421) — encodes circuits as binary "quantum genes" and evolves them. *(Q2: Quantum Information Processing, retitled on publication)*
- [EQNAS: Evolutionary Quantum Neural Architecture Search for Image Classification (2023)](https://www.sciencedirect.com/science/article/pii/S0893608023005348) — quantum evolutionary algorithm with rotation-gate updates and interference crossover. *(Q1: Neural Networks, Elsevier)*
- [Application of ZX-calculus to quantum architecture search (2024)](https://arxiv.org/abs/2406.01095) — Genetic Programming combined with ZX-calculus rewrite rules for circuit optimization. *(Q1: Quantum Machine Intelligence)*
- [Trainability maximization using estimation of distribution algorithms assisted by surrogate modelling for quantum architecture search (2024)](https://arxiv.org/abs/2407.20091) — an Estimation-of-Distribution Algorithm (a population/probabilistic-model evolutionary method) guided by an online surrogate. ↳ also relevant to [§4 Predictor & Surrogate-Assisted](#4-predictor--surrogate-model-assisted-qas). *(Preprint only)*
- [DeQompile: quantum circuit decompilation using genetic programming for explainable quantum architecture search (2025)](https://arxiv.org/abs/2504.08310) — genetic-programming-based decompiler for explainable QAS. *(Preprint only)*
- [QuProFS: An Evolutionary Training-free Approach to Efficient Quantum Feature Map Search (2025)](https://arxiv.org/abs/2508.07104) — evolutionary search that ranks candidates with fast proxies instead of training. ↳ also relevant to [§5 Training-Free](#5-training-free--zero-cost-proxy-based-qas). *(Preprint only)*
- [Neural Architecture Search for Quantum Autoencoders (2025)](https://arxiv.org/abs/2511.19246) — genetic algorithm evolving gate types, layers, and parameters for autoencoder circuits. *(Other, not CORE-tiered: IEEE QCE 2025)*
- [Noise-Aware Quantum Architecture Search Based on NSGA-II Algorithm (2026)](https://arxiv.org/abs/2601.10965) — variable-depth NSGA-II (multi-objective evolutionary algorithm), hybridized with an ε-greedy Hamiltonian strategy. *(Preprint only)*
- [Probabilistic Design of Parametrized Quantum Circuits through Local Gate Modifications (2026)](https://arxiv.org/abs/2602.12465) — probabilistic, evolution-inspired local gate mutations applied to existing circuits. *(Preprint only)*
- [Q-PhotoNAS: Hybrid Quantum Neural Architecture Search Framework on Photonic Devices (2026)](https://arxiv.org/abs/2605.22097) — genetic search combined with learnable quantum phase encoding for photonic circuits. *(Preprint only)*

## 2. Reinforcement Learning-Based QAS
The largest category: an agent sequentially places gates/actions, trained with policy-gradient, actor-critic, or value-based algorithms (A2C, PPO, A3C, DQN, SAC, TD3, etc.) to maximize a task-specific reward.

- [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715) — A2C/PPO agent builds circuits for GHZ-state preparation from Pauli expectation values. *(Preprint only)*
- [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779) — PPR-DQL reuses learned policies across changing noise environments. *(Preprint only)*
- [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585) — a CNN-based DRL agent learns hardware-specific circuit optimization strategies. *(Preprint only)*
- [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) — feedback-driven curriculum learning balances expressivity and depth. *(A\*: NeurIPS 2021)*
- [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) — a rollback-clipped, trust-region-constrained variant of PPO for QAS. *(Q1: Scientific Reports)*
- [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) — A3C agent for multi-qubit GHZ-state circuit construction. *(Other, not CORE-tiered: QCCC'23 workshop, GECCO 2023 satellite)*
- [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) (CRLQAS) — noise-aware RL with 3D tensor circuit encoding and episode-halting curriculum. *(A\*: ICLR 2024)*
- [KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search (2024)](https://arxiv.org/abs/2406.17630) — RL-based QAS where the Q-network/policy uses KANs instead of MLPs. *(Q1: EPJ Quantum Technology)*
- [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) — RL discovers QML model architectures for supervised tasks under NISQ constraints. *(Other, not CORE-tiered: IEEE QCE 2024)*
- [A quantum information theoretic analysis of reinforcement learning-assisted quantum architecture search (2024)](https://arxiv.org/abs/2404.06174) — analyzes entanglement, initial-condition, and correlation effects in RL-QAS for state diagonalization. *(Q1: Quantum Machine Intelligence)*
- [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) — sample-efficient model-based RL over a reduced, layered search space. *(Q1: Machine Learning: Science and Technology)*
- [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) — RNN-based policy-gradient controller with layer-based search acceleration. *(Q2: Quantum Information Processing)*
- [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) — deep RL minimizing T-count via tensor-decomposition-based circuit optimization. *(Q1: Nature Machine Intelligence)*
- [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) — an MDP-based RL benchmarking framework (QCD) comparing PPO, A2C, TD3, SAC. *(Other, not CORE-tiered: IEEE QCE 2024)*
- [TensorRL-QAS: Reinforcement learning with tensor networks for scalable quantum architecture search (2025)](https://arxiv.org/abs/2505.09371) — RL warm-started from a matrix-product-state approximation, scaling to 20 qubits. *(A\*: NeurIPS 2025)*
- [Reinforcement learning with learned gadgets to tackle hard quantum problems on real hardware (2025)](https://arxiv.org/abs/2411.00230) (GRL) — RL + program synthesis, using composite "gadget" actions. *(Q1: Communications Physics)*
- [Scaling the Automated Discovery of Quantum Circuits via Reinforcement Learning with Gadgets (2025)](https://arxiv.org/abs/2503.11638) — extends gadget-based RL for scalability to more complex circuits/codes. *(Preprint only)*
- [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) — RL + CNNs with a composite entropy/energy reward for thermal-state circuits. *(Q1: Machine Learning: Science and Technology)*
- [QAS-QTNs: Curriculum Reinforcement Learning-Driven Quantum Architecture Search for Quantum Tensor Networks (2025)](https://arxiv.org/abs/2507.12013) — curriculum RL extended to benchmark multiple classical/quantum-enhanced agents. *(Other, not CORE-tiered: IEEE QCE 2025)*
- [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198) — RL agent discovers low-complexity classification circuits. *(Preprint only)*
- [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967) (HyRLQAS) — a single policy jointly learns discrete gate placement and continuous parameter initialization. *(Preprint only)*
- [QASER: Breaking the Depth vs. Accuracy Trade-Off for Quantum Architecture Search (2025)](https://arxiv.org/abs/2511.16272) — RL with an exponential reward jointly optimizing depth, two-qubit gate count, and accuracy. *(Preprint only)*
- [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708) — multiple agents each control a local block of the circuit, reducing per-agent action-space dimensionality. *(Preprint only)*
- [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) — treats the RL replay buffer itself as the main lever (annealed prioritization, curriculum amortization, noiseless-to-noisy transfer). *(Preprint only)*

## 3. Differentiable / Gradient-Based QAS
Circuit structure is relaxed into continuous (often probabilistic/softmax or density-matrix) form so that architecture and parameters can be jointly optimized with gradient descent — the quantum analogue of DARTS.

- [DQAS: Differentiable quantum architecture search (2021)](https://arxiv.org/abs/2010.08561) — the foundational differentiable/probabilistic-programming QAS framework. *(Q1: Quantum Science and Technology)*
- [QuantumDARTS: Differentiable Quantum Architecture Search for Variational Quantum Algorithms (2023)](https://proceedings.mlr.press/v202/wu23v.html) — Gumbel-Softmax relaxation with macro/micro search for scalable sub-circuit patterns. *(A\*: ICML 2023)*
- [Differentiable Quantum Architecture Search for Quantum Reinforcement Learning (2023)](https://arxiv.org/abs/2309.10392) — applies DQAS-style differentiable optimization to quantum deep Q-learning. *(Other, not CORE-tiered: IEEE QCE 2023)*
- [Differentiable Quantum Architecture Search in Asynchronous Quantum Reinforcement Learning (2024)](https://arxiv.org/abs/2407.18202) — joint gradient-based optimization of VQC structure and parameters, with asynchronous RL for throughput. *(Other, not CORE-tiered: IEEE QCE 2024)*
- [SA-DQAS: Self-attention Enhanced Differentiable Quantum Architecture Search (2024)](https://arxiv.org/abs/2406.08882) — adds a self-attention mechanism to capture dependencies between gate placeholders in DQAS. *(Other, not ICML main track: ICML 2024 workshop)*
- [Differentiable Quantum Architecture Search in Quantum-Enhanced Neural Network Parameter Generation (2025)](https://arxiv.org/abs/2505.09653) — differentiable QAS applied within the Quantum-Train (QT) parameter-generation framework. *(Other, not CORE-tiered: IEEE QCE 2025)*
- [RhoDARTS: Differentiable Quantum Architecture Search with Density Matrix Simulations (2025)](https://arxiv.org/abs/2506.03697) — models search as evolution of a quantum mixed state rather than a discrete supernet. *(Preprint only)*
- [Quantum Long Short-term Memory with Differentiable Architecture Search (2025)](https://arxiv.org/abs/2508.14955) — end-to-end joint optimization of QLSTM circuit architecture and parameters. *(Other, not in CORE: IEEE QAI 2025)*
- [Quantum-Based Self-Attention Mechanism for Hardware-Aware Differentiable Quantum Architecture Search (2025)](https://arxiv.org/abs/2512.02476) (QBSA-DQAS) — quantum-native self-attention plus hardware-aware multi-objective differentiable search. ↳ also relevant to [§7 Meta-Learning](#7-meta-learning--transfer-learning-based-qas). *(Q1: Physical Review Applied)*
- [Differentiable Architecture Search for Adversarially Robust Quantum Computer Vision (2026)](https://arxiv.org/abs/2601.18058) — DQAS augmented with a trainable Classical Noise Layer, jointly learning structure and adversarial robustness. *(Q1: Quantum Machine Intelligence)*

## 4. Predictor & Surrogate-Model-Assisted QAS
Methods whose core contribution is a learned model (neural predictor, GNN, self-supervised encoder) that estimates circuit performance cheaply, filtering or ranking candidates before/without full evaluation.

- [Neural predictor based quantum architecture search (2021)](https://arxiv.org/abs/2103.06524) — a neural predictor cuts required evaluations ~10× versus random search. *(Q1: Machine Learning: Science and Technology)*
- [A GNN-based predictor for quantum architecture search (2023)](https://link.springer.com/article/10.1007/s11128-023-03881-x) — represents circuits as DAGs and uses a graph neural network to estimate performance. *(Q2: Quantum Information Processing)*
- [GSQAS: Graph Self-supervised Quantum Architecture Search (2023)](https://arxiv.org/abs/2303.12381) — self-supervised graph representation learning to overcome scarce labeled circuit data. *(Q2: Physica A)*
- [Quantum Architecture Search with Unsupervised Representation Learning (2025)](https://arxiv.org/abs/2401.11576) — decouples unsupervised circuit-representation learning from the downstream search (REINFORCE or Bayesian Optimization). ↳ also relevant to [§6 Bayesian/Tree-Search](#6-bayesian-optimization--tree-search-based-qas) and [§2 RL](#2-reinforcement-learning-based-qas). *(Q1: Quantum journal)*
- [Benchmarking Quantum Architecture Search with Surrogate Assistance (2025)](https://arxiv.org/abs/2506.06762) (SQuASH) — a standardized benchmarking framework built around surrogate models trained on pre-collected circuit data. *(Preprint only)*

## 5. Training-Free / Zero-Cost Proxy-Based QAS
Approaches that avoid parameter training almost entirely, instead scoring candidate circuits with cheap, training-free proxy metrics (path counts, expressibility, landscape statistics, rotation-based indicators).

- [Search space pruning for quantum architecture search (2022)](https://link.springer.com/article/10.1140/epjp/s13360-022-02714-7) — rotation-based indicators progressively eliminate low-potential gates before costly evaluation. *(Q2: European Physical Journal Plus)*
- [Training-Free Quantum Architecture Search (2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29135) — a zero-cost path-based proxy filters candidates, followed by an expressibility-based proxy. *(A\*: AAAI 2024)*
- [Distributed quantum architecture search (2024)](https://arxiv.org/abs/2403.06214) — a two-stage training-free strategy (path-based proxy, then expressibility) adapted to multi-QPU circuits. ↳ also relevant to [§10 Hybrid/Other](#10-hybrid-agent-based--other-approaches) for its distributed-QPU contribution. *(Q1: Physical Review A)*
- [Scalable Quantum Architecture Search via Landscape Analysis (2025)](https://arxiv.org/abs/2505.05380) — predicts circuit learnability from landscape-fluctuation statistics rather than training. *(Preprint only)*
- ↳ [QuProFS (2025)](https://arxiv.org/abs/2508.07104) — see [§1 Evolutionary](#1-evolutionary--genetic-algorithm-based-qas); ranks candidates via fast proxy metrics (expressivity, robustness) inside an evolutionary loop.

## 6. Bayesian Optimization & Tree-Search-Based QAS
Search driven by Bayesian Optimization or Monte Carlo Tree Search (MCTS) over the architecture space, typically paired with a learned surrogate to guide exploration.

- [Self-supervised representation learning for Bayesian quantum architecture search (2025)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.032403) — a GIN predictor pretrained on expressibility builds a latent space explored via Bayesian Optimization. *(Q1: Physical Review A)*
- [Magic-Informed Quantum Architecture Search (2026)](https://arxiv.org/abs/2605.03932) — a GNN predictor of circuit "magic" (nonstabilizerness) steers a Monte Carlo Tree Search. ↳ also relevant to [§4 Predictor-Assisted](#4-predictor--surrogate-model-assisted-qas). *(Preprint only)*
- [Zero-shot Quantum Neural Architecture Search (2026)](https://arxiv.org/abs/2605.27410) (MZeQAS) — a Quantum Neural Tangent Kernel convergence surrogate combined with MCTS for zero-shot candidate scoring. ↳ also relevant to [§5 Training-Free](#5-training-free--zero-cost-proxy-based-qas). *(Preprint only)*

## 7. Meta-Learning & Transfer-Learning-Based QAS
Search accelerated by transferring architecture/parameter knowledge learned on prior tasks to new ones, rather than searching from scratch each time.

- [Quantum Architecture Search with Meta-Learning (2021)](https://arxiv.org/abs/2106.06248) (MetaQAS) — pre-trains meta-architectures and gate parameters across tasks for fast adaptation via fine-tuning. *(Q1: Advanced Quantum Technologies)*
- ↳ [QBSA-DQAS (2025)](https://arxiv.org/abs/2512.02476) — see [§3 Differentiable](#3-differentiable--gradient-based-qas); framed as a meta-learning framework built on differentiable, hardware-aware search.

## 8. Weight-Sharing / Supernet-Based QAS
Decouples architecture search from parameter training by embedding all candidate circuits inside one large "super-circuit" whose weights are shared and inherited by sub-circuits.

- [QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits (2022)](https://arxiv.org/abs/2107.10845) — a SuperCircuit decouples circuit search from parameter training for noise-robust VQAs. *(A\*: HPCA 2022)*
- ↳ [Topology-Driven Quantum Architecture Search Framework (2025)](https://arxiv.org/abs/2502.14265) — see [§10 Hybrid/Other](#10-hybrid-agent-based--other-approaches); its gate-search stage inherits parameters from the topology-search stage, a lightweight form of weight-sharing.

## 9. Generative Model-Based QAS
Uses deep generative modeling (denoising diffusion, etc.) to directly synthesize circuit architectures conditioned on a target task or operation.

- [Quantum circuit synthesis with diffusion models (2024)](https://arxiv.org/abs/2311.02041) — a text-conditioned denoising diffusion model generates quantum operations for gate-based circuits, avoiding exponential-cost dynamics simulation during training. *(Q1: Nature Machine Intelligence)*

## 10. Hybrid, Agent-Based & Other Approaches
Papers that don't fit cleanly into a single search paradigm above — objective-function redesign, hardware demonstrations without a specified algorithm, two-stage/topology-first pipelines, LLM/agent-driven design loops, or property-driven (algorithm-agnostic) circuit selection criteria.

- [Quantum optimization with a novel Gibbs objective function and ansatz architecture search (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023074) — redesigns the QAOA *objective function* (Gibbs objective) alongside an automated ansatz architecture search; the contribution is orthogonal to the search algorithm used. *(Q1: Physical Review Research)*
- [Quantum circuit architecture search for variational quantum algorithms (2022)](https://arxiv.org/abs/2010.10217) — an early, general QAS framework balancing expressivity and noise resilience for VQAs, hardware-validated on IBM devices; algorithmic mechanism not specific to one category above. *(Q1: npj Quantum Information)*
- [Quantum Circuit Architecture Search on a Superconducting Processor (2022)](https://arxiv.org/abs/2201.00934) — a hardware demonstration of QAS-tailored hardware-efficient ansätze on an 8-qubit superconducting processor. *(Q2: Entropy, MDPI)*
- [Topology-Driven Quantum Architecture Search Framework (2025)](https://arxiv.org/abs/2502.14265) — two-stage pipeline: search circuit *topology* first, then fine-tune gate types with parameters inherited from stage one. ↳ also relevant to [§8 Weight-Sharing](#8-weight-sharing--supernet-based-qas). *(Q1: Science China Information Sciences)*
- [Neural Architecture Search Algorithms for Quantum Autoencoders (2025)](https://arxiv.org/abs/2509.15451) — explicitly combines evolutionary search with reinforcement learning to discover generalizable autoencoder architectures. ↳ also relevant to [§1 Evolutionary](#1-evolutionary--genetic-algorithm-based-qas) and [§2 RL](#2-reinforcement-learning-based-qas). *(Q1: IEEE Transactions on Quantum Engineering)*
- [AI Agents for Variational Quantum Circuit Design (2026)](https://arxiv.org/abs/2602.19387) — an autonomous (LLM-style) agent proposes, trains, and refines candidate circuits with minimal human intervention, rather than using a fixed classical search algorithm. *(Preprint only)*
- [How to find expressible and trainable parameterized quantum circuits? (2026)](https://arxiv.org/abs/2603.14451) — a property-based (expressibility, trainability, entanglement, complexity) ansatz-selection framework, agnostic to the underlying search procedure. *(Preprint only)*

---

## Secondary View: By Application Domain

For readers who care more about *what the circuit is for* than *how it is found*, here is a rough grouping by target task (a paper may span more than one domain):

- **State preparation (GHZ, Bell, ground/thermal states):** RL via deep RL (2021), Continual RL (2021), Quantum RL A3C (2023), CRLQAS (2024), GRL gadgets (2025), Scaling RL gadgets (2025), SYK thermal-state RL (2025), QASER (2025), Zero-shot QAS (2026).
- **Variational quantum eigensolver / quantum chemistry:** Reinforcement learning for optimization of VQE circuits (2021), QuantumSupernet-style VQA search (2022), QuantumNAS (2022), TensorRL-QAS (2025), Local Gate Modifications (2026).
- **QML classification / feature maps:** Neural predictor (2021), superconducting-processor QAS (2022), EQNAS (2023), Training-Free QAS (2024), RL-based architecture search for QML (2024), QuProFS (2025), QAS for QML tasks (2025), Q-PhotoNAS (2026).
- **Combinatorial optimization (QAOA/Max-Cut):** Gibbs-objective AAS (2020), DQAS (2021), QuantumDARTS (2023).
- **Circuit compilation / gate-count & T-count reduction:** AlphaTensor (2024), ZX-calculus GP (2024), Challenges for RL in Circuit Design (2024).
- **Quantum reinforcement learning agents (circuits *for* RL, not RL *for* circuits):** Evolutionary QAS for PQCs (2022), Differentiable QAS for QRL (2023), Differentiable QAS in Async QRL (2024).
- **Autoencoders / data compression:** NAS Algorithms for Quantum Autoencoders (2025), NAS for Quantum Autoencoders (2025).
- **Sequence learning (LSTM/NLP/time-series):** Quantum LSTM with Differentiable Architecture Search (2025).
- **Noise- and hardware-awareness (cross-cutting):** QuantumNAS (2022), CRLQAS (2024), Distributed QAS (2024), NA-QAS NSGA-II (2026), QBSA-DQAS (2025), Replay-buffer engineering (2026).
- **Adversarial robustness:** Differentiable Architecture Search for Adversarially Robust QCV (2026).
- **Benchmarking / evaluation frameworks:** SQuASH (2025), Challenges for RL in Quantum Circuit Design (2024).
- **Explainability:** DeQompile (2025).

---

*This file is a derived, methods-oriented view of [README.md](README.md). If you add a paper to the README, please also file it into the appropriate category (or categories) here.*
