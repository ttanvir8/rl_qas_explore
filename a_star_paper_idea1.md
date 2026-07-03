# A* Paper Idea 1 — Amortized Quantum Architecture Search: One Policy for a Family of Hamiltonians

**Working title:** *"Train Once, Search Everywhere: Amortized Reinforcement Learning for Quantum Architecture Search across Molecules and Geometries"*

**One-line thesis:** Replace the lineage's one-agent-per-problem convention with a single Hamiltonian-conditioned, graph-based policy trained across a distribution of molecules and geometries, warm-started by tensor networks, that constructs ansatz circuits for unseen Hamiltonians with little or no retraining.

**Source material:** `conference_comparison.md` (the three-paper A* lineage: Ostaszewski et al., NeurIPS 2021 → CRLQAS, ICLR 2024 → TensorRL-QAS, NeurIPS 2025). This document combines several ideas from `next_paper_ideas.md` into one pipeline: the amortized Hamiltonian-conditioned policy, the permutation-equivariant graph state encoding, per-Pauli-term expectation values in the state, the potential-energy-surface evaluation task, and the inherited tensor-network warm start.

---

## 1. The gap, and why it is worth exploring

### 1.1 The gap itself

Every paper in the lineage trains a fresh agent from scratch for every single problem instance. Paper 1 ran three fully separate trainings for three LiH bond distances of the *same molecule*. CRLQAS trains separately per molecule, per qubit count, and per noise profile. TensorRL-QAS, despite its 100-fold efficiency gains, still trains one DDQN per Hamiltonian. Nothing learned on one problem is ever reused on the next — no weights, no discovered structural motifs, no policy. The entire cost of training (hundreds of thousands to millions of environment steps, each one a quantum-circuit simulation plus a classical optimization run) is paid again for every new Hamiltonian, every new geometry, and every recalibration of a device.

### 1.2 Why this gap, out of all the gaps

Four reasons, each grounded in what the comparison document establishes.

**First, it is the bottleneck the whole lineage has been attacking indirectly without ever naming.** Read as a sequence, the three papers are a progression of attempts to make per-instance training cheaper: paper 1's curriculum exists to make per-instance training *converge at all* on harder instances; CRLQAS's Pauli-transfer-matrix and GPU infrastructure makes each per-instance training step cheaper; TensorRL-QAS's warm start makes each per-instance training need far fewer steps. All three accept the per-instance framing and optimize inside it. Amortization attacks the framing itself: if one trained policy serves many instances, the per-instance cost that all three papers fight drops toward the cost of inference. This is the same kind of "change the shape of the problem rather than its constants" move that made TensorRL-QAS's warm start the strongest contribution of the lineage — applied one level higher.

**Second, the intended use case is a family of problems, not a point.** The scientific product a chemist actually needs is a dissociation curve or a reaction pathway — energies across many geometries of the same molecule — and neighboring geometries have Hamiltonians that differ only in their Pauli coefficients, not their structure. The lineage's per-instance framing throws this structure away; paper 1's three independent LiH trainings are the clearest illustration. A method evaluated on whole-curve accuracy per unit of total compute matches deployment reality in a way no paper in the lineage attempts.

**Third, the lineage's own design choices are what block transfer, and the comparison document shows those choices are already under strain.** Both the flat vector (paper 1) and the 3D tensor (CRLQAS, TensorRL-QAS) are fixed-size encodings tied to a specific qubit count, with a flat one-hot action space whose size is a function of that qubit count. This is why papers 1 and 2 had to grow the network (1000, then 2000, then 5000 neurons) as qubit count grew, and why a trained agent cannot even be *evaluated* on a differently-sized problem, let alone transferred. TensorRL-QAS already found that a single fixed-capacity network suffices across 6–20 qubits once the effective search is small — evidence that the learning problem itself does not demand per-size networks, only the rigid input and output encodings do. Removing that rigidity is the enabling step for amortization, and nobody in the lineage has taken it.

**Fourth, the risk profile is right for an A* venue.** The reward formula is treated as a solved primitive across all three papers and two independent research groups; the DDQN-family value-based approach is validated across the lineage and by the BenchRL-QAS study TensorRL-QAS cites. We keep both. The novelty budget is spent on exactly two coupled changes (representation and training regime), each with a clear falsifiable claim and a clean ablation path back to the lineage's per-instance setup. And the contribution type — replacing per-instance search with a conditioned generalist model — is the framing that NeurIPS and ICLR have consistently rewarded across domains (neural combinatorial optimization, amortized inference, learned optimizers).

### 1.3 What failure would look like, honestly

The bet can fail in an informative way: it may turn out that optimal circuit structure is so instance-specific that a conditioned policy generalizes no better than a random-agent baseline on held-out molecules. Even then, the paper's controlled comparison (zero-shot versus fine-tuned versus from-scratch at matched compute, on a shared benchmark family) would be the first quantitative answer to "how transferable is RL-QAS knowledge?" — a question the lineage leaves completely open. The experimental design below is built so that a negative result on zero-shot transfer still leaves a publishable positive result on fine-tuning efficiency, which is a much weaker requirement (the policy only needs to be a better starting point than random initialization).

---

## 2. What we inherit from each paper, and what we change

The lineage's own pattern, documented in the comparison file, is that each successive paper holds most axes fixed and spends its novelty on one or two. We follow that pattern deliberately.

| Axis | Inherited from the lineage | Changed in this paper |
| :--- | :--- | :--- |
| **Task and MDP skeleton** | Paper 1's formulation, unchanged: episodic construction, one structural decision per step, classical optimizer fits angles, episode ends on success threshold or budget exhaustion | The agent is trained on a *distribution* of MDPs (one per Hamiltonian) rather than a single MDP, and the policy is conditioned on which MDP it is in |
| **Reward formula** | The ±5 terminal / dense-shaping formula, verbatim — the one component that survived all three papers and a change of research group; we keep it as the lineage's control variable | Nothing. Deliberately untouched, following the lineage convention |
| **Success threshold** | TensorRL-QAS's fixed chemical-accuracy threshold (no moving-threshold curriculum), which its warm start made viable; paper 1's negative-sum-of-Pauli-coefficients lower bound for the dense-shaping denominator, so no per-instance ground-truth energy is ever needed | Nothing structural; the lower-bound trick matters more here because an amortized method cannot assume oracle energies for unseen instances by construction |
| **Warm start** | TensorRL-QAS's full pipeline: DMRG at low bond dimension, then MPS-to-circuit mapping via Riemannian optimization, used in the *fixed* (state-invisible) variant, which is the cheap one | The warm start becomes per-instance preprocessing inside an amortized loop: it is what makes episodes short enough (roughly 20–40 steps, per TensorRL-QAS's Table 11) that training across many instances per batch is affordable at all |
| **State representation** | CRLQAS's insight that the state should carry structural information beyond gate order (its moment/depth axis), and TensorRL-QAS's warm-start-visibility distinction | **Replaced**: the fixed-size tensor becomes a size-agnostic graph over qubits (Section 3.2), carrying the Hamiltonian's own interaction structure and per-Pauli-term measurement feedback that the lineage's single energy scalar discards |
| **Action space** | The same gate semantics (single-qubit rotations plus CNOT, one gate appended per step) and CRLQAS's illegal-action masking idea | **Re-encoded**: actions are scored on the graph's nodes and edges rather than through a fixed one-hot vector, so the action space scales with the instance automatically and one output head serves all qubit counts (Section 3.3) |
| **Agent** | Value-based learning with the lineage's DDQN training loop: n-step returns, the same epsilon-greedy schedule, replay buffer, target network, separate greedy evaluation phase | **The Q-function becomes a single graph neural network shared across all instances and sizes**, replacing the per-size fully-connected MLPs; the replay buffer mixes transitions from many Hamiltonians (Section 3.4) |
| **Classical angle optimization** | The hybrid split, unchanged: COBYLA refits angles for agent-placed gates; the Riemannian optimizer fits the warm start | Nothing. Angles stay out of the state and out of the action space, as in all three papers |
| **Evaluation protocol** | TensorRL-QAS's baseline discipline: matched-settings reruns of the predecessor, a no-RL control (TN-VQE), multiple seeds | **Extended** with the axis the lineage never had: held-out-instance generalization, measured zero-shot and after fine-tuning, at matched total compute (Section 4) |

Two of CRLQAS's contributions are consciously *not* carried forward, for the same reason TensorRL-QAS dropped them: the moving-threshold curriculum and random halting were solutions to the empty-circuit sparse-reward and depth-pressure problems, and the warm start removes both problems upstream. Our training-distribution setting adds one new reason to avoid the moving threshold: a per-instance adaptive threshold would make the reward scale drift differently on every instance in the batch, destabilizing a shared value function.

---

## 3. The pipeline

### 3.0 Overview

The pipeline has an offline training phase run once, and a deployment phase run per new problem.

**Training (once):** build a distribution of Hamiltonian instances; for each, compute a cheap tensor-network warm start; train a single graph-based DDQN across all instances simultaneously, with episodes sampled round-robin from the distribution.

**Deployment (per new Hamiltonian):** compute the same warm start for the new instance, run the trained policy greedily (zero-shot), and optionally fine-tune briefly on that instance alone if greedy rollouts fall short of chemical accuracy.

### 3.1 Stage 1 — Problem distribution construction

Assemble a training set of Hamiltonian instances spanning the lineage's own benchmark family so all published numbers remain comparable: H2 (2–4 qubits), LiH (4–6 qubits), BeH2 (6 qubits), and H2O (8 qubits), each at a dense sweep of bond geometries (for example 10–20 geometries per molecule rather than the lineage's 1–3), under standard mappings. Hold out entirely: (a) unseen geometries of training molecules, testing interpolation and extrapolation along potential-energy surfaces; (b) unseen molecules at sizes inside the training range; and (c) larger unseen instances (10-qubit H2O, 12-qubit LiH — TensorRL-QAS's headline cases) testing size extrapolation, which is the hardest and most interesting split. Every instance is represented by its Pauli decomposition, which is all the conditioning information the policy ever receives — no quantum-chemistry features beyond what the Hamiltonian itself contains.

### 3.2 Stage 2 — Per-instance warm start (inherited, repurposed)

For each instance, run TensorRL-QAS's preprocessing exactly: DMRG at bond dimension 2, then the Riemannian MPS-to-circuit mapping, used in the fixed variant so the warm start only sets the simulator's initial statevector and never enters the agent's state. This stage is inherited rather than novel, and that is the point: TensorRL-QAS showed it cuts episodes to roughly 20–40 steps at 6–12 qubits. Amortized training must run episodes on many instances, so short episodes are what make the training distribution affordable. The known weakness documented in the comparison file — the warm start's quality is molecule-dependent (poor for 12-qubit LiH) — becomes a feature of the evaluation: it lets us measure whether the amortized policy compensates more effectively on instances where the warm start is weak, since it has seen many instances where more RL work was needed.

### 3.3 Stage 3 — Graph state and graph actions (the representational change)

**State.** The state is a graph whose nodes are qubits. Two edge sets connect them: *Hamiltonian edges*, one per multi-qubit Pauli term, carrying the term's coefficient and Pauli-string type as features; and *circuit edges*, one per two-qubit gate placed so far, carrying gate type and the moment index at which the gate sits (preserving the depth information CRLQAS's tensor introduced). Node features carry each qubit's single-qubit gate history and the single-qubit Pauli terms acting on it. Two global features are appended: the current energy estimate (the lineage's standard scalar) and — new — the vector of per-Pauli-term expectation values from the most recent evaluation, attached to their corresponding Hamiltonian edges and nodes. These per-term values cost nothing extra to obtain (they are already measured to compute the energy) and tell the policy *which interactions* remain unconverged, giving it an instance-specific signal for where to place gates next. Because the Hamiltonian is part of the graph, conditioning on the instance and representing the search state are the same mechanism — there is no separate task-embedding module to design.

**Actions.** A shared message-passing network computes node and edge embeddings. Single-qubit rotation actions are scored by a small head applied to each node embedding (three scores per node, one per rotation axis); CNOT actions are scored by a head applied to each ordered qubit pair's embeddings. This reproduces exactly the lineage's action semantics — the same gate set, the same one-gate-per-step granularity — but the action space now grows with the graph automatically, and the same output heads serve a 4-qubit and a 12-qubit instance. Because scores are computed per node and per pair, the policy is equivariant under qubit relabeling, which is a physical symmetry of the problem the lineage's flat encodings ignore. CRLQAS's illegal-action masking carries over naturally as masks on node and pair scores (same-wire repetition checks read directly off the circuit edges' moment indices, and coupling-map constraints become a mask on which pairs are scorable at all).

### 3.4 Stage 4 — Amortized value-based training (the training-regime change)

Training keeps the lineage's DDQN loop wholesale — n-step returns, the epsilon-greedy schedule, replay buffer, periodic target-network synchronization, a separate greedy testing phase — with two changes forced by the distributional setting. First, each episode is run on an instance sampled from the training distribution, so the replay buffer mixes transitions from many Hamiltonians; the graph representation makes these transitions structurally compatible even across qubit counts. Second, instance sampling is lightly curriculum-weighted toward instances the greedy policy currently fails on (a standard prioritized-task scheme), replacing the function paper 1's moving threshold served — matching difficulty to competence — without any per-instance threshold adaptation, for the reward-scale-stability reason given in Section 2. The reward, threshold, and angle-optimization subroutines are the lineage's, unchanged: the ±5 and dense-shaping formula, fixed chemical-accuracy threshold, the Pauli-coefficient lower bound in the shaping denominator, COBYLA refits after each rotation-gate placement.

### 3.5 Stage 5 — Deployment on a new Hamiltonian

Given an unseen Hamiltonian: compute its warm start (Stage 2, minutes of classical compute), build its graph, and run the trained policy greedily for a small number of rollouts. If the best rollout reaches chemical accuracy, done — the marginal cost of the new instance was inference plus a handful of circuit evaluations, against the lineage's full training run. If not, fine-tune the shared network on this instance alone with the standard per-instance loop; the hypothesis, tested explicitly in the evaluation, is that fine-tuning from the amortized policy reaches chemical accuracy in a small fraction of the environment steps that training from scratch requires.

---

## 4. Evaluation plan

**Primary claims to test, in decreasing order of strength:**

1. **Zero-shot generalization across geometry.** On held-out bond distances of training molecules, the greedy policy reaches chemical accuracy without any fine-tuning. Measured as success rate and circuit size versus per-instance-trained CRLQAS and TensorRL-QAS at those geometries. The headline deliverable is a full dissociation curve (LiH is the natural choice, echoing paper 1) at a stated total compute budget, compared against the cost of the lineage's approach of retraining per point.
2. **Few-shot generalization across molecules.** On held-out molecules, measure environment steps to chemical accuracy when fine-tuning from the amortized policy versus training from scratch with an identical architecture, and versus TensorRL-QAS's published per-instance numbers. The claim worth an A* venue is an order-of-magnitude reduction; anything below roughly 3x would be a negative result for this split.
3. **Size extrapolation.** Train with an 8-qubit ceiling, evaluate on 10-qubit H2O and 12-qubit LiH. This is the split most likely to fail and is reported either way; the graph representation makes it *possible* to run, which no lineage method can even attempt.

**Baselines,** following TensorRL-QAS's discipline: matched-settings reruns of CRLQAS and TensorRL-QAS per instance; TN-VQE (warm start plus plain VQE, no RL) to isolate the warm start's contribution; a random agent on the graph action space; and — the ablation specific to this paper — the identical graph agent trained per-instance from scratch, which isolates amortization's contribution from the representation's contribution. A second ablation removes the per-Pauli-term features (energy scalar only, as in the lineage) to isolate their value. Ten seeds per configuration, restoring paper 1's statistical standard rather than CRLQAS's three.

**Compute accounting is the paper's honesty requirement:** every comparison reports total classical-optimizer function evaluations and simulator calls including the amortized policy's full training cost, amortized over the number of deployment instances. The break-even instance count — how many new problems must be solved before training the generalist is cheaper than repeated per-instance training — is reported explicitly, because it is the number a practitioner actually needs.

---

## 5. Risks and mitigations

- **Zero-shot transfer across molecules may fail.** Mitigated by the claim ladder in Section 4: geometry transfer (chemically the easiest and practically the most valuable, since it is the dissociation-curve use case) is the primary claim; molecule transfer is claimed only at the few-shot level; size extrapolation is exploratory. The paper is publishable on claim 1 plus honest negative results on 2–3.
- **Mixed-instance replay may destabilize value learning** (reward scales and episode lengths differ across instances). The dense-shaping term is already normalized per-instance by construction (it divides by the gap to the instance's own lower bound), which is a quiet advantage of inheriting the lineage's reward; if instability appears anyway, per-instance return normalization is the standard fix and is reported as an implementation finding.
- **Graph network may underperform the MLP even in-distribution.** The per-instance-graph-agent ablation detects this directly. If representation alone costs accuracy at matched training, that finding plus the transfer results still answers the paper's question; but prior evidence (TensorRL-QAS succeeding with a fixed-size MLP across 6–20 qubits) suggests capacity and representation are not the binding constraint once warm starts shrink the search.
- **The warm start's molecule-dependence could confound transfer measurements.** Controlled by reporting the pure warm-start energy (TensorRL-QAS's Appendix J protocol) for every instance, so policy contribution is always measured relative to its actual starting point.

---

## 6. Contribution list, as it would appear in the paper

1. The first amortized formulation of RL-based quantum architecture search: a single policy, conditioned on the target Hamiltonian's own Pauli structure, trained across a distribution of molecules and geometries — replacing the one-agent-per-problem convention of all prior RL-QAS work.
2. A size-agnostic, permutation-equivariant graph encoding of the joint Hamiltonian-and-circuit state, with actions scored on nodes and edges, subsuming the depth information of CRLQAS's tensor and the masking mechanisms it enabled while removing the fixed-size restriction that forced per-instance networks throughout the lineage.
3. Per-Pauli-term measurement feedback in the state at zero additional measurement cost, replacing the lineage's single-scalar energy signal with a spatially-resolved one.
4. A dissociation-curve benchmark and compute-accounting protocol (including amortization break-even analysis) for evaluating QAS methods on problem *families* rather than single points, with the first quantitative measurement of how transferable RL-QAS knowledge is across geometries, molecules, and system sizes.
5. Everything else held fixed, by design: the lineage's reward formula, threshold scheme, warm-start pipeline, hybrid classical-optimizer split, and DDQN training loop are inherited unchanged, so every measured gain is attributable to the two axes this paper actually changes.

---

## 7. Why this fits the lineage's own trajectory

The comparison document's closing reading of each transition is the template. CRLQAS took paper 1's recipe into a harder *regime* (noise) by adding machinery. TensorRL-QAS took the recipe to harder *scale* by adding one upstream stage and removing machinery it obsoleted. The natural third move, and the one this pipeline makes, is to take the recipe from single problems to problem *families* — again by changing what is upstream of the agent (what the state and network are, and what distribution the agent trains on) while leaving the core recipe untouched. Each prior paper's central lever is preserved and load-bearing here: paper 1's reward and hybrid split are the training signal, CRLQAS's structural-state insight and masking survive inside the graph encoding, and TensorRL-QAS's warm start is what makes distributional training affordable. The paper extends all three at once, from the one direction none of them looked.
