# Next A* Paper Ideas — Built on Gaps in the Ostaszewski → CRLQAS → TensorRL-QAS Lineage

Source: `conference_comparison.md`. Ideas are organized by the four RL design dimensions the three papers share, plus a fifth section for angles none of them touch. Each idea states what it changes, why the lineage leaves the gap open, and the experiment that would decide it.

---

## 1. State representation

**Expose per-Pauli-term expectation values instead of one energy scalar.**
All three papers compress the circuit's entire performance into a single number appended to the state. But that energy is computed by measuring individual Pauli terms of the Hamiltonian, so a vector of per-term values is available at zero extra measurement cost — and it tells the agent *which parts* of the Hamiltonian are unconverged, not just that something is. The deciding experiment is whether per-term feedback shortens search on 10–12 qubit molecules; the risk is that the term count grows fast with molecule size and needs pooling or grouping.

**Permutation-equivariant graph encoding for cross-size transfer.**
Both the flat vector and the 3D tensor are fixed-size and qubit-count-specific, which is why papers 1 and 2 had to rescale the network per problem and why no trained agent transfers to anything. Encoding the circuit and Hamiltonian as a graph (qubits as nodes, gates and interaction terms as edges) processed by a graph neural network makes one policy valid across qubit counts. The payoff to demonstrate is training on small molecules and getting nontrivial zero-shot or few-shot performance on larger ones — a capability the lineage structurally cannot have.

**Put classical-optimizer diagnostics into the state.**
The lineage deliberately hides angles from the agent, but it also hides everything about how the optimizer behaved: which angles collapsed to zero or pi (a redundant gate), how slowly it converged (a hard-to-optimize region). These diagnostics are free byproducts of the COBYLA calls already being made. The question is whether this feedback lets the agent avoid circuit structures that are technically expressive but practically untrainable — a failure mode completely invisible in the current design.

**Condition the state on device noise characteristics.**
CRLQAS trains a separate agent per noise profile and keeps noise entirely inside the environment dynamics. Real device calibration drifts daily, so a per-profile agent is stale on arrival. Encoding the calibration snapshot (per-qubit error rates, coupling map) into the state allows a single noise-conditioned policy, evaluated by training across simulated profiles and testing on held-out profiles or across real calibration drift.

---

## 2. Reward engineering

**Constrained or multi-objective RL that makes the accuracy-versus-depth trade explicit.**
Both cross-paper comparisons in the file document that no configuration strictly dominates on energy error and circuit size simultaneously, and the lineage's only depth levers are indirect (random halting) or emergent (warm start). Formulating a depth or CNOT budget as a hard constraint (Lagrangian-constrained RL) or learning a preference-conditioned Pareto front replaces that accident with a dial. The natural evaluation is the shared LiH-6 and H2O-8 benchmarks where the trade-off is already quantified.

**A reward built on error-mitigated energy estimates.**
CRLQAS shows that under hardware noise, minimizing the noisy energy does not reliably track the true noiseless energy — the reward is chasing a corrupted target, and the agent failed to reach chemical accuracy on LiH-4 under realistic noise. Building the reward on mitigated estimates (zero-noise extrapolation or similar) so the curriculum threshold chases the debiased value directly attacks that documented failure. Success is measured by restoring the correlation between training progress and true noiseless error at the noise level where CRLQAS failed.

**Replace the energy threshold with energy variance.**
The variance of the Hamiltonian vanishes exactly at eigenstates, so it gives a success signal that needs no ground-truth energy, no lower bound, and no per-task threshold calibration — note that paper 3 had to pre-run simulated annealing just to pick thresholds for its spin models. The cost is that measuring the squared Hamiltonian is more expensive, so the paper's core question is whether a calibration-free objective justifies the extra measurement budget.

**Meta-learn the curriculum, and run the ablation the lineage skipped.**
Paper 1's moving threshold carries five hand-tuned hyperparameters; paper 3 dropped the curriculum entirely but never tested a fixed threshold with an empty-circuit start, so nobody knows whether the warm start or the curriculum was doing the work. Treating the threshold schedule as an outer-loop learning problem (a bandit over tightening decisions), with the full curriculum-by-initialization ablation grid as the empirical backbone, gives a publishable answer in either direction.

---

## 3. Action space

**Macro-actions: place multi-gate blocks instead of single gates.**
One gate per step means episode length scales with circuit size — the wall the lineage keeps hitting (450 steps per episode at 12 qubits before warm starts). Giving the agent a vocabulary of blocks (a brickwork layer, an entangling motif on a qubit pair, a chemistry-inspired excitation block) as single actions attacks episode length and the expressivity bottleneck at 20 qubits simultaneously, where paper 3 had to patch the gate pool by hand.

**Circuit editing rather than append-only construction.**
No agent in the lineage can undo a mistake; the only pressure against wasted gates is reward shaping or masking. Adding delete, replace, and mid-circuit insert actions turns the search over circuit prefixes into a search over circuit space. This is especially natural on top of a warm start: the agent could prune the warm-start circuit's own imperfect gates, which paper 3's fixed variant cannot touch at all.

**Generate the action pool from the Hamiltonian itself.**
Paper 3 found the standard four-gate pool binding at 20 qubits and fixed it by manually adding XX, YY, and ZZ operators. Instead, derive the candidate operator pool from the target Hamiltonian's own Pauli terms — like ADAPT-VQE's operator pool, but with selection learned by the policy rather than done by gradient screening. The decisive experiment is a head-to-head against ADAPT-VQE using the same pool.

**Make shot allocation an action.**
Under finite shots, every paper fixes the measurement budget per evaluation as a hyperparameter. Letting the agent decide how many shots to spend before committing to a gate turns it into a shot-frugal experimenter, optimizing the quantity that actually costs money on hardware. This pairs naturally with the noisy-reward ideas above and has no counterpart anywhere in the lineage.

---

## 4. Agent architecture

**An amortized, Hamiltonian-conditioned policy: train once, deploy across problems.**
This is arguably the largest untouched gap: every paper trains a fresh DDQN per molecule, per geometry, and per noise setting, and nothing is ever reused. A policy conditioned on an embedding of the Hamiltonian (its Pauli coefficients), trained across a distribution of molecules and bond distances, would turn architecture search from per-instance optimization into learned inference. The evaluation is zero-shot and fine-tuned performance on held-out molecules, and the framing — a generalist model replacing per-instance search — is exactly what NeurIPS and ICLR reward.

**Model-based planning to cut simulator calls.**
The environment is deterministic and every step costs a quantum simulation — the expense papers 2 and 3 attacked with GPU infrastructure and warm starts respectively. A learned surrogate that predicts post-optimization energy from the circuit encoding, combined with tree search over cheap surrogate rollouts before each real evaluation, is a third, complementary lever. The success metric is real simulator calls to chemical accuracy at matched accuracy versus TensorRL-QAS, and it composes with warm starts rather than competing with them.

**Distributional or ensemble Q-learning for the noisy regime.**
With stochastic rewards, a scalar Q-value confounds shot-noise variance with genuine uncertainty about the circuit's quality. Distributional RL or bootstrapped ensembles give calibrated uncertainty, enabling both more stable learning under noise and uncertainty-directed exploration to replace the epsilon-greedy schedule that has been carried unchanged since 2021. The benchmark is the LiH-4 realistic-noise case where CRLQAS fell short of chemical accuracy.

**End the structure/parameter split: let the policy propose angles.**
All three papers invoke a classical optimizer at every rotation-gate placement — up to 1000 COBYLA iterations per step, which is precisely the cost paper 3's 100-fold savings targeted. A policy with a continuous action head that proposes *initial* angles for the optimizer to merely polish (rather than replacing the optimizer outright, which paper 1 argued would slow training) could cut per-step cost by an order of magnitude while keeping the hybrid design's robustness.

---

## 5. Outside the four dimensions

**Search across the whole potential-energy surface, not one geometry.**
Chemistry users need dissociation curves, and the lineage retrains from scratch per bond distance — paper 1 ran three fully separate trainings for three LiH geometries. A method that finds a single ansatz family valid across a curve, or continues training across neighboring geometries, evaluated on whole-curve accuracy per unit of total compute, matches how the tool would actually be used.

**Train the agent on real hardware, not just validate there.**
All three papers train in simulation and only execute finished circuits on devices. On-device training confronts drift, queue latency, and shot budgets head-on; combining warm starts, shot-frugal rewards, and aggressive off-policy sample reuse could make the budget feasible for a small molecule. Even a modest end-to-end on-hardware training result would be a first for this line of work.

**Extend the recipe beyond ground states.**
The entire lineage optimizes one objective family: ground-state energy. Excited states (via state-averaged or orthogonality-constrained objectives) and short-depth compilation of time-evolution circuits are adjacent problems where hand-designed ansatzes are even weaker, and where the reward, curriculum, and warm-start machinery would all need genuine rethinking rather than reuse.

**A theory of when RL-based architecture search works.**
The lineage is entirely empirical: nothing explains why this reward shape succeeds, when curricula are necessary (paper 3's removal was justified by argument, not proof), or how sample complexity scales with qubit count. Even partial results — landscape analysis of the discrete search space, regret analysis of the moving threshold, or a connection to barren plateaus in the induced parameter landscapes — paired with the lineage's own benchmarks would be a novel contribution type for this area.

**Make the warm start adaptive, and take it where matrix product states fail.**
Paper 3's warm start is a one-shot preprocessing step, and its own results show the weakness: the 12-qubit LiH warm start was two orders of magnitude worse than the others, and the fixed bond dimension became a binding constraint at 20 qubits. Two follow-ups suggest themselves: interleave tensor-network compression with RL additions in a loop instead of running them once in sequence, or target 2D lattice systems where matrix product states are inherently poor and a different warm-start family (such as PEPS or hardware-native heuristics) is needed.
