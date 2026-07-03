# Conference Comparison — A* Lineage, Paper 1

**Paper:** Ostaszewski, Trenkwalder, Masarczyk, Scerri, Dunjko — *"Reinforcement learning for optimization of variational quantum circuit architectures"* — **NeurIPS 2021**
**File:** `3_a_star_papers/paper1_2021_neurlps_reinforcement-learning-for-optimization-of-variational-quantum-circuit-architectures-Paper-.md`
**Position in lineage:** seed paper of the 3-paper A*-conference chain in this repo — Ostaszewski et al. (NeurIPS 2021) → CRLQAS (ICLR 2024) → TensorRL-QAS (NeurIPS 2025). All three attack the same task family (RL-driven VQE ansatz construction for molecular ground-state energy estimation); CRLQAS and TensorRL-QAS both directly extend this paper's reward/curriculum design (see `RL-APPROACHES.md`, Appendix 3).

---

## 1. Problem being solved

VQE prepares a parameterized state $|\psi(\vec\theta)\rangle = U(\vec\theta)|\psi_0\rangle$ and classically optimizes $\vec\theta$ to minimize $\langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle$, bounding the true ground-state energy $E_{\min}$ from above (the variational principle). The bottleneck the paper targets is **ansatz structure**, not parameter optimization: existing ansatzes are either chemistry-inspired (UCCSD — expressive but very deep) or hardware-inspired (Hardware Efficient, HE — shallow but often insufficiently expressive), and neither is *searched* — both are hand-fixed architectures. The paper casts *structure discovery itself* as a sequential decision problem and solves it with deep RL, with the explicit dual objective of hitting chemical accuracy ($<0.0016$ Ha of $E_{\min}$) while minimizing circuit depth/gate count for NISQ compatibility.

## 2. Methodology, comprehensively

### 2.1 RL formulation of ansatz construction
- **Environment dynamics are deterministic.** An episode starts from an empty circuit; at each step $t$ the agent appends exactly one gate to the end of the current circuit. The next state is fully determined by the previous state + the chosen action — there is no environment stochasticity beyond the agent's own policy (and the classical angle optimizer, which is treated as part of the reward computation, not the transition).
- Every episode starts from the identical empty-circuit condition, deliberately removing initial-state randomness as a confound.

### 2.2 State representation
- The state is an **ordered list of layers**, each layer being a single quantum gate, restricted to CNOT and single-qubit rotations ($R_x, R_y, R_z$) — the NISQ-realizable gate set.
- CNOT gates are encoded as (control-qubit index, target-qubit index); rotation gates are encoded as (qubit index, rotation axis ∈ {X=1, Y=2, Z=3}).
- The circuit has a user-fixed maximum length $L$; positions not yet filled are padded with identity operators (encoded via the "qubit index = max qubit count" sentinel), giving every state a fixed-size representation suitable for a feed-forward network.
- **Rotation angles are deliberately excluded from the state.** Instead, a single scalar — the energy $E_t$ estimated for the current circuit after angle optimization — is appended to the state vector. This is a specific design choice to keep the state representation gate-structure-only and network-friendly, offloading the continuous-parameter dimension entirely to a classical subroutine (see 2.4).
- Actions are one-hot encoded; the action space is $|Q|(|Q|+2) = 3|Q| + 2\binom{|Q|}{2}$ discrete gates (3 rotation axes × $|Q|$ qubits, plus all ordered CNOT pairs), where $|Q|$ is the qubit count.

### 2.3 Reward engineering
Piecewise reward at each step $t$:

$$R = \begin{cases} +5 & \text{if } E_t < \xi \\ -5 & \text{if } t \ge L \text{ and } E_t \ge \xi \\ \max\!\left(\dfrac{E_{t-1}-E_t}{E_{t-1}-E_{\min}},\,-1\right) & \text{otherwise} \end{cases}$$

- $\xi$ is the current energy threshold (initially set to a value corresponding to chemical accuracy, or dynamically adjusted — see 2.5).
- The terminal cases are **sparse/threshold-triggered** ($\pm5$): success is binary (below threshold) or failure (ran out of the length budget $L$ without reaching threshold).
- The intermediate case is **dense reward shaping**: normalized, clipped fractional improvement in energy relative to the best possible improvement from $E_{t-1}$ down to $E_{\min}$, floored at $-1$ so a bad step is bounded rather than catastrophic.
- This reward shape (threshold terminal + dense normalized-improvement shaping) is the template every later paper in the lineage reuses unmodified (CRLQAS, TensorRL-QAS) and that several non-lineage Q1 papers (KANQAS, SYK thermal-state prep, GRL) also adopt — per `RL-APPROACHES.md` this is treated as a "solved primitive" that the paper introduces and that the field converges on.

### 2.4 Decoupling structure search from parameter optimization (hybrid design)
- The agent **never predicts continuous angles**. After every step in which a rotation gate is added, an independent classical optimizer — **COBYLA** or **Rotosolve** — is invoked to fit the rotation angles.
- Two optimization *strategies* are compared: **global** (re-optimize all angles in the circuit after every step; 25 COBYLA/Rotosolve iterations) vs. **local** (re-optimize only the last 5 rotation gates; 5 iterations) — testing whether a cheap, partial re-optimization gives a good-enough energy signal to guide the RL agent.
- Rationale given in the paper: (a) the goal is to isolate RL's efficacy for *structure* search, since parameter optimization is a separately well-studied problem; (b) training an agent to also emit continuous angles would substantially lengthen training.
- This hybrid split is what allows angles to be excluded from the state representation (2.2) — the environment's "hidden" continuous optimization is invisible to the network, only its scalar outcome (the resulting energy) is exposed.

### 2.5 Agent and network architecture
- **Double Deep Q-Network (DDQN)** with $\epsilon$-greedy exploration and Adam optimizer.
- **n-step DDQN**: $n=1$ for the smaller 4-qubit experiments; $n=6$ for the reported results, controlling how far returns are bootstrapped in the TD update.
- Discount factor $\gamma = 0.88$; $\epsilon$ decays multiplicatively by 0.99995/step from 1.0 down to a floor of 0.05.
- Experience replay buffer of 20,000 transitions; target network synced every 500 actions.
- A separate **testing phase** after every training episode runs the greedy policy ($\epsilon=0$) with replay logging disabled, so test-time performance is decoupled from the exploration noise used for data collection.
- Network: fully connected, 5 hidden layers × 1000 neurons (4-qubit experiments) or × 2000 neurons (6-qubit experiments).
- Max circuit length $L = 40$ gates.

### 2.6 Feedback-driven curriculum learning (the paper's central algorithmic contribution)
The naive fixed-threshold reward (set $\xi$ directly to chemical accuracy from episode 1) fails on the larger 6-qubit problem — the paper reports **zero successful circuits** with a fixed threshold there, attributing this to reward sparsity. Hand-designed threshold schedules were also tried and rejected ("did not pass the desired lowest threshold using pre-defined sequences").

Instead, the paper proposes an **autonomous, feedback-driven curriculum** ("moving threshold"):
1. Start with an arbitrary threshold $\xi_1$ (does not need to be tight).
2. Every $G$ episodes, greedily tighten the threshold to the best energy achieved so far ($\xi \to \xi_2$).
3. An **amortization value $\delta$** is added back on top of the threshold whenever (a) a greedy shift just occurred, or (b) the agent fails several consecutive episodes at the current threshold — this prevents the threshold from becoming unreachably tight too fast.
4. After a run of successful episodes, $\delta$ is decayed by a fixed factor, so the added slack shrinks back toward zero over time, eventually converging the threshold back down to the best-found energy.
5. Concretely tuned as: threshold greedily shifted every 2000 episodes, amortization radius $10^{-4}$, radius reduced by $10^{-5}$ after every 50 consecutive successes, initial $\xi = 0.005$.

This makes the task difficulty a function of the *agent's own measured performance* rather than a human-authored schedule — the paper's own framing of the novelty.

### 2.7 Removing the "known ground-truth energy" assumption
Sections 3–4 initially assume $E_{\min}$ is known exactly (an unrealistic assumption, since estimating it is VQE's actual goal). Section 5 relaxes this: instead of $E_{\min}$, the method substitutes an arbitrary **lower bound** $\alpha < E_{\min}$ and sets $\xi$ large enough that $E_{\min}$ is guaranteed to fall in $(\alpha, \alpha+\xi)$. A concrete cheap-to-compute lower bound is used: for $H = \sum_j c_j P_j$ (Pauli decomposition), take $\alpha = -\sum_j |c_j|$. Because the moving-threshold curriculum only ever tightens toward whatever energy the agent can actually reach, the method converges to chemical accuracy around the *true* $E_{\min}$ even though it never had access to it — demonstrated on the 4-qubit LiH case.

## 3. Experimental setup

- **Molecule:** LiH, STO-3G basis, Hamiltonians built with Qiskit.
  - 4-qubit system (parity mapping, symmetry-reduced), 3 bond distances: 1.2 Å, 2.2 Å, 3.4 Å.
  - 6-qubit system (Jordan-Wigner mapping, symmetry-reduced only), bond distance 2.2 Å — chosen to match a prior evolutionary-algorithm baseline (Rattew et al./ref. [23]) for comparison.
- 10 random seeds per RL configuration.
- Baselines: Hardware-Efficient (HE) ansatz (depth tuned per-Hamiltonian to the minimum achieving chemical accuracy) and UCCSD (naive/un-optimized implementation).
- Simulation via Qulacs; no sampling noise modeled (statevector simulation, "sufficiently many measurements" assumed) — an explicitly acknowledged limitation/simplification, not a novel contribution.

## 4. Key results

- **4-qubit LiH:** for bond distances 1.2 Å and 2.2 Å, RL-found circuits were shallower and had fewer gates than HE/UCCSD in *every* trial (10/10). At the harder 3.4 Å geometry, global optimization succeeded in 10/10 trials; local optimization succeeded in only 2–3/10 (COBYLA vs. Rotosolve respectively) — establishing that **global re-optimization outperforms local/partial re-optimization** for guiding the agent.
- **6-qubit LiH (moving-threshold curriculum):** succeeded in 7/10 trials with avg. depth 16 / min depth 11, vs. HE depth 17 and UCCSD depth 377 — roughly **5× shallower** than the only other reported result on this exact 6-qubit benchmark (evolutionary approach, ref. [23], not directly numerically comparable due to differing gate compilations but qualitatively far deeper).
- **Lower-bound-only run (no exact $E_{\min}$):** agent still converges to chemical accuracy despite the injected lower bound being far below the true minimum, validating that the curriculum removes the need for oracle energy knowledge.

## 5. What is novel (the paper's actual contributions)

1. **First application of deep RL (DDQN) to VQE ansatz *structure* search targeting a real quantum-chemistry benchmark (LiH) at state-of-the-art circuit depth**, as distinct from prior QAS-via-RL work that targeted simpler multi-qubit GHZ-state preparation (ref. [21] in the paper) rather than chemistry Hamiltonians.
2. **Feedback-driven ("moving-threshold") curriculum learning** — an autonomous mechanism that adapts task difficulty to the agent's own demonstrated performance (greedy tightening + amortization decay), replacing brittle hand-authored threshold schedules that the authors show fail outright on the harder 6-qubit problem.
3. **A hybrid RL + classical-optimizer architecture** that removes continuous parameters from the state/action space entirely: RL handles discrete structural decisions (which gate, where), while COBYLA/Rotosolve handle continuous angle fitting as an environment-internal subroutine invoked after each rotation-gate placement. This is presented as a deliberate scoping choice (isolate RL's value-add to structure search) rather than a limitation.
4. **Removing the requirement of a known exact ground-state energy** — replacing $E_{\min}$ with an easily computable lower bound (negative sum of absolute Pauli coefficients) and showing the moving-threshold curriculum still converges to chemical accuracy, making the method deployable on genuinely unknown Hamiltonians.
5. **Systematic ablation of the angle-optimization strategy** (local vs. global re-optimization scope, COBYLA vs. Rotosolve) as a first-class hyperparameter of the RL environment, showing global optimization is markedly more reliable, especially as problem difficulty (bond distance, qubit count) increases.
6. **State-of-the-art depth/gate-count results** against both hardware-inspired (HE) and chemistry-inspired (UCCSD) fixed ansatzes on a standard chemistry benchmark, and against a prior evolutionary-algorithm baseline on the harder 6-qubit case (~5× shallower).

## 6. RL design summary

| Axis | Design in this paper |
| :--- | :--- |
| **State representation** | Fixed-length ordered gate-sequence vector (CNOT: control/target qubit; rotation: qubit + axis), identity-padded to max length $L$; continuous angles excluded; scalar current energy estimate $E_t$ appended after each optimizer call |
| **Reward engineering** | Threshold-triggered terminal reward: $+5$ if $E_t<\xi$; $-5$ if episode length cap $L$ reached without success; else dense shaping $\max\!\big(\tfrac{E_{t-1}-E_t}{E_{t-1}-E_{\min}},-1\big)$. $\xi$ is not fixed — it is driven down over training by the feedback-driven "moving threshold" curriculum (greedy tightening every $G$=2000 episodes + amortization radius $\delta$, initial $10^{-4}$, decayed by $10^{-5}$ every 50 successes) |
| **Action space** | Discrete, flat, single-gate insertion per step: $\mathcal{A} = (\{R_x,R_y,R_z\}\times[Q]) \cup (\{\text{CNOT}\}\times[Q]\times[Q])$, size $3|Q|+2\binom{|Q|}{2}$; one-hot encoded; no masking of illegal actions (unlike later CRLQAS, which adds topology-based action masking) |
| **Agent & network architecture** | Double DQN (DDQN), $n$-step returns ($n=1$ or $6$), $\epsilon$-greedy ($1.0\to0.05$, decay $0.99995$/step), Adam optimizer, $\gamma=0.88$, replay buffer 20,000, target-network sync every 500 actions, separate greedy test phase per episode; fully-connected network, 5×1000 (4-qubit) or 5×2000 (6-qubit) hidden units; angle fitting delegated to COBYLA/Rotosolve (external to the network) |

## Input, Process and Output

A plain-language walkthrough of the pipeline, stripped of jargon.

**Preparing the input**
1. Pick a molecule and a geometry — here, LiH at a chosen bond distance (1.2 Å, 2.2 Å, or 3.4 Å).
2. Turn the molecule into a qubit problem — compute its Hamiltonian and map it onto 4 or 6 qubits.
3. Decide the building blocks the agent is allowed to use — single-qubit rotation gates ($R_x, R_y, R_z$) and CNOT gates.
4. Set a size limit — the circuit can never grow past $L=40$ gates.
5. Set up the learning machinery — an empty starting circuit, an initial (loose) success threshold, and the DDQN network.

**What happens with the input (the process, step by step)**
1. Start from a blank circuit.
2. The agent picks one gate and adds it to the end of the circuit.
3. If that gate was a rotation, a classical optimizer (COBYLA or Rotosolve) quickly tunes all the rotation angles in the circuit so far.
4. The circuit's energy is measured — how close it gets to the true ground-state energy.
5. The agent gets a score for that step: a big bonus if it's already accurate enough, a big penalty if it ran out of room without succeeding, otherwise a small score based on whether the energy improved.
6. The bar for "accurate enough" isn't fixed — it automatically gets stricter as the agent keeps succeeding, and loosens temporarily if the agent starts failing (the curriculum).
7. Steps 2–6 repeat, gate by gate, until the circuit either succeeds or hits the length limit — that's one episode.
8. The agent's network is updated using what it just experienced, so its gate choices get better over many thousands of episodes.

**Output**
A short, specific quantum circuit (an exact sequence of gates plus fitted rotation angles) that estimates LiH's ground-state energy to chemical accuracy — while being shallower and using fewer gates than the standard hand-designed circuits (Hardware-Efficient, UCCSD).

## 7. Position relative to the rest of the lineage (context only)

Per `RL-APPROACHES.md` (Appendix 3), this paper is the shared ancestor of the two other A*-conference papers in `3_a_star_papers/`:
- **CRLQAS (ICLR 2024)** keeps this paper's $\pm5$/dense-shaping reward family and DDQN backbone, adding a 3D tensor state encoding, topology-aware action masking, and noise-aware curriculum extensions (random/negative-binomial episode halting) for hardware-error robustness.
- **TensorRL-QAS (NeurIPS 2025)** reuses the reward and DDQN backbone from this lineage unchanged, adding a matrix-product-state warm start to scale to larger qubit counts (up to 20).

Paper 2 (CRLQAS) is covered in full below, followed by Paper 3 (TensorRL-QAS).

---

# Conference Comparison — A* Lineage, Paper 2

**Paper:** Patel, Kundu, Ostaszewski, Bonet-Monroig, Dunjko, Danaci — *"Curriculum reinforcement learning for quantum architecture search under hardware errors"* (CRLQAS) — **ICLR 2024**
**File:** `3_a_star_papers/paper2_2024_iclr_curriculum-reinforcement-learning-for-quantum-architecture-search-under-hardware-errors-.md`
**Position in lineage:** middle link of the 3-paper A*-conference chain — Ostaszewski et al. (NeurIPS 2021) → **CRLQAS (ICLR 2024)** → TensorRL-QAS (NeurIPS 2025). One of the paper's authors (Ostaszewski) is also lead author of paper 1, making this a direct, same-group extension rather than an independent replication.

## 1. Problem being solved

Paper 1 (Ostaszewski et al. 2021) established RL-driven ansatz construction (DDQN + moving-threshold curriculum) as viable for VQE, but did so entirely in **noiseless, statevector simulation**. CRLQAS's stated problem is that the *effects of noise on the architecture-search process itself* — as opposed to noise's well-studied effects on parameter optimization alone — were "poorly understood" prior to this work. Realistic NISQ deployment requires an RL-QAS method that (a) still converges to chemical accuracy when the reward signal (energy) is itself computed from noisy, sampled circuit evaluations, (b) scales the search-space encoding to more qubits/molecules than paper 1's 4–6 qubit LiH cases, and (c) does so without exploding the classical compute cost of simulating noisy circuits at every RL step (the dominant cost driver once noise/Kraus channels enter the simulation).

## 2. Methodology, comprehensively

### 2.1 RL formulation (inherited core)
- Same episodic structure as paper 1: every episode starts from an empty circuit; the agent appends one gate per step from the gate set {CNOT, $R_x, R_y, R_z$}; a classical optimizer fits angles after each rotation-gate placement; the episode ends on a threshold-crossing success or an action-budget cutout.
- Same one-hot action encoding convention (CNOT as control/target qubit pair; rotation as qubit + axis), same $3N + 2\binom{N}{2}$ action-count formula.
- CRLQAS explicitly frames itself as building on top of this shared MDP skeleton, not replacing it — the paper's own related-work section credits (Ostaszewski et al., 2021b) as the DDQN-for-VQE precedent it extends.

### 2.2 State representation — the first architectural change
- Paper 1 encoded the circuit as a **flat, fixed-length ordered gate-sequence vector** (identity-padded to max length $L$).
- CRLQAS replaces this with a **3D binary tensor**: axes are qubit index × circuit moment (depth position) × gate-type. Each 2D "slice" of the tensor is a $((N+3)\times N)$ matrix representing one moment: the first $N$ rows locate CNOT control/target qubits for that moment, and the next 3 rows locate $R_x/R_y/R_z$ placements. Stacking $n_{\text{act}}$ (max actions) such slices gives the full tensor.
- This is a genuine representational upgrade, not just a re-encoding: a flat sequence only records *order*, while the tensor records *which moment (parallel time-slice) each gate occupies*, i.e. it distinguishes gates that can execute concurrently on different qubits from gates that are serially dependent. This directly supports the depth-awareness the paper's episode-halting and gate-efficiency goals need.
- As in paper 1, continuous rotation angles are still excluded from the state; only the scalar post-optimization energy estimate is exposed to the agent — this part of the design is explicitly kept unchanged.

### 2.3 Reward engineering — kept, then extended around the edges
- The core reward formula is **numerically identical** to paper 1's: $+5$ if $C_t<\xi$; $-5$ if the action budget $T_s^e$ is exhausted without success; otherwise dense shaping $\max\!\big(\tfrac{C_{t-1}-C_t}{C_{t-1}-C_{\min}},-1\big)$.
- The **feedback-driven ("moving threshold") curriculum** from paper 1 (Appendix B.1 here is an explicit restatement of paper 1's mechanism, with the same amortization-radius/greedy-shift machinery) is adopted verbatim as the mechanism for adapting $\xi$, including reuse of the lower-bound-Hamiltonian proxy $\mu = -\sum_j|c_j|$ in place of a known $E_{\min}$.
- What's new is not the reward formula but what it has to operate under: in the noisy setting, $C_t$ is now a **noisy, stochastic estimate** rather than a deterministic one (compounded by finite-shot sampling and physical depolarizing/thermal-relaxation noise). The paper explicitly notes this breaks a tacit assumption of the original threshold curriculum — the agent now chases a *noisy* energy floor, and post-hoc evaluation shows minimizing noisy energy does not always track minimizing the true noiseless energy (Fig. 3 in the paper: noisy training curve tracks the threshold, but the noiseless energy behind it does not mirror that trend and can overshoot chemical accuracy despite a looser noisy threshold).

### 2.4 Action space — added illegal-action masking (genuinely new)
- The discrete action set itself (rotation × qubit, CNOT × qubit-pair) is unchanged from paper 1.
- CRLQAS adds an **illegal-actions (IA) mechanism** that paper 1 did not have: before each action selection, a subroutine scans the current 3D tensor and flags actions that would be redundant given unitary-cancellation and re-optimization arguments — (i) appending the same CNOT to the same wire pair as the immediately preceding moment (which would idle/cancel), and (ii) appending the same rotation axis to the same wire as the previous moment (redundant given continuous re-optimization of that axis already happened). Flagged actions have their Q-values forced to $-\infty$ before the arg-max action selection, effectively pruning the search space rather than leaving the agent to learn the redundancy through experience.
- This is a first-class, load-bearing contribution: the paper's own ablation (Appendix E) shows IA changes which circuits are found and interacts with the halting scheme (see 2.6).

### 2.5 Agent & network architecture — same backbone, scaled
- Same **DDQN** backbone, $n$-step returns ($n=1$ for small noiseless systems, up to $n=6$ for 6–8 qubit or heavily-noised systems), $\epsilon$-greedy ($1.0\to0.05$, decay $0.99995$/step), Adam optimizer, $\gamma=0.88$, replay buffer 20,000, target-network sync every 500 actions — all hyperparameter values are carried over unchanged from paper 1.
- Network width is scaled up for larger tensors: 5 fully-connected hidden layers, 1000 neurons (2–4 qubit noiseless/small-noisy), 2000 neurons (4-qubit noisy or 6-qubit noiseless), 5000 neurons (8-qubit $H_2O$) — i.e. the architecture *family* is unchanged, only capacity is scaled with input tensor size.
- Testing-phase protocol (greedy $\epsilon=0$ evaluation decoupled from training-time exploration, replay-logging disabled) is retained from paper 1, but its cadence changes: every 100 episodes here vs. every episode in paper 1.

### 2.6 Random halting — a new curriculum axis orthogonal to the moving threshold
- Paper 1's curriculum only ever adapted the energy threshold $\xi$; the episode-length cap $L$ was a fixed hyperparameter.
- CRLQAS adds **random halting (RH)**: at the start of each episode, the actual action budget $T_s^e$ for that episode is itself sampled from a **negative binomial distribution** over $[0, n_{\text{act}}]$ (parameterized by a failure probability $p$), rather than always being the fixed cap $n_{\text{act}}$.
- Motivation: without RH, the agent has no incentive to prefer short successful circuits over long ones once it can already succeed — RH forces early exposure to short-episode conditions, biasing the policy toward compact circuits at some cost to how quickly the first success is found (an explicit, measured trade-off in the ablation table).

### 2.7 Noise-aware classical optimization — Adam-SPSA (new)
- Paper 1 used COBYLA/Rotosolve for angle-fitting, both of which assume deterministic (or effectively noiseless) cost evaluations.
- CRLQAS introduces a **novel Adam-SPSA optimizer variant**: standard SPSA's stochastic-gradient estimate (two-point finite-difference along a Rademacher-sampled direction) is combined with Adam-style adaptive moment estimation, and run in a **3-stage** schedule with increasing measurement-shot budgets ($N_{\text{shots}}/10 \to N_{\text{shots}} \to 10N_{\text{shots}}$) — critically, with the decaying SPSA hyperparameters evolving **continuously across stages rather than resetting**, which the paper identifies as the specific novel ingredient (prior 3-stage SPSA variants reset decay parameters at each stage transition). This is what makes the reward signal ($C_t$) usable at all under finite-shot and physical noise, and it is reported to roughly **halve** the number of function evaluations needed versus the alternatives tested.
- This optimizer is external to the RL agent, exactly like paper 1's COBYLA/Rotosolve — the hybrid RL-plus-classical-optimizer split from paper 1 is preserved architecturally, only the noisy-regime optimizer itself is new.

### 2.8 Simulation efficiency — Pauli-transfer-matrix (PTM) formalism + GPU (new, infrastructural)
- Simulating noisy circuits with Kraus-operator sum formalism at every RL step (potentially millions of steps across training) is identified as a computational bottleneck unique to this noisy setting — paper 1 never had to face this since it was noiseless-only.
- CRLQAS fuses each gate with its noise channel offline into a precomputed **Pauli-transfer matrix** in the Pauli-Liouville basis, avoiding online recompilation of noise channels at every step, and runs this on GPU with JIT-compiled JAX code — reported up to a **6× speedup** in noisy-circuit RL training. This is an enabling/infrastructural contribution rather than an RL-design one, but it's what makes noisy-regime experiments at this scale tractable at all.

## 3. Experimental setup

- **Molecules:** $H_2$ (2-, 3-, 4-qubit, Jordan-Wigner), LiH (4-qubit parity, 6-qubit Jordan-Wigner), $H_2O$ (8-qubit Jordan-Wigner) — STO-3G basis, OpenFermion/PSI4-generated Hamiltonians. This is a strict superset of paper 1's LiH-only benchmark.
- **Noise profiles:** noiseless; finite-shot-only; IBM Mumbai device noise (median / max / 10× max 1- and 2-qubit depolarizing rates); IBM Ourense noise with coupling-map connectivity constraints. Also a **real-hardware validation** run on IBM Lagos, comparing measured Pauli-string expectation values against the paper's own noisy simulator (paper 1 had no real-hardware component at all).
- **Baselines:** modified qubit-ADAPT-VQE, quantumDARTS, QCAS (Du et al. 2022), EVQE (Rattew et al., evolutionary), and paper 1's own RLQAS results (direct head-to-head on LiH-6).
- 3 seeds per configuration (vs. paper 1's 10 seeds) for the noisy/scaled experiments, reflecting the higher per-trial simulation cost.

## 4. Key results

- **LiH-6, head-to-head against paper 1:** paper 1's RLQAS succeeded in only 7/10 seeds; CRLQAS succeeds in **all** seeds tested, with lower energy error (paper 1: 0.0024 Ha noiseless error vs. CRLQAS's best ablation configuration reaching $8\times10^{-4}$ Ha) — a direct, same-benchmark improvement attributable to the tensor encoding + IA + RH stack.
- **4-qubit $H_2$ vs. QCAS (Du et al. 2022)** under matched Ourense noise/connectivity: CRLQAS finds $-1.136$ Ha vs. QCAS's reported $-0.963$ Ha; noiseless re-evaluation of CRLQAS's noisy-trained circuit reaches energy errors of $8\times10^{-5}$ (with RH, 32 gates) to $1.5\times10^{-5}$ (without RH, 40 gates) — three orders of magnitude better than QCAS's $1.9\times10^{-2}$ at comparable gate count.
- **Ablation (Appendix E) isolates each new component's effect:** IA alone speeds first success but yields larger circuits; RH alone yields shorter circuits at the cost of a later first success; the paper reports these as a real trade-off rather than IA+RH being strictly dominant together.
- **Noise-scaling trend confirmed empirically:** increasing noise level (IBM Mumbai median → max → 10× max) increases required gate count, parameter count, and depth to still hit chemical accuracy — validating a hypothesis from prior literature (Sharma et al. 2020) rather than a claim unique to this paper.
- **Real-hardware corroboration:** IBM Lagos measurements of Pauli-string expectation values for the CRLQAS-found $H_2$-4 circuit align closely with the paper's classical noisy-simulator predictions, supporting the simulator's fidelity as a proxy for real deployment.
- **LiH-4 with realistic hardware noise (Mumbai-median-level depolarizing + shot noise):** the agent *fails* to reach chemical accuracy (minimum noiseless error $3.3\text{–}3.4\times10^{-3}$, still ~2× the $1.6\times10^{-3}$ Ha target) — reported candidly as a current limit of the noisy-regime approach, not glossed over.

## 5. What is novel (the paper's actual contributions, relative to paper 1)

1. **First systematic study of how hardware noise affects the architecture-search process itself** (not just parameter optimization) — the explicit gap paper 1 and prior QAS-RL work left open, since all of it assumed noiseless/all-to-all-connected simulation.
2. **3D tensor-based binary circuit encoding** that adds depth/moment information paper 1's flat sequence lacked, enabling coupling-map/connectivity-aware masking and depth-aware analysis.
3. **Illegal-actions (IA) masking** — a combinatorial search-space pruning mechanism absent in paper 1, which relied purely on reward shaping to disincentivize redundant gates.
4. **Random halting (RH)** — a second, independent curriculum axis (stochastic episode-length cap) layered on top of paper 1's threshold curriculum, specifically targeting gate/depth efficiency rather than success rate.
5. **Novel 3-stage Adam-SPSA optimizer** with continuously-evolving (non-reset) decay hyperparameters — a new noise-robust angle-optimization method the paper claims is a previously undiscovered variant, addressing a problem (stochastic/noisy cost evaluation) paper 1's COBYLA/Rotosolve never had to solve.
6. **PTM-formalism + GPU/JAX simulation infrastructure** giving a 6× training speedup for noisy circuits — an enabling contribution with no analog in paper 1's noiseless-only setup.
7. **Real-hardware validation (IBM Lagos)** comparing simulator predictions to measured device behavior — paper 1 never touched real hardware.
8. **Scale-up of the benchmark set**: from LiH-only (paper 1) to $H_2$ (2/3/4-qubit) + LiH (4/6-qubit) + $H_2O$ (8-qubit), across noiseless, finite-shot, and multiple device-calibrated noise profiles — a substantially broader empirical footprint than paper 1's three LiH bond-distance points.
9. **Direct, same-benchmark improvement over paper 1**: 10/10 vs. 7/10 seed success rate on LiH-6, and lower energy error, demonstrating the noise-focused additions (tensor state + IA + RH) also help even in the noiseless case paper 1 studied.

What CRLQAS explicitly does *not* change: the reward formula (identical $\pm5$/dense-shaping expression), the feedback-driven moving-threshold curriculum mechanism (reused wholesale, including the lower-bound-Hamiltonian trick), the DDQN agent family, the hybrid RL-plus-classical-optimizer split, and the discrete single-gate-per-step action granularity. Per `RL-APPROACHES.md` (Appendix 3), this is the pattern across the whole A* lineage: reward engineering is treated as a solved primitive, and each successive paper spends its novelty on a different axis (curriculum → noise/depth-awareness here; tensor-network warm-start in TensorRL-QAS).

## 6. RL design summary

| Axis | Paper 1 (Ostaszewski et al. 2021) | **CRLQAS (this paper, 2024)** |
| :--- | :--- | :--- |
| **State representation** | Flat, fixed-length ordered gate-sequence vector (identity-padded to $L$); angles excluded; scalar energy appended | **3D binary tensor** (qubit × moment × gate-type), $((N+3)\times N)$ slice per moment stacked to depth $n_{\text{act}}$; angles still excluded, scalar energy still appended; noise/coupling-map info lives in the environment dynamics, not encoded directly in the state |
| **Reward engineering** | $+5$ if $E_t<\xi$; $-5$ on budget exhaustion; else dense shaping $\max\!\big(\tfrac{E_{t-1}-E_t}{E_{t-1}-E_{\min}},-1\big)$; $\xi$ driven by moving-threshold curriculum (greedy shift every $G$ episodes + amortization decay) | **Identical formula and curriculum mechanism**, reused verbatim; the only change is that $C_t$ is now a noisy/stochastic estimate (shot noise + physical noise), which the paper shows can decouple noisy-training progress from true noiseless-energy progress |
| **Action space** | Discrete, flat, single-gate insertion, $3|Q|+2\binom{|Q|}{2}$ actions, no masking | Same discrete action set and insertion mechanics, **plus illegal-action (IA) masking** (Q-values forced to $-\infty$ for redundant CNOT/rotation repeats on the same wire(s) as the previous moment) and, separately, **coupling-map-constrained placement** under IBM Ourense connectivity |
| **Agent & network architecture** | DDQN, $n$-step ($n=1,6$), $\epsilon$-greedy, Adam, $\gamma=0.88$, replay 20,000, target sync/500 actions; FC 5×1000/2000; angle fitting via COBYLA/Rotosolve (global vs. local ablation) | **Same DDQN hyperparameter family** (identical $\gamma$, $\epsilon$-schedule, replay size, target-sync cadence), FC network scaled to 5×1000/2000/5000 for larger tensors; **plus random-halting (RH)** curriculum (negative-binomial episode-length sampling) and angle fitting via the paper's **novel 3-stage Adam-SPSA** (noisy case) or COBYLA (noiseless case, as before) |

## 7. Input, Process and Output

**Preparing the input**
1. Pick a molecule, geometry, and qubit mapping — $H_2$ (2/3/4q, Jordan-Wigner), LiH (4q parity or 6q Jordan-Wigner), or $H_2O$ (8q Jordan-Wigner).
2. Choose a noise setting: ideal simulation, finite-shot-only, or a calibrated IBM device noise profile (Mumbai at three severities, or Ourense with its qubit-connectivity map).
3. Fix the gate palette (CNOT, $R_x, R_y, R_z$), the max-action budget $n_{\text{act}}$, and initialize the 3D-tensor-shaped empty circuit and DDQN network (sized to the tensor).
4. Set up the two curricula: the moving energy threshold $\xi$ (as in paper 1) and, if using RH, the negative-binomial episode-length sampler.

**What happens with the input (process, step by step)**
1. Start from an empty circuit (empty tensor).
2. Before choosing an action, mask out illegal actions (redundant same-wire repeats from the previous moment) by setting their Q-values to $-\infty$; if running under a coupling-map constraint, also mask disallowed-connectivity CNOTs.
3. The agent picks a gate from the remaining legal actions and appends it, updating the tensor's next open moment/slot.
4. If the appended gate was a rotation, the classical optimizer refits circuit angles — COBYLA in the noiseless case, the paper's 3-stage Adam-SPSA in the noisy/finite-shot case — using the noisy or noiseless cost function as configured.
5. The (possibly noisy) energy is computed and scored against the current threshold $\xi$: big bonus if already below threshold, big penalty if the episode's action budget (itself possibly randomly shortened by RH) is exhausted, otherwise dense improvement-based shaping.
6. The threshold $\xi$ and, separately, the episode-length distribution (RH), keep adapting to the agent's demonstrated performance — success tightens $\xi$; the RH sampler independently biases the agent toward shorter successful circuits over time.
7. Steps 2–6 repeat until the episode ends; the DDQN is updated from the experience, using PTM-precomputed noisy-gate simulation on GPU to keep each step's noisy-circuit evaluation fast.
8. After training, the resulting policy's circuits are evaluated both under the noisy conditions they were trained on and, separately, re-evaluated noiselessly to check whether noisy-regime training actually converged toward the true (noiseless) ground-state energy.

**Output**
A gate-efficient, depth-aware parameterized ansatz circuit that reaches (or approaches) chemical accuracy for the target molecule under a specified noise profile — validated not only in simulation but, for one case, against real IBM hardware measurements — while using fewer CNOTs/gates and succeeding more reliably (10/10 vs. 7/10 seeds on the shared LiH-6 benchmark) than paper 1's noiseless-only method.

## 8. Position relative to the rest of the lineage (context only)

Per `RL-APPROACHES.md` (Appendix 3): CRLQAS is the middle link connecting Ostaszewski et al. (NeurIPS 2021) to TensorRL-QAS (NeurIPS 2025). It inherits paper 1's reward formula, moving-threshold curriculum, and DDQN backbone wholesale, and contributes the 3D tensor encoding, illegal-action masking, random-halting curriculum, noise-aware Adam-SPSA optimizer, and PTM/GPU simulation infrastructure — all in service of extending the same task (VQE ansatz construction) from an idealized noiseless setting to realistic NISQ hardware-error conditions. TensorRL-QAS in turn reuses CRLQAS's (and, transitively, paper 1's) reward and DDQN backbone unchanged, adding a matrix-product-state warm start to scale further in qubit count (up to 20).

---

# Comparative Improvement Analysis — Paper 1 (Ostaszewski et al., NeurIPS 2021) → Paper 2 (CRLQAS, ICLR 2024)

This section isolates *only* the deltas: for every axis of the method, what specifically changed between the two papers, the mechanism of the change, and whether it constitutes a strict improvement, a trade-off, or an extension into a harder problem regime that paper 1 never attempted. Numbers are pulled directly from both papers' bodies/appendices (see §§1–8 above for full source context) rather than re-derived.

## A. Scope of the improvement: what problem got harder

Before comparing mechanisms, it's worth being precise about what CRLQAS actually took on that paper 1 didn't attempt at all — this is the frame that makes every downstream mechanism change legible as "necessary to survive the new regime" rather than "improvement for its own sake":

| Dimension | Paper 1 | CRLQAS | Nature of the change |
| :--- | :--- | :--- | :--- |
| Simulation fidelity | Ideal statevector only ("sufficiently many measurements" assumed, sampling noise explicitly ignored) | Noiseless **and** finite-shot **and** physical device noise (IBM Mumbai depolarizing at 3 severities, IBM Ourense with real coupling-map constraints) | Strict superset — CRLQAS reproduces paper 1's exact regime as one of its settings, then adds three harder ones |
| Real hardware | None — simulation only, explicitly out of scope | IBM Lagos measurement run, cross-checked against the noisy simulator | New capability, not present at all in paper 1 |
| Molecule/qubit coverage | LiH only, 4-qubit (3 bond distances) and 6-qubit (1 bond distance) | $H_2$ (2/3/4-qubit), LiH (4/6-qubit), $H_2O$ (8-qubit) | Superset — same LiH-6 point is reused as a direct head-to-head benchmark |
| Qubit connectivity | Implicit all-to-all (no coupling-map constraint modeled) | All-to-all **and** IBM Ourense's real restricted coupling map | New constraint class added |
| Cost-function determinism | Deterministic given fixed angles (statevector expectation value) | Stochastic — same angles can yield different $C_t$ across evaluations (shot noise, physical noise) | Fundamentally changes what the RL environment's "reward" is: no longer a pure function of state, but a noisy estimator of one |

This is the reason almost every mechanism below had to change or be added: paper 1's design implicitly assumed a deterministic, noiseless cost oracle, and CRLQAS had to either harden that design against stochasticity or add new machinery the deterministic case never needed.

## B. Component-by-component delta

### B.1 State representation: flat vector → 3D tensor
- **Paper 1:** state = single flat vector of length $L$ (max gates), each slot a (qubit, axis)-or-(control, target) pair, identity-padded. This encodes gate *order* only — there is no representation of which gates can execute in parallel versus which are serially dependent.
- **CRLQAS:** state = 3D binary tensor, axes (qubit × moment × gate-type), i.e. $((N+3)\times N)$ slices stacked to depth $n_{\text{act}}$.
- **Mechanism of the improvement:** the tensor makes *circuit depth* a first-class, directly-readable feature of the state rather than something the network would have to infer from gate order and count. This is a prerequisite for two downstream capabilities paper 1 didn't have: (a) depth-aware illegal-action detection (checking "was this exact gate placed on this exact wire in the immediately preceding *moment*" requires knowing moment structure, not just sequence position), and (b) coupling-map-aware CNOT masking (checking whether a proposed CNOT's control/target pair is physically connected at the hardware level).
- **Verdict:** strict representational upgrade — a superset of the information in paper 1's encoding (order is still recoverable from the tensor), used to unlock features paper 1's format structurally could not support.

### B.2 Reward formula: unchanged — but its *operating conditions* changed underneath it
- The formula itself — $+5$/$-5$/dense-shaping — is copied verbatim, digit for digit, from paper 1's Eq. 2 to CRLQAS's Eq. 2. This is **not** an area of improvement; it's explicitly a "solved primitive" the authors chose not to touch (both papers share an author, Ostaszewski, so this is a deliberate reuse decision, not an oversight).
- The actual change is contextual: in paper 1, $C_t$ (called $E_t$ there) is a deterministic number given fixed angles. In CRLQAS, $C_t$ is a **noisy estimator** of that same quantity. The paper is explicit that this breaks an implicit assumption: Fig. 3 of the CRLQAS paper shows the noisy training curve tightly tracking the moving threshold, while the *noiseless* energy computed from the same parameters does **not** track the threshold curve at all — it can overshoot chemical accuracy while the noisy threshold is still loose, or vice versa. Paper 1 never had to confront this decoupling because its threshold and its "ground truth" were the same deterministic number.
- **Verdict:** no improvement to the reward function itself; the improvement is entirely in the supporting infrastructure (see B.5, B.6) built to make that unchanged reward usable when its input becomes noisy.

### B.3 Curriculum: one axis (moving threshold) → two independent axes (moving threshold + random halting)
- **Paper 1** has exactly one curriculum mechanism: the feedback-driven moving energy threshold $\xi$ (greedy tightening every $G$ episodes + amortization decay). Episode length cap $L$ is a **fixed** hyperparameter (40 gates, for both the 4- and 6-qubit systems).
- **CRLQAS** keeps the moving-threshold mechanism completely intact (same greedy-shift + amortization-decay logic, and — notably — the *exact same numeric hyperparameters* for the 6-/8-qubit case: $G=2000$ episodes, amortization radius $\delta=10^{-4}$ decayed by $10^{-5}$ every 50 successes, initial $\xi_1=0.005$). It then adds a second, independent axis: **random halting (RH)**, where the per-episode action budget $T_s^e$ is itself sampled from a negative binomial distribution over $[0, n_{\text{act}}]$, rather than always equal to the fixed cap.
- **What this specifically fixes that paper 1 couldn't:** in paper 1, once the agent could reliably succeed under the (increasingly strict) energy threshold, there was no separate pressure pushing it toward *shorter* successful circuits — the only depth control was the fixed budget $L$ and the reward's implicit preference for fewer steps via the intermediate dense-shaping term. RH adds an explicit, tunable, curriculum-level lever for depth/gate-count that operates independently of energy accuracy.
- **A subtlety not present in paper 1 at all:** CRLQAS also introduces a *size-dependent* threshold-shift cadence — $G=500$ episodes for the smaller 2-/3-/4-qubit systems vs. $G=2000$ for 6-/8-qubit systems — whereas paper 1 used a single $G=2000$ regardless of system size (its only two system sizes, 4- and 6-qubit, both effectively used the same schedule). This is a small but concrete refinement: the curriculum's pace is now itself qubit-count-aware.
- **Trade-off exposed (not present as a concern in paper 1):** the paper's own ablation (Appendix E) shows RH is not free — it delays the *first* successful episode substantially (e.g. LiH-6, IA+RH: 641,593 actions to first success vs. IA+wo-RH: 356,604) in exchange for shorter final circuits. Paper 1 never had to reason about this trade-off since it had no analogous second axis.

### B.4 Action space: unmasked → illegal-action masking + coupling-map masking
- **Paper 1:** the full $3|Q|+2\binom{|Q|}{2}$ action set is available at every step with no restriction; the agent must learn via reward signal alone that redundant gates (e.g., re-applying the same rotation axis to a wire immediately after already optimizing it) are wasteful.
- **CRLQAS:** adds a hard **illegal-actions (IA)** filter applied *before* the arg-max action selection — two heuristics (same-CNOT-on-same-wires-as-previous-moment; same-rotation-axis-on-same-wire-as-previous-moment) get their Q-values forced to $-\infty$, so the agent cannot select them at all, rather than merely being discouraged from doing so. Separately, under the IBM Ourense experiments, CNOT placements that violate the physical coupling map are masked in the same way.
- **Mechanism difference from paper 1's implicit approach:** paper 1 relies entirely on the DDQN's learned Q-values converging to disprefer redundant/invalid actions through experience — this costs training samples and never *guarantees* the agent won't occasionally pick a wasteful action during exploration. CRLQAS's masking is a hard combinatorial constraint enforced at inference/decision time, independent of how well-trained the Q-network is.
- **Measured effect (Appendix E ablation, not present in paper 1):** IA alone (without RH) shortens the time to first success relative to no-IA (e.g., LiH-6 noiseless: 356,604 actions-to-first-success with IA vs. no exact wo-IA/wo-RH number reported as it reached N.A. in some rows, but the wo-IA/RH row shows worse noiseless error 0.0037 vs. IA/RH's 0.0012) — but the paper is explicit that IA alone tends to produce *larger* circuits, i.e. it speeds up finding *a* successful circuit without necessarily minimizing depth, which is exactly why RH is added as a separate, complementary lever rather than IA being expected to solve gate-efficiency alone.
- **Verdict:** a genuinely new search-space-pruning mechanism with no analog in paper 1, motivated by the larger action/state spaces (up to 8 qubits vs. paper 1's max 6) making unconstrained exploration proportionally more expensive.

### B.5 Classical angle-optimization subroutine: COBYLA/Rotosolve → COBYLA (noiseless) or novel 3-stage Adam-SPSA (noisy)
- **Paper 1** uses COBYLA or Rotosolve exclusively, compared head-to-head via a local (5 gates, 5 iterations) vs. global (all gates, 25 iterations) re-optimization ablation. Both optimizers assume deterministic cost evaluations — reasonable, since paper 1 never evaluates a noisy circuit.
- **CRLQAS**, in its noiseless experiments, keeps **COBYLA with a global strategy** (paper 1's own conclusion — "global outperforms local" — is adopted as the default rather than re-litigated), but scales the iteration budget up to **1000 iterations** (vs. paper 1's 25 for 4-qubit or 200 for its 6-qubit case) to handle the larger 6-/8-qubit landscapes.
- For the **noisy** case, where COBYLA/Rotosolve's implicit determinism assumption breaks down, CRLQAS introduces a genuinely new optimizer: a **3-stage Adam-SPSA** variant, combining SPSA's stochastic-gradient two-point estimate with Adam-style adaptive moment terms, where — the paper's specifically-claimed novel ingredient — the decaying hyperparameters evolve *continuously* across the three shot-budget stages ($N_{\text{shots}}/10 \to N_{\text{shots}} \to 10N_{\text{shots}}$) instead of resetting at each stage transition (as prior 3-stage SPSA variants did). Reported effect: roughly **halves** the number of function evaluations needed relative to the reset variants tested.
- **Verdict:** for the noiseless regime, this is an incremental scale-up of paper 1's own chosen configuration (more iterations, same optimizer, same global-vs-local conclusion inherited unchanged); for the noisy regime, it's an entirely new component paper 1's design never needed because it never had a noisy cost function to optimize against.

### B.6 Simulation infrastructure: CPU/statevector → PTM formalism + GPU/JAX
- **Paper 1:** Qulacs on CPU, statevector simulation only (no noise channels to simulate at all).
- **CRLQAS:** for noiseless runs, still Qulacs/CPU (unchanged). For noisy runs — a regime paper 1 never enters — gate+noise-channel pairs are precompiled offline into **Pauli-transfer matrices** in the Pauli-Liouville basis, executed on GPU with JIT-compiled JAX, avoiding online Kraus-operator recompilation at every one of the potentially millions of RL steps across training. Reported speedup: **up to 6×** relative to a naive noisy-simulation approach.
- **Verdict:** purely enabling/infrastructural — this has no counterpart in paper 1 because paper 1's noiseless-only scope never created the computational bottleneck (noise-channel simulation cost) that motivated it. Without this, the noisy experiments in CRLQAS would likely have been computationally infeasible at the qubit counts/episode counts reported.

### B.7 Agent and network architecture: same family, scaled capacity, refined granularity
- **Held constant, exactly:** DDQN backbone, Adam optimizer, discount factor $\gamma=0.88$, $\epsilon$-greedy schedule ($1.0\to0.05$, decay factor $0.99995$/step), replay buffer size 20,000, target-network sync interval (every 500 actions), fully-connected 5-hidden-layer network family. None of these numbers changed between the two papers.
- **Scaled up, not redesigned:** network width grows from paper 1's two tiers (1000 neurons/4-qubit, 2000/6-qubit) to CRLQAS's three tiers (1000/2-3-qubit, 2000/4-qubit-noisy-or-6-qubit-noiseless, 5000/8-qubit) — a direct, proportional extension of the same scaling logic paper 1 established, applied to the new, larger $H_2O$-8 case.
- **$n$-step granularity increased:** paper 1 uses only $n=1$ (small/4-qubit) or $n=6$ (its reported/harder results). CRLQAS adds an intermediate $n=5$ specifically for the 2-/3-/4-qubit *noisy* Mumbai-noise experiments — a size **and** noise-level-dependent choice paper 1's binary 1-or-6 scheme didn't need to make since it had no noisy configurations to tune for.
- **Testing-phase cadence changed (a trade-off, not a pure improvement):** paper 1 runs its greedy ($\epsilon=0$) evaluation phase after **every** training episode. CRLQAS reduces this to **every 100 episodes**. This is very likely a necessary compute-cost concession — greedy evaluation under noisy/finite-shot simulation is far more expensive per call than paper 1's noiseless evaluation — but it does mean CRLQAS's training curves have coarser-grained ground-truth checkpoints than paper 1's, which is a real (if minor) regression in monitoring granularity that the paper does not explicitly flag as a limitation.
- **Verdict:** the agent itself is deliberately *not* where CRLQAS innovates — every core DDQN hyperparameter is inherited unchanged, and the only changes are proportional scaling (network width, $n$-step count) plus one cost-driven concession (test cadence).

### B.8 Gate/depth budget: fixed constant → qubit-count-scaled
- **Paper 1:** a single fixed maximum circuit length, $L=40$ gates, applied identically to both the 4-qubit and 6-qubit experiments.
- **CRLQAS:** the max-gate cap now scales with system size: 40 gates (4-qubit noiseless), 60 gates (4-qubit noisy, noise-level-dependent), 70 gates (6-qubit noiseless), 250 gates (8-qubit $H_2O$ noiseless).
- **Verdict:** a direct, concrete acknowledgment that a one-size-fits-all budget (reasonable when paper 1's largest system was 6 qubits) does not extend to an 8-qubit molecule with a much larger Hilbert space and expected ansatz depth — a refinement forced by the benchmark-scale increase in §A above.

### B.9 Statistical rigor: seed count reduced (a trade-off, not an improvement)
- **Paper 1:** 10 random seeds per configuration, uniformly, across all experiments.
- **CRLQAS:** 3 seeds per configuration for the noisy/larger-scale experiments (noiseless small-system experiments are not specified as differing, but the headline noisy results use 3 seeds).
- **Verdict:** an explicit cost trade-off, not a methodological improvement — noisy-circuit simulation is markedly more expensive per trial (motivating B.6's GPU/PTM investment in the first place), and reporting 3 seeds instead of 10 is the direct consequence. This makes CRLQAS's noisy-regime success-rate claims statistically thinner than paper 1's noiseless ones, which is worth flagging explicitly rather than treating every change in this lineage as strictly "better."

## C. Quantitative before/after: the shared LiH-6 benchmark

This is the one benchmark point run in *both* papers with a directly comparable setup (6-qubit LiH, Jordan-Wigner, bond distance 2.2 Å, noiseless), making it the cleanest apples-to-apples improvement measurement in the lineage:

| Metric | Paper 1 (RL global COBYLA) | CRLQAS (IA, wo-RH) | CRLQAS (IA, RH) | CRLQAS (wo-IA, RH) | CRLQAS (wo-IA, wo-RH) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Noiseless energy error | 0.0024 Ha | **0.0008 Ha** | 0.0012 Ha | 0.0037 Ha | 0.0016 Ha |
| Avg. depth | 16 | 27 | 30 | 23 | 31 |
| Min depth | 11 | — | — | — | — |
| Avg. # gates | 35 | 56 | 59 | 45 | 62 |
| Min # gates | 27 | — | — | — | — |
| Successful seeds | 7/10 | — (3 seeds) | — (3 seeds) | — (3 seeds) | — (3 seeds) |

Reading this table honestly (not just citing the favorable cells): CRLQAS's best configuration (IA, without RH) achieves a **3× lower energy error** (0.0008 vs. 0.0024 Ha) than paper 1 on the identical benchmark, and — per the narrative in §4 above — succeeds in all tested seeds vs. paper 1's 7/10. But this comes at the cost of **substantially deeper, larger circuits** (depth 27 vs. 16, gates 56 vs. 35) — CRLQAS's IA-only configuration is *not* depth/gate-count-competitive with paper 1's result on this exact benchmark; it trades circuit compactness for lower energy error and higher success rate. Only when RH is added does CRLQAS start pulling depth back down (23 with wo-IA/RH), but that configuration has worse energy error (0.0037 Ha) than paper 1. **No single CRLQAS ablation configuration in this table strictly dominates paper 1 on both energy error and depth/gate-count simultaneously** — the improvement is real but is a Pareto shift (better accuracy/reliability at a depth cost, or comparable depth at an accuracy cost), not a strict, all-metrics win. This nuance is not visible from the headline "10/10 vs. 7/10 seeds, 3× lower error" framing used earlier in this document (§4, §5) and is worth stating plainly.

## D. Net assessment

Distilling the above into a single verdict per axis:

| Axis | Verdict |
| :--- | :--- |
| State representation (tensor vs. flat vector) | **Strict improvement** — strictly more information-preserving, unlocks depth/connectivity-aware features paper 1's format structurally couldn't support |
| Reward formula | **Unchanged** — deliberately reused, not improved; correctly treated as a solved primitive by the shared author |
| Curriculum (moving threshold) | **Unchanged mechanism**, reused with identical hyperparameters for the matched 6-/8-qubit case; only the threshold-shift *cadence* gained a size-dependent variant (new) |
| Curriculum (random halting) | **New capability**, not an improvement to an existing one — addresses a gap (no depth-preference mechanism beyond the fixed budget) paper 1's design left open, at a measured cost to first-success speed |
| Action space (illegal-action + coupling-map masking) | **New capability** — hard constraint enforcement paper 1 achieved only implicitly (and imperfectly) through reward shaping |
| Angle optimizer (noiseless case) | **Incremental scale-up** of paper 1's own concluded-best configuration (global COBYLA), more iterations, same method |
| Angle optimizer (noisy case) | **New capability**, unavailable in paper 1's scope entirely (no noisy cost function existed to optimize against) |
| Simulation infrastructure | **New, enabling capability** — a bottleneck paper 1 never encountered because it never simulated noise |
| Agent/network core hyperparameters | **Unchanged** — same DDQN family, same $\gamma$, $\epsilon$-schedule, replay size, sync interval |
| Network capacity & $n$-step granularity | **Proportional scale-up**, following paper 1's own established pattern to a new (8-qubit) tier |
| Testing-phase cadence | **Regression in monitoring granularity** (every episode → every 100 episodes), almost certainly a compute-cost concession, not flagged as a limitation in the paper |
| Gate/depth budget | **Refinement** — moved from one fixed constant to a qubit-count-scaled schedule, forced by the larger benchmark set |
| Seed count (statistical rigor) | **Regression** (10 → 3 seeds for noisy/scaled runs) — a real, if reasonable, cost trade-off, not an improvement |
| Benchmark breadth / real-hardware validation | **Strict expansion** — LiH-only/simulation-only → 3 molecule families × multiple noise regimes × 1 real-device validation run |

**Overall reading:** CRLQAS is best understood not as "a better version of paper 1's method" across the board, but as *the same core RL recipe (reward, curriculum-threshold logic, DDQN family) hardened and extended to survive a strictly harder operating regime* — noisy, finite-shot, connectivity-constrained, larger-qubit-count VQE — that paper 1 explicitly scoped itself out of. Every mechanism that changed did so either because the new regime demanded it (Adam-SPSA, PTM/GPU simulation, illegal-action/coupling-map masking, tensor state) or because the larger benchmark set demanded proportional scaling (network width, gate budget, $n$-step). The one place CRLQAS is a direct, same-benchmark, same-condition improvement over paper 1 (LiH-6 noiseless) shows a genuine but *not unconditional* gain: better accuracy and reliability, at the cost of circuit compactness in its best-accuracy configuration — a nuance the paper's own framing (and the summary earlier in this document) tends to compress into an unqualified "CRLQAS wins."

---

# Conference Comparison — A* Lineage, Paper 3

**Paper:** Kundu, Mangini — *"TensorRL-QAS: Reinforcement learning with tensor networks for improved quantum architecture search"* — **NeurIPS 2025**
**File:** `3_a_star_papers/paper3_2025_neurlps_tensorrl-qas-reinforcement-learning-wit.md`
**Position in lineage:** final link of the 3-paper A*-conference chain — Ostaszewski et al. (NeurIPS 2021) → CRLQAS (ICLR 2024) → **TensorRL-QAS (NeurIPS 2025)**. Not a same-author continuation like paper 1→2 (different research group, Helsinki/Algorithmiq rather than Leiden/Delft), but the paper explicitly adopts CRLQAS/paper 1's reward formula and DDQN backbone as its starting point and benchmarks directly against a CRLQAS rerun.

## 1. Problem being solved

Paper 1 established RL-driven ansatz construction as viable for noiseless VQE; CRLQAS extended it to survive hardware noise. Neither addressed **scalability**: per the paper's own framing, RL-QAS methods had only been demonstrated up to 8 qubits in noiseless simulation and 4 qubits under realistic noise, because (a) the discrete action space grows combinatorially with qubit count, (b) episodes must get longer to build deeper circuits for larger systems, and (c) each RL step requires a full quantum-circuit (statevector or noisy-channel) simulation whose cost grows rapidly with depth and qubit count — starting every episode from an *empty* circuit (as in papers 1–2) means the agent spends enormous numbers of expensive simulator queries rediscovering basic structure from scratch. TensorRL-QAS's stated goal is to remove this scalability bottleneck by **warm-starting** the RL search from a classically-obtained, physically-meaningful approximate solution rather than from nothing, cutting both the effective search space and the per-episode simulation/optimization cost.

## 2. Methodology, comprehensively

### 2.1 Three-stage pipeline (the paper's core idea)
Given a target Hamiltonian $H$:
1. **DMRG** (density matrix renormalization group) finds a matrix product state (MPS) $|\psi\rangle$ approximating $H$'s ground state, at a chosen maximum bond dimension $\chi$ (a hyperparameter trading expressivity for classical tractability; $\chi=2$ is the default across most experiments).
2. **MPS → PQC mapping via Riemannian optimization**: find a brickwork circuit of 2-qubit unitaries $U=\prod_k U_k$, $U_k\in U(4)$, that maximizes overlap $L=|\langle\psi|U|0\rangle|$ with the MPS. This is a genuinely new optimization step absent from papers 1–2 (detailed in 2.2 below).
3. **RL-based QAS** (DDQN, inherited from the paper 1/CRLQAS lineage) continues adding gates on top of the warm-started circuit, $U|0\rangle \to VU|0\rangle$, to further reduce the energy toward chemical accuracy.
If DMRG alone already reaches the target accuracy, steps 2–3 are unnecessary — the RL/QAS machinery is explicitly framed as kicking in only where classical tensor methods become intractable (see 2.7/Appendix K).

### 2.2 MPS-to-circuit mapping via Riemannian optimization (new, precedes the RL stage entirely)
- The 2-qubit unitaries $\{U_k\}$ are optimized on the complex Stiefel manifold (the space of unitary matrices) so that gradient updates never leave the manifold — i.e., every intermediate iterate remains a valid unitary automatically, rather than requiring re-unitarization after each Euclidean gradient step.
- Concretely: compute Euclidean gradients of the overlap loss via automatic differentiation on the tensor-network contraction, project them to Riemannian gradients (Eq. 7: $\nabla_{RL}(U_k)=\partial L/\partial U_k - U_k[\partial L/\partial U_k]^\dagger U_k$), run an Adam-style update in that tangent space, and move to the next point on the manifold via a **Cayley-transform retraction** (shown empirically superior to an SVD-based retraction in prior work the authors cite).
- This whole procedure is a classical, RL-free optimization pass — a new component with no analog anywhere in papers 1–2 (which never needed to convert a non-circuit representation into a circuit before RL even begins).
- The chosen circuit topology (brickwork of 2-qubit gates) is deliberately hardware-native (no ancillas, natively 2-qubit interactions), chosen for compatibility with the RL stage's own eventual hardware deployment.

### 2.3 Two TensorRL variants + one ablation baseline (StructureRL)
- **TensorRL (trainable TN-init):** both the *structure* and the *parameters* of the warm-start circuit are encoded into the RL state via the binary tensor encoding; the agent's optimizer can further retrain the warm-start's own parameters alongside newly added gates. State-tensor size: $(D_{MPS}+D)\times N\times(N+N_{1\text{-qubit}})$, i.e. the warm-start occupies the first $D_{MPS}$ "moment" slices, exactly like any other part of the circuit the agent can see and (in principle) reason about.
- **TensorRL (fixed TN-init):** the warm-start circuit is **not** fed into the RL state at all — it is used only to set the *initial statevector* the simulator starts from. The binary-tensor state shrinks to $D\times N\times(N+N_{1\text{-qubit}})$ (the $D_{MPS}$ block is dropped entirely). The paper identifies three concrete mechanisms by which this reduces cost: (i) smaller state → faster statevector simulation per step; (ii) smaller state → smaller/faster neural-network forward passes; (iii) the warm-start's parameters are non-trainable, so there are strictly fewer trainable parameters for the classical optimizer to fit, directly cutting the number of energy evaluations needed per step.
- **StructureRL (ablation, not a method of independent interest):** identical warm-start *structure* to the TensorRL variants, but with all parameters zeroed out and left fully trainable (both warm-start and agent-added gates). This isolates whether TensorRL's advantage comes from the MPS's actual *information content* (the fitted parameters) versus merely from a good *structural template* — with sufficient training, StructureRL converges to match TensorRL (trainable), confirming both ultimately optimize the same parameter space, but TensorRL (trainable) gets there faster because it starts from an already-informed point rather than zero.

### 2.4 State representation
- Inherits the binary-tensor encoding convention from the paper 1/CRLQAS lineage (the paper cites [35, 37] directly for the encoding scheme) rather than inventing a new one — the innovation here is entirely about *what occupies* the tensor (warm-start-visible vs. warm-start-invisible), not the tensor's format.
- Continuous rotation angles for agent-added gates remain excluded from the state, exactly as in papers 1–2 — the hybrid RL/classical-optimizer split is preserved unchanged for the RL-agent's own decisions.
- A qualitatively new state-design axis this lineage hadn't faced: whether a *non-empty, non-trivial initial condition* is exposed to the agent at all (trainable variant) or deliberately hidden and handled purely at the simulator/statevector level (fixed variant) — papers 1–2 always started from a truly empty circuit, so this distinction didn't exist for them.

### 2.5 Reward engineering
- The paper explicitly reuses the identical reward formula from Ostaszewski et al. [21] and CRLQAS [35] (its own Eq. 18): $+5$ if $C_t<\xi$; $-5$ if the episode's step budget $T_s^e$ is exhausted without success; otherwise dense shaping on the normalized energy improvement, floored at $-1$ — no changes to the reward's functional form.
- A genuine **departure from the lineage**, however: TensorRL-QAS does **not** use the feedback-driven "moving-threshold" curriculum that both paper 1 and CRLQAS treat as central. The threshold $\xi$ is set as a **fixed constant** per task — $1.6\times10^{-3}$ Ha for quantum-chemistry problems (the literal chemical-accuracy bound), $10^{-3}$ for the Heisenberg model, $10^{-2}$ for TFIM (the latter two calibrated by running SA-QAS on those Hamiltonians first). There is no greedy threshold-tightening, no amortization radius, no episode-count-triggered schedule anywhere in the paper's own method description (Appendix E.1's detailed curriculum walkthrough is presented purely as a description of the **CRLQAS baseline being compared against**, not of TensorRL-QAS itself).
- The implicit argument: once the search starts from a warm start already close to the target energy, the sparse-reward failure mode that motivated the moving-threshold curriculum in paper 1 (zero successes at a fixed threshold on the harder 6-qubit case) no longer arises, so the curriculum machinery becomes unnecessary overhead rather than a requirement. This is a real methodological simplification relative to both prior lineage papers, not an oversight — but it is worth flagging as a place where the lineage's "reward engineering is a solved primitive" framing (per `RL-APPROACHES.md` Appendix 3) is more precisely "the terminal/dense reward formula is a solved primitive; the curriculum wrapped around it is not always needed."

### 2.6 Action space
- Same discrete gate pool as papers 1–2: $\{R_x, R_y, R_z, \text{CNOT}\}$, one gate appended per step.
- No illegal-action masking or coupling-map masking is described as a TensorRL-QAS contribution (unlike CRLQAS, which added both) — the paper's efficiency gains come entirely from the warm start, not from action-space pruning.
- A load-bearing finding surfaces only at extreme scale (Appendix H.6, 20-qubit TFIM): with only the 4-gate pool, the fixed-TN-init variant essentially plateaus (only 1% relative improvement over the raw MPS state) because of a combined bottleneck of the fixed bond dimension ($\chi=2$) and the limited action space. Expanding the pool to add two-qubit $\{XX,YY,ZZ\}$ operators on top of the existing gates recovers meaningful improvement (1% → 9%). This is presented as a genuine limitation acknowledged by the authors, not a solved problem: action-space expressivity becomes the binding constraint precisely where the warm-start advantage should matter most (very large qubit counts).

### 2.7 Agent and network architecture
- **DDQN** (unchanged algorithm choice from the whole lineage), **$n=5$-step** rollouts, $\gamma=0.88$, $\epsilon$-greedy ($1\to0.05$, geometric decay $0.99995$/step), target network synced every 500 steps, replay buffer 20,000, batch size 1000, Adam optimizer at $\eta=3\times10^{-4}$.
- **Network capacity held constant across all qubit counts (6–20 qubits)**: a single 5-hidden-layer, 1000-neuron-wide MLP for every experiment — a deliberate departure from papers 1–2, both of which scaled network width with qubit count (paper 1: 1000/2000; CRLQAS: 1000/2000/5000). The paper does not flag this as a limitation, but it is a notable design choice: the warm start apparently reduces the *effective* complexity of the remaining search enough that a fixed-capacity network suffices even at 20 qubits, whereas the empty-circuit-start lineage needed to keep growing the network with problem size.
- **Explicit justification for DDQN over alternatives (Appendix D)**: rather than re-running a full algorithm comparison, the paper leans on the separate BenchRL-QAS benchmark study (which tested A2C/A3C/PPO/TPPO/DQN-family/DDQN across VQE/VQSD/VQC/GHZ tasks) and adopts its finding that DDQN gives the best accuracy/depth trade-off specifically for VQE-style ansatz construction, while explicitly noting DDQN is *not* claimed optimal for other QAS problem classes (e.g. Hamiltonian diagonalization or QML classification, where policy-gradient/actor-critic methods reportedly do better).
- **Classical optimization is now two separate, decoupled subroutines** rather than one: (i) the lineage's familiar COBYLA angle-refit (1000 iterations) after each agent-placed rotation gate, applied only to the RL-agent's own additions; and (ii) the Riemannian-Adam/Cayley-transform optimizer (2.2) that fits the warm-start circuit's parameters *before* the RL stage ever begins, and — only in the trainable-TN-init variant — continues to be refined jointly with new gates during RL. This second optimizer has no counterpart in papers 1–2.

## 3. Experimental setup

- **Molecules/systems:** 6-BeH$_2$, 8-H$_2$O, 10-H$_2$O, 12-LiH (main chemistry benchmark, STO-3G except 10-H$_2$O at 6-31G); 8-CH$_2$O, 10-CH$_2$O (Appendix H.1, showing the 4-gate set's expressivity limits); 5-qubit Heisenberg and 6-to-20-qubit transverse-field Ising model (Appendix H.5/H.6, demonstrating generality beyond chemistry). This is a strict superset in scale of both prior lineage papers (paper 1: LiH only, ≤6 qubits; CRLQAS: H$_2$/LiH/H$_2$O, ≤8 qubits).
- **Baselines:** CRLQAS (both as originally published and rerun under matched agent/environment settings for fairness), vanilla RL-QAS (paper 1-style), RA-QAS (random agent), training-free QAS, SA-QAS (simulated annealing), quantumDARTS, and a new **TN-VQE** baseline (MPS warm start followed by a standard hardware-efficient-ansatz VQE with no RL at all) introduced specifically to isolate whether RL contributes anything beyond the TN initialization alone.
- **Noise:** noiseless for the main chemistry sweep; 8-qubit H$_2$O under amplified depolarizing noise (single-qubit error ×10, two-qubit error ×5 relative to current IBM devices) plus finite-shot noise ($10^4$ shots); real-hardware validation by executing noiseless-trained circuits on **IBMQ-Brisbane** (a different device from CRLQAS's IBM Lagos validation).
- **DMRG bond dimension:** $\chi=2$ default, ablated up to $\chi=32$ (Appendix K) to study the classical-tractability/RL-benefit trade-off.
- Reported over 5 random neural-network initializations per configuration (error bars = std. dev. or max/min range across these).

## 4. Key results

- **Efficiency:** up to **100-fold** reduction in classical-optimizer function evaluations and up to **98%** faster per-episode execution time versus CRLQAS (fixed-TN-init variant), attributable directly to the warm start rather than to any new simulation infrastructure.
- **Compactness/accuracy:** across 6–12-qubit chemistry systems, TensorRL variants achieve **5–13× fewer CNOTs and shallower depth** than baselines while matching or beating chemical accuracy; e.g., 12-LiH: 30–37 CNOTs vs. 68 (CRLQAS rerun), 321 (vanilla RL), 390 (SA-QAS).
- **Reliability:** 100% chemical-accuracy success across all tested seeds for TensorRL, vs. 70% (vanilla RL) / 60% (RA-QAS) / 0% (SA-QAS) at 10-qubit H$_2$O.
- **Scaling:** success probability up to **50%** at 10 qubits vs. <1% for every non-tensor baseline tested; the paper explicitly frames this as "unprecedentedly efficient circuit discovery" at this scale.
- **Noisy regime (8-qubit H$_2$O, amplified depolarizing):** TensorRL (fixed) reaches 100% success vs. CRLQAS's 30% at comparable error, with roughly half the CNOT count — the paper attributes this partly to TensorRL's non-trainable warm-start statevector potentially insulating the agent's learning signal from noise corruption.
- **Scale ceiling:** demonstrated up to **20-qubit** TFIM — far beyond paper 1 (6 qubits) and CRLQAS (8 qubits) — though the 20-qubit case only improves meaningfully once the action pool is expanded beyond the lineage's standard 4-gate set (1% → 9% relative improvement), an explicitly reported limitation rather than an unqualified win.
- **TN-VQE (no-RL) baseline consistently underperforms every RL variant**, confirming RL contributes real, non-redundant value on top of the tensor-network initialization rather than the initialization alone being sufficient.
- **Non-monotonic warm-start depth (Appendix I):** increasing the number of TN layers used for the warm-start from 1→3 *hurts* RL performance for 8-/10-qubit H$_2$O (more RL failures at 3 layers), attributed to reduced learnability rather than overfitting — a genuinely counter-intuitive finding not present anywhere in papers 1–2's simpler empty-circuit setup.
- **DMRG-bond-dimension scaling (Appendix K):** as $\chi$ increases toward the point where DMRG alone reaches chemical accuracy, the RL stage's marginal contribution shrinks to near-zero — precisely characterizing TensorRL-QAS's operating regime as "where classical tensor methods are still intractable," not a universal replacement for classical methods.

## 5. What is novel (the paper's actual contributions)

1. **Tensor-network (DMRG + Riemannian-optimization) warm start for RL-QAS** — replacing the lineage's universal "start from an empty circuit" convention with a physically-motivated, classically-computed non-trivial starting point, the paper's central and namesake contribution.
2. **A genuinely new classical sub-procedure** (MPS-to-PQC mapping via Cayley-transform Riemannian-Adam on the Stiefel manifold) that runs entirely before the RL stage begins — a component with no counterpart in papers 1–2's hybrid RL/classical-optimizer split.
3. **Two clearly differentiated integration modes** (trainable vs. fixed TN-init) plus a dedicated ablation (StructureRL) that isolates whether the benefit comes from the MPS's actual fitted information versus merely a good structural template — a controlled comparison absent from prior lineage papers.
4. **Order-of-magnitude efficiency gains** (10–100× fewer optimizer evaluations, up to 98% faster episodes) achieved via a fundamentally different lever than CRLQAS's noise-focused infrastructure (PTM/GPU simulation) — this paper's speedup comes from *starting closer to the answer*, not from simulating faster.
5. **Scaling the lineage's demonstrated qubit count from 8 (CRLQAS) to 20** (TFIM), including candidly identifying and empirically probing the action-space/bond-dimension bottleneck that emerges at that scale, rather than presenting scaling as unconditionally solved.
6. **Simplifying away the moving-threshold curriculum** that both paper 1 and CRLQAS treat as essential, using a fixed chemical-accuracy threshold instead — an implicit claim that a sufficiently good warm start removes the sparse-reward problem the curriculum was originally invented to solve.
7. **A new no-RL control baseline (TN-VQE)** cleanly separating "value of the TN initialization alone" from "value of RL search on top of it," plus a DMRG-bond-dimension-vs-RL-benefit scaling study (Appendix K) that precisely delineates when RL-based refinement stops being worthwhile.
8. **Broadened generality beyond quantum chemistry**: demonstrated on Heisenberg and TFIM spin models (Appendix H.5/H.6), and on molecules (CH$_2$O) large enough to expose the 4-gate set's expressivity limits — a wider benchmark net than either prior lineage paper attempted.
9. **New real-hardware validation point** (IBMQ-Brisbane) independent of CRLQAS's own IBM Lagos validation, plus the largest reported noisy RL-QAS simulation to date (8 qubits, per the paper's own claim relative to prior work capped at 4 qubits noisy).

What TensorRL-QAS explicitly does *not* change: the terminal $\pm5$/dense-shaping reward formula (reused verbatim from papers 1–2), the DDQN algorithm choice, the discrete single-gate-per-step action granularity, and the general hybrid RL-plus-classical-optimizer split for the agent's own gate placements. Per the pattern already established in this lineage (`RL-APPROACHES.md`, Appendix 3), each successive paper spends its novelty on a different axis — curriculum/noise-awareness for CRLQAS, warm-start/scalability for TensorRL-QAS — while leaving the reward formula and core agent algorithm untouched.

## 6. RL design summary

| Axis | Design in this paper |
| :--- | :--- |
| **State representation** | Binary tensor encoding inherited from the paper 1/CRLQAS lineage, in one of two configurations: $(D_{MPS}+D)\times N\times(N+N_{1\text{-qubit}})$ if the warm-start circuit's structure+parameters are exposed to the agent (*trainable TN-init*), or $D\times N\times(N+N_{1\text{-qubit}})$ if the warm start is invisible to the agent and used only to set the initial statevector (*fixed TN-init*); continuous angles for agent-added gates remain excluded from the state, as in papers 1–2 |
| **Reward engineering** | Identical $\pm5$/dense-shaping formula reused verbatim from papers 1–2 ($+5$ if $C_t<\xi$; $-5$ on timeout; else $\max\!\big(\tfrac{C_{t-1}-C_t}{C_{t-1}-C_{min}},-1\big)$); **departs from the lineage by dropping the moving-threshold curriculum entirely** — $\xi$ is a fixed constant per task ($1.6\times10^{-3}$ Ha chemistry, $10^{-3}$ Heisenberg, $10^{-2}$ TFIM), since the warm start is argued to remove the sparse-reward failure mode the curriculum was built to solve |
| **Action space** | Discrete $\{R_x,R_y,R_z,\text{CNOT}\}$, single-gate insertion per step, same mechanics as papers 1–2; no illegal-action or coupling-map masking claimed as a contribution; ablated to add two-qubit $\{XX,YY,ZZ\}$ operators specifically to overcome an expressivity bottleneck identified at 20-qubit scale |
| **Agent & network architecture** | DDQN, $n=5$-step returns, $\gamma=0.88$, $\epsilon$-greedy ($1.0\to0.05$, decay $0.99995$/step), Adam ($\eta=3\times10^{-4}$), replay buffer 20,000, batch size 1000, target sync every 500 steps; **single fixed-capacity 5×1000 MLP used unchanged across all qubit counts (6–20)**, unlike papers 1–2's qubit-count-dependent network scaling; angle-fitting is now two decoupled subroutines — COBYLA (1000 iterations) for the agent's own gate placements, plus a new Cayley-transform Riemannian-Adam optimizer (Stiefel manifold) that fits the warm-start circuit's parameters before/alongside RL, with no counterpart in papers 1–2 |

## 7. Input, Process and Output

**Preparing the input**
1. Pick a target Hamiltonian — a molecule/geometry/basis (e.g. LiH, STO-3G, 12 qubits) or a spin model (Heisenberg, TFIM) — and a qubit mapping.
2. Choose a DMRG bond dimension $\chi$ (default 2) that governs how expressive, and how classically expensive, the initial approximation is allowed to be.
3. Decide which TensorRL variant to run: *trainable* (agent sees and can retrain the warm start) or *fixed* (agent only inherits the resulting statevector, cheaper per step).
4. Fix the gate palette (RX, RY, RZ, CNOT, optionally extended with two-qubit XX/YY/ZZ for very large systems), the DDQN hyperparameters, and the noise setting (noiseless, amplified depolarizing + shot noise, or none).

**What happens with the input (the process, step by step)**
1. Run DMRG to obtain an MPS approximation of the target ground state at the chosen bond dimension.
2. Convert that MPS into an actual quantum circuit (a brickwork of 2-qubit gates) by optimizing gate parameters via Riemannian gradient descent on the Stiefel manifold (Cayley-transform retraction), so the mapped circuit's output state closely overlaps the MPS.
3. Hand this warm-start circuit to the RL stage — either embedded directly in the state tensor (trainable) or only as the simulator's starting statevector (fixed).
4. From there, the DDQN agent appends gates one at a time exactly as in papers 1–2: pick a gate, refit rotation angles via COBYLA, measure the resulting energy, and score the step against the (now fixed, non-curriculum) chemical-accuracy threshold.
5. Repeat until the episode succeeds (energy below threshold) or the step budget runs out.
6. Periodically, evaluate the resulting circuits under whatever noise model is configured, and — for select circuits — execute them on real IBM hardware to check that simulated performance transfers.

**Output**
A compact ansatz circuit — a short sequence of gates sitting on top of a physically-motivated tensor-network warm start — that reaches (or surpasses) chemical accuracy for the target Hamiltonian, using far fewer CNOTs/shallower depth and far fewer classical-optimizer evaluations than starting the same RL search from an empty circuit, and scaling to qubit counts (up to 20) well beyond what the empty-circuit-start lineage (papers 1–2) had demonstrated.

## 8. Position relative to the rest of the lineage (context only)

Per `RL-APPROACHES.md` (Appendix 3), TensorRL-QAS is the final link connecting back through CRLQAS to Ostaszewski et al.: it reuses the reward formula and DDQN backbone from the whole lineage essentially unchanged, while contributing the DMRG/Riemannian-optimization warm-start pipeline, the trainable-vs-fixed TN-init distinction (with the StructureRL ablation to isolate its source of benefit), and a scale-up from CRLQAS's 8-qubit ceiling to 20 qubits. Unlike CRLQAS, which is a same-author, direct methodological extension of paper 1, TensorRL-QAS comes from a different research group and reads as a more loosely coupled continuation — it adopts the lineage's reward/agent core as a known-good baseline to benchmark against and improve on efficiency/scale, rather than extending paper 1's own curriculum machinery (which it in fact discards). The three papers, read together, trace a single throughline — reward/curriculum design (2021) → noise-aware curriculum and depth-awareness (2024) → tensor-network warm start and scalability (2025) — applied throughout to the same core task (RL-driven VQE ansatz construction) and evaluated on a growing, overlapping benchmark family of molecular and spin Hamiltonians.

---

# Comparative Improvement Analysis — Paper 2 (CRLQAS, ICLR 2024) → Paper 3 (TensorRL-QAS, NeurIPS 2025)

This section isolates *only* the deltas between the middle and final links of the lineage: for every axis of the method, what specifically changed, the mechanism of the change, and whether it is a strict improvement, a trade-off, or a simplification made possible by a new upstream capability (the tensor-network warm start). Note a structural difference from the Paper 1→Paper 2 comparison above: that transition was almost entirely *additive* (CRLQAS kept everything from paper 1 and layered new machinery — IA, RH, PTM/GPU, Adam-SPSA — on top to survive a harder regime). The CRLQAS→TensorRL-QAS transition is markedly more *subtractive*: TensorRL-QAS's central move is to insert one powerful new upstream stage (the TN warm start) and then actively **remove** several pieces of machinery CRLQAS had found necessary (the moving-threshold curriculum, illegal-action masking, random halting, qubit-count-scaled network capacity), on the argument that a sufficiently good starting point makes them redundant. Numbers below are pulled directly from both papers' bodies/appendices (§§1–8 above for source context) rather than re-derived.

## A. Scope of the change: what problem got harder, and what got removed as unnecessary

| Dimension | CRLQAS | TensorRL-QAS | Nature of the change |
| :--- | :--- | :--- | :--- |
| Max qubit count demonstrated | 8 ($H_2O$, noiseless); 4 (noisy) | 12 (chemistry: BeH$_2$/H$_2$O/LiH/CH$_2$O); 20 (TFIM spin model, noiseless) | Strict scale-up — roughly 1.5× on chemistry, 2.5× on general Hamiltonians, and the first RL-QAS demonstration past the combinatorial wall the paper's own related-work section cites (8 qubits noiseless / 4 qubits noisy as the prior state of the art) |
| Noisy-regime qubit count | 4 (LiH-4, $H_2$-4) | 8 ($H_2O$-8, amplified depolarizing + shot noise) | 2× — the paper explicitly claims this is "the largest simulations to date of RL-QAS in a noisy setting" |
| Starting point of every episode | Empty circuit (inherited unchanged from paper 1) | Non-trivial warm-start circuit from DMRG + Riemannian-optimized MPS-to-circuit mapping | New pipeline stage entirely absent from papers 1–2; changes what problem the RL agent is actually solving (refine a good guess vs. discover structure from nothing) |
| Curriculum machinery | Two independent axes: moving-threshold (inherited from paper 1) + random halting (new in CRLQAS) | **Neither** — fixed threshold, no episode-length curriculum | Net removal of two active mechanisms, justified by the warm start already avoiding the sparse-reward/long-episode failure modes those mechanisms were built to solve |
| Action-space pruning | Illegal-action masking + coupling-map masking | None claimed as a contribution (only an ad hoc pool expansion at the 20-qubit ceiling) | Net removal — CRLQAS's combinatorial-pruning machinery has no counterpart here |
| Network capacity policy | Scaled with qubit count (1000/2000/5000 neurons) | **Fixed** at 1000 neurons across 6–20 qubits | Reversal of CRLQAS's (and paper 1's) own established scaling pattern |
| Steps needed per episode to converge | 70 (6q) → 250 (8q) → 450 (12q, extrapolated/rerun) | 20 (6–10q) → 40 (12q) → 80 (20q) | 3.5–12× fewer steps at matched qubit counts (see §C, Table 11) — the clearest, most direct signature of "starting closer to the answer" |
| Simulation cost bottleneck being solved | Kraus-operator noisy-channel simulation cost (solved via PTM formalism + GPU/JAX, 6× speedup) | Statevector simulation cost scaling with circuit *depth* at high qubit count (solved via shallower circuits from the warm start, not faster simulation machinery) | Different bottleneck, different lever — CRLQAS made noisy simulation faster per call; TensorRL-QAS instead makes fewer, shorter calls necessary in the first place |
| Real-hardware validation device | IBM Lagos ($H_2$-4 circuit) | IBMQ-Brisbane ($H_2O$-8 circuits: best episode + 2 random samples) | New device, larger circuit, and multiple sampled circuits rather than one |
| Baseline breadth compared against | qubit-ADAPT-VQE, quantumDARTS, QCAS, EVQE, paper 1's RLQAS | CRLQAS (original + matched rerun), vanilla RL-QAS, RA-QAS, training-free QAS, SA-QAS, quantumDARTS, GQAS/GQAS-SSL, plus a new no-RL control (TN-VQE) | Broader — includes a rerun of the immediately preceding paper (CRLQAS) under matched settings for a fair head-to-head, which papers 1–2 did not do to each other in the original manuscripts |
| Problem domains | Quantum chemistry only ($H_2$, LiH, $H_2O$) | Quantum chemistry (BeH$_2$, $H_2O$, LiH, $CH_2O$) **and** spin models (Heisenberg, TFIM) | Genuine domain broadening beyond VQE-for-chemistry |

This table is the mirror image of the Paper 1→Paper 2 scope table: there, nearly every row showed CRLQAS taking on a *harder* condition (noise, connectivity, stochastic cost) that paper 1 never faced, forcing new machinery. Here, several rows show TensorRL-QAS reaching a *harder* scale (qubits, domains) while simultaneously *discarding* machinery, because the warm start changes the shape of the problem rather than just its size.

## B. Component-by-component delta

### B.1 The central new stage: DMRG + Riemannian-optimization warm start (has no counterpart in CRLQAS at all)
- **CRLQAS**, like paper 1, always begins every episode from a genuinely empty circuit; the entire burden of discovering *any* useful structure falls on the RL agent from step zero.
- **TensorRL-QAS** inserts a two-step classical pre-processing pipeline before RL ever starts: (i) DMRG finds a matrix-product-state (MPS) approximation of the target ground state at a chosen bond dimension $\chi$ (default 2); (ii) that MPS is mapped to an actual brickwork quantum circuit of 2-qubit unitaries by maximizing state overlap via Riemannian optimization on the Stiefel manifold (Cayley-transform retraction, Adam-style momentum adapted to the manifold — a fully new classical optimization routine, detailed in Appendix B of paper 3, with no analogue anywhere in the paper 1/CRLQAS lineage).
- **Mechanism of the improvement:** this converts the RL agent's task from "construct a good circuit from nothing" to "refine an already-close-to-correct circuit," which is a qualitatively different (and typically much shorter) search. This is the single root cause behind nearly every downstream number in §C below (fewer steps per episode, fewer optimizer function evaluations, shallower final circuits, faster wall-clock training).
- **A concrete quantitative illustration (Appendix J, Table 9 of paper 3):** the *pure* TN-compiled circuit alone (zero RL refinement) already reaches energy errors of $5.9\times10^{-3}$ (6-BeH$_2$), $2.7\times10^{-3}$ (8-$H_2O$), $4.8\times10^{-3}$ (10-$H_2O$) — i.e., within 1–2 orders of magnitude of chemical accuracy *before the RL agent takes a single action*. CRLQAS's RL agent, by contrast, starts every episode at whatever energy an empty circuit has (essentially the Hartree-Fock reference energy, far worse).
- **A caveat the paper is explicit about, and CRLQAS never had to be:** for 12-LiH, the pure TN-compiled circuit is *not* close (error $1.9\times10^{-1}$ — two orders of magnitude worse than for the other molecules), so the warm start's value is molecule-dependent, not universal. The DMRG-bond-dimension scaling study (Appendix K, Table 10) makes this precise: as $\chi$ increases toward the point where DMRG alone reaches chemical accuracy, RL's marginal contribution shrinks to near-zero — TensorRL-QAS's operating regime is explicitly "where classical tensor methods are still intractable," a scope boundary CRLQAS's design (which never had a classical pre-solver to lean on) does not have.
- **Verdict:** an entirely new capability, not an improvement to any existing CRLQAS component — this is the paper's namesake contribution and the mechanism that makes nearly every other simplification in section B below defensible.

### B.2 Curriculum: two active axes → none (a net removal, not an extension)
- **CRLQAS** runs the moving-threshold curriculum (greedy tightening every $G$ episodes + amortization decay, inherited from paper 1) *and* random halting (negative-binomial episode-length sampling, CRLQAS's own addition) simultaneously.
- **TensorRL-QAS drops both.** The threshold $\xi$ is a **fixed constant** set per task ($1.6\times10^{-3}$ Ha for chemistry — the literal chemical-accuracy bound — $10^{-3}$ for Heisenberg, $10^{-2}$ for TFIM, the latter two calibrated by pre-running SA-QAS on those Hamiltonians). There is no greedy threshold-tightening, no amortization radius, and no stochastic episode-length sampling anywhere in TensorRL-QAS's own method description.
- **Mechanism/justification:** paper 1's curriculum was invented specifically because a *fixed* threshold failed outright (zero successes) on its harder 6-qubit case when starting from an empty circuit — sparse reward with no learning signal. TensorRL-QAS's implicit argument is that once the search starts from a warm start already close to the target energy (per B.1), that sparse-reward failure mode doesn't arise in the first place, so the curriculum becomes unneeded overhead rather than a requirement. Random halting was CRLQAS's answer to "the agent has no incentive to prefer short circuits once it can already succeed"; TensorRL-QAS's episodes are already short by construction (see §C, Table 11: 20 steps vs. CRLQAS's 250 at 8 qubits), so the depth-preference problem RH solved barely exists here either.
- **What this costs, that CRLQAS never had to weigh:** by fixing $\xi$ to the literal chemical-accuracy value from episode 1, TensorRL-QAS re-introduces exactly the risk paper 1's ablation warned about (a fixed, tight threshold can produce zero successes under reward sparsity) — it is only rescued from that risk by the warm start already sitting close to the threshold. This is a bet that would not obviously transfer back to an empty-circuit start; the paper does not test a fixed-threshold, empty-circuit-start ablation to confirm the curriculum specifically (rather than the warm start) is what's dispensable.
- **Verdict:** a genuine methodological simplification enabled by B.1, not a refinement of CRLQAS's curriculum design — it is a different, smaller design (constant threshold) that only works *because* the upstream problem changed shape.

### B.3 Action-space pruning: illegal-action + coupling-map masking → none (also a net removal)
- **CRLQAS** hard-masks two categories of redundant actions (repeated CNOT/rotation on the same wire as the immediately preceding moment) via forced $-\infty$ Q-values, plus separately masks CNOTs violating a hardware coupling map.
- **TensorRL-QAS** claims no equivalent masking mechanism; the paper's efficiency gains are attributed entirely to the warm start, not to combinatorial search-space pruning.
- **Where this shows a real limitation TensorRL-QAS does *not* paper over:** at 20-qubit scale (Appendix H.6), the fixed-TN-init variant plateaus at only 1% relative improvement over the raw MPS state using the standard 4-gate pool — the paper attributes this to a *combined* bottleneck of fixed bond dimension ($\chi=2$) and limited action-space expressivity, and only recovers meaningful improvement (1%→9%) by *expanding* the action pool (adding two-qubit $XX/YY/ZZ$ operators), which is the opposite direction from CRLQAS's action-space intervention (CRLQAS *restricted* the action space via masking; TensorRL-QAS, when it needed to intervene on the action space at all, *expanded* it). This is presented candidly as an open limitation, not a solved problem.
- **Verdict:** removed without a direct replacement for the qubit ranges where the warm start does its job (≤17 qubits); at the 20-qubit ceiling, a different and opposite type of action-space intervention (expansion, not pruning) becomes necessary, revealing that CRLQAS's pruning-based efficiency lever and TensorRL-QAS's warm-start-based efficiency lever address different failure modes and are not simply substitutes for one another in all regimes.

### B.4 Reward formula: unchanged (a "solved primitive" reused across all three papers, now confirmed across a third, independent research group)
- The terminal $\pm5$/dense-shaping formula (Eq. 18 in paper 3) is reused **verbatim** from CRLQAS's (and originally paper 1's) Eq. 2 — same $+5$ if $C_t<\xi$, same $-5$ on timeout, same $\max\!\big(\tfrac{C_{t-1}-C_t}{C_{t-1}-C_{\min}},-1\big)$ dense-shaping term otherwise.
- Notably, this is the *one* component that survives even the cross-institution handoff (paper 1/CRLQAS from Leiden/Delft/Warsaw; TensorRL-QAS from Helsinki/Algorithmiq, a different group entirely) completely intact, reinforcing the `RL-APPROACHES.md` (Appendix 3) framing that the terminal/dense reward expression itself is a field-wide "solved primitive" — what's *not* solved, and what each paper actually contests, is the curriculum wrapped around it (added in CRLQAS, then removed again in TensorRL-QAS, per B.2).
- **Verdict:** no change, no improvement — a control variable held constant across the entire lineage.

### B.5 Classical angle-optimization: one subroutine → two decoupled subroutines
- **CRLQAS** uses a single classical-optimizer slot for the agent's own gates: COBYLA (global, 1000 iterations) in the noiseless case, or its novel 3-stage Adam-SPSA in the noisy case.
- **TensorRL-QAS** splits this into two independent subroutines operating at different pipeline stages: (i) the same lineage-standard COBYLA (1000 iterations, global) refits angles for gates the *RL agent* places, exactly as in CRLQAS; (ii) a **new**, separate Riemannian-Adam/Cayley-transform optimizer (B.1) fits the warm-start circuit's own parameters *before* RL begins and — only in the *trainable*-TN-init variant — continues to be refined jointly with the agent's new gates during RL.
- **Consequence for the noisy regime specifically:** CRLQAS's headline noisy-optimization contribution (the continuously-decaying 3-stage Adam-SPSA, claimed to halve function evaluations under shot noise) is not mentioned as being reused, replaced, or benchmarked against in TensorRL-QAS's own noisy experiment (8-qubit $H_2O$, amplified depolarizing + shot noise, §4.2/Appendix F of paper 3). Paper 3's noise-modeling appendix (Appendix F) describes only the depolarizing and shot-noise channel mathematics, with no mention of PTM formalism, JAX/GPU acceleration, or Adam-SPSA — leaving it unclear from the paper's own text whether CRLQAS's noise-specific optimizer/simulation innovations were quietly reused, replaced by something undocumented, or simply unnecessary at the one noisy qubit-count tested. This is a genuine gap in traceability between the two papers, not a claimed regression — but it means the "noisy-regime angle-optimization" axis cannot be marked either "improved" or "reused" with confidence from the text alone.
- **Verdict:** a net *addition* of one new optimizer stage (Riemannian-Adam for the warm start), decoupled from and running prior to the lineage's existing COBYLA/Adam-SPSA choice for the agent's own placements — not a replacement of CRLQAS's angle-fitting logic, but an extra stage bolted in front of it.

### B.6 Network capacity policy: qubit-count-scaled → fixed regardless of qubit count
- **CRLQAS** (continuing paper 1's own pattern) scales network width with problem size: 1000 neurons (2–4 qubit), 2000 (4-qubit noisy or 6-qubit noiseless), 5000 (8-qubit $H_2O$) — capacity growing directly with input-tensor size.
- **TensorRL-QAS** uses a **single, fixed** 5-hidden-layer, 1000-neuron network across the *entire* 6–20-qubit range tested — a direct reversal of the scaling policy both prior lineage papers established.
- **Mechanism:** because the warm start reduces the *effective* remaining search (B.1) and correspondingly the input tensor the agent must reason over stays smaller/simpler for longer (fewer RL-added gates before success, per §C), a fixed-capacity network apparently suffices even at 20 qubits — a claim CRLQAS's own empty-circuit-start design could never have supported, since its state/action complexity kept growing directly with qubit count all the way to its own 8-qubit ceiling.
- **Verdict:** an emergent simplification, not a targeted design choice with its own ablation — the paper does not report what happens if network capacity *were* scaled up for the largest (20-qubit) case, so it's not established that fixed capacity is optimal, only that it was sufficient for the results reported.

### B.7 Computational-efficiency lever: faster simulation vs. fewer/shorter episodes
- **CRLQAS's** efficiency contribution (PTM formalism + GPU/JAX) attacks the *per-call* cost of simulating a noisy circuit — same number of RL steps and optimizer calls as before, but each one is up to 6× cheaper to evaluate.
- **TensorRL-QAS's** efficiency contribution attacks the *number* of calls needed: up to **100-fold** fewer classical-optimizer function evaluations and up to **98%** faster per-episode execution time versus CRLQAS, achieved with *no* new simulation infrastructure — purely by needing fewer RL-added gates (and correspondingly fewer angle-refits) to reach the target because of the warm start.
- **Directly comparable numbers (Appendix M.2, Table 11 of paper 3 — steps needed per episode to converge):**

| Qubits | TensorRL-QAS | CRLQAS | RA-QAS |
| :--- | :--- | :--- | :--- |
| 6 | 20 | 70 | 97 |
| 8 | 20 | 250 | 300 |
| 10 | 20 | 350 | 477 |
| 12 | 40 | 450 | 450 |

This is a **3.5×–17.5× reduction** in steps-per-episode at matched qubit counts, growing (not shrinking) as qubit count increases — i.e., the warm-start advantage *compounds* with scale rather than being a fixed-size head start, which is why the paper can credibly claim the approach "scales" rather than just "starts faster."
- **A genuinely new, TensorRL-specific efficiency finding CRLQAS's infrastructure work never surfaced:** Appendix L of paper 3 shows *statevector* simulation time (not noisy-channel simulation, which was CRLQAS's target) becomes the dominant bottleneck at high qubit count purely from circuit *depth* — at 12 qubits, CRLQAS-scale circuits (depth ≈400) take ≈10s per statevector call, projecting to >27 hours of pure simulation time over a 10,000-episode training run, versus TensorRL-QAS's shallower circuits (depth ≈100) taking <0.5s per call (≈1.5 hours projected). CRLQAS's PTM/GPU investment was aimed at a different cost driver (noise-channel recompilation) and does not address this depth-driven statevector cost at all — the two papers' "make training computationally tractable" contributions are complementary, not competing, solutions to different bottlenecks.
- **Verdict:** a different and, at scale, larger-magnitude lever than CRLQAS's — but not a strict improvement on CRLQAS's specific contribution, since it solves a different problem (call *count*, not per-call *cost*) that only becomes dominant once qubit count/depth grows past where CRLQAS operated.

### B.8 State representation: same tensor format, new "visibility" axis for the warm start
- **CRLQAS** uses the 3D binary tensor (qubit × moment × gate-type) it introduced over paper 1's flat vector; the state always corresponds to a fully agent-visible circuit built entirely from the agent's own past actions.
- **TensorRL-QAS inherits this tensor format unchanged** (paper 3 cites CRLQAS/[35] directly for the encoding) but adds a design axis CRLQAS never needed: whether the *warm-start* portion of the circuit is exposed to the agent's state at all.
  - *Trainable TN-init*: warm-start structure + parameters occupy the first $D_{MPS}$ tensor slices, fully visible and further trainable by the agent, growing the state to $(D_{MPS}+D)\times N \times (N+N_{1\text{-qubit}})$.
  - *Fixed TN-init*: the warm start is invisible to the state entirely — it only sets the simulator's initial statevector — shrinking the state to $D\times N\times(N+N_{1\text{-qubit}})$, with $D_{MPS}$ dropped completely.
- **Mechanism/consequence:** this is a genuinely new lever with no CRLQAS analogue, because CRLQAS (like paper 1) never had a non-trivial initial condition to choose whether to expose. The *fixed* variant is what actually delivers TensorRL-QAS's largest efficiency numbers (98% faster episodes, CPU-only trainability up to 8 qubits) precisely because it keeps the tensor small regardless of warm-start depth; the *trainable* variant sacrifices some of that speed but (per Table 1 of paper 3) generally achieves *lower* energy error, since the agent can correct imperfections in the DMRG/Riemannian-mapping stage rather than being stuck with them.
- **Verdict:** an extension of CRLQAS's own state-encoding convention along a new axis (warm-start visibility) that didn't exist as a question before TensorRL-QAS's upstream stage (B.1) created it — not a re-design of the tensor format itself.

### B.9 Random-agent and non-RL control baselines: broadened
- **CRLQAS** compares against QCAS, quantumDARTS, qubit-ADAPT-VQE, EVQE, and paper 1's own RLQAS — all either RL-based or non-RL search methods, but none of them isolate "value of a good initialization alone, with zero search."
- **TensorRL-QAS** adds **TN-VQE** — the same DMRG/Riemannian warm start followed by plain hardware-efficient-ansatz VQE with *no* RL/QAS refinement at all — specifically to answer whether the warm start alone (without any architecture search on top) is sufficient. Results (Table 1 of paper 3) show TN-VQE consistently underperforms every RL-based variant and often fails to reach chemical accuracy for larger molecules, empirically confirming that RL search still contributes real, non-redundant value on top of the tensor-network initialization — i.e., B.1's warm start is a force-multiplier for RL, not a replacement for it.
- **Verdict:** a new type of control baseline (isolating initialization-only performance) that CRLQAS's experimental design had no need for, since CRLQAS never had a non-trivial initialization to isolate.

### B.10 Benchmark/scale breadth and real-hardware validation
- **CRLQAS:** $H_2$ (2/3/4q), LiH (4/6q), $H_2O$ (8q); noiseless + finite-shot + IBM Mumbai/Ourense noise profiles; one real-hardware point (IBM Lagos, $H_2$-4).
- **TensorRL-QAS:** BeH$_2$ (6q), $H_2O$ (8/10q), LiH (12q), $CH_2O$ (8/10q) for chemistry; Heisenberg (5q) and TFIM (6/15/17/20q) for spin models; noiseless + one amplified-noise 8-qubit point; real-hardware validation on IBMQ-Brisbane across three sampled circuits (best episode + 2 random).
- **Verdict:** a strict expansion in both qubit-count ceiling and problem-domain breadth (chemistry-only → chemistry + spin models), though the *noisy*-regime breadth is comparatively narrower than CRLQAS's (CRLQAS tested multiple named device-noise profiles at multiple severities across 3 molecules; TensorRL-QAS tests one amplified-noise configuration on one molecule) — the paper's own framing ("largest simulation to date... at most 4-qubit prior work") is about qubit-count scale specifically, not about noise-condition breadth, where CRLQAS remains the more thorough study.

## C. Quantitative before/after: the shared $H_2O$-8 benchmark

$H_2O$-8 (noiseless) is the cleanest shared, matched-condition benchmark point: CRLQAS reports it directly, and TensorRL-QAS reruns CRLQAS under matched agent/environment settings specifically for a fair comparison (its own Table 1), in addition to citing CRLQAS's original numbers.

| Method | Energy error | Depth | CNOT | ROT |
| :--- | :--- | :--- | :--- | :--- |
| CRLQAS (original, paper 2) | $1.8\times10^{-4}$ | 75 | 105 | 35 |
| CRLQAS (rerun, in paper 3) | $1.7\times10^{-4}$ | 100 | 85 | 58 |
| TensorRL (fixed TN-init) | $8.9\times10^{-4}$ | **6** | **9** | 15 |
| TensorRL (trainable TN-init) | $2.0\times10^{-4}$ | 36 | 30 | 146 |
| StructureRL (ablation) | $1.3\times10^{-4}$ | 33 | 30 | 133 |

Reading this honestly rather than only citing the favorable cells: **TensorRL (fixed)'s energy error (8.9×10⁻⁴) is actually higher — worse — than either CRLQAS configuration (1.7–1.8×10⁻⁴)**, roughly 5× worse, even though its circuit is 8–12× shallower and uses 9–12× fewer CNOTs. TensorRL (trainable) closes most of that accuracy gap (2.0×10⁻⁴, within range of CRLQAS) while still using a much shallower/lower-CNOT circuit (36 depth / 30 CNOTs vs. CRLQAS's 75–100 depth / 85–105 CNOTs) — this configuration is the one that comes closest to a genuine same-metric win over CRLQAS on this exact benchmark. As with the Paper 1→Paper 2 LiH-6 comparison in §C above, **no single TensorRL-QAS configuration strictly dominates CRLQAS on every metric simultaneously** at this shared benchmark point: *fixed* wins hugely on compactness/speed at a real accuracy cost; *trainable* is roughly accuracy-competitive with CRLQAS while still meaningfully more compact, making it the more defensible "strict improvement" candidate of the two variants, at the cost of losing most of *fixed*'s speed advantage (see B.7 — trainable needs the larger, warm-start-visible tensor and therefore forfeits much of the 98% episode-time reduction).

The **noisy** $H_2O$-8 comparison (Table 2 of paper 3) tells a cleaner, more unambiguous story: TensorRL (fixed) reaches $9.0\times10^{-4}$ error at depth 7 / 5 CNOTs with **100% success rate**, versus CRLQAS (rerun) at $1.3\times10^{-3}$ error, depth 22 / 11 CNOTs, and only **30% success rate** — here TensorRL (fixed) *does* dominate on every reported metric (lower error, shallower, fewer CNOTs, much higher reliability) simultaneously, making the noisy-regime comparison the stronger, less caveated improvement claim of the two.

The **steps-per-episode** comparison (§B.7, Table 11) is unambiguous in both regimes: TensorRL-QAS needs 3.5–17.5× fewer RL steps to converge at every matched qubit count from 6 to 12, regardless of which accuracy/compactness trade-off is examined.

## D. Net assessment

| Axis | Verdict |
| :--- | :--- |
| Warm-start pipeline (DMRG + Riemannian MPS-to-circuit mapping) | **New capability** — the root-cause enabler behind nearly every other change in this section; no CRLQAS analogue |
| Curriculum (moving threshold + random halting) | **Net removal** — both of CRLQAS's active curriculum axes are dropped in favor of a fixed threshold, justified by (not independently proven apart from) the warm start |
| Action-space masking (illegal actions + coupling map) | **Net removal** for ≤17 qubits; replaced at the 20-qubit ceiling by the *opposite* intervention (action-pool expansion, not pruning) to escape an expressivity bottleneck |
| Reward formula | **Unchanged** — the one component that survives intact across all three papers and a change of research group |
| Classical angle optimization | **Extended, not replaced** — a new pre-RL Riemannian-Adam stage added in front of CRLQAS's existing COBYLA/Adam-SPSA choice; the noisy-regime optimizer's fate (reused vs. replaced vs. unneeded) is not traceable from paper 3's text |
| Network capacity policy | **Reversed** — fixed-size network replaces both prior papers' qubit-count-scaled convention, apparently viable only because the warm start shrinks the effective remaining search |
| Computational-efficiency lever | **New, complementary lever** — attacks call-*count* (steps/evaluations needed) rather than CRLQAS's per-call *cost* (noisy-simulation speed); the two are not substitutes, they solve different bottlenecks that dominate at different qubit-count regimes |
| State representation (tensor format) | **Unchanged format**, extended with a new warm-start-visibility axis (trainable vs. fixed) that only exists because B.1 introduced something worth choosing to hide |
| Non-RL control baselines | **New** (TN-VQE) — isolates warm-start-alone performance, confirming RL still adds real value on top |
| Benchmark/domain breadth | **Expanded** in qubit count (8→20) and domain (chemistry-only → chemistry + spin models); **narrowed** in noise-condition variety (CRLQAS's multi-device, multi-severity noise sweep vs. TensorRL-QAS's single amplified-noise configuration) |
| Real-hardware validation | **Expanded** — new device (IBMQ-Brisbane), larger circuit (8q vs. 4q), multiple sampled circuits rather than one |
| Steps needed per episode | **Strict, compounding improvement** — 3.5–17.5× fewer steps at every matched qubit count, advantage growing with scale |
| Shared-benchmark accuracy (noiseless $H_2O$-8) | **Not unconditional** — TensorRL (fixed) trades accuracy for extreme compactness/speed; TensorRL (trainable) is closer to a strict win but forfeits most of the speed advantage |
| Shared-benchmark accuracy (noisy $H_2O$-8) | **Strict improvement** — TensorRL (fixed) dominates CRLQAS on error, depth, CNOT count, and success rate simultaneously |

**Overall reading:** where CRLQAS's relationship to paper 1 was best described as *the same core recipe, hardened and extended with new machinery to survive a harder regime*, TensorRL-QAS's relationship to CRLQAS is closer to *the same core recipe, relieved of most of its machinery because a new upstream stage changed the shape of the problem being handed to it*. CRLQAS spent its novelty budget on surviving noise and connectivity constraints by *adding* curriculum axes, masking rules, and noise-specific infrastructure. TensorRL-QAS spends its novelty budget almost entirely upstream — on the warm start itself — and then *subtracts* nearly every downstream mechanism CRLQAS had found necessary (curriculum, masking, qubit-scaled capacity), betting that a good enough starting point makes them redundant rather than proving each removal is safe in isolation. This bet pays off clearly on efficiency (3.5–17.5× fewer steps, up to 100× fewer optimizer evaluations, up to 98% faster episodes, scaling to 20 qubits vs. CRLQAS's 8) and on the noisy shared benchmark (an unambiguous, all-metrics win at $H_2O$-8), but is more qualified on noiseless shared-benchmark accuracy (a genuine Pareto trade rather than a strict win, exactly as in the Paper 1→Paper 2 comparison) and leaves at least one traceability gap the paper does not close: whether CRLQAS's own noisy-regime innovations (PTM/GPU simulation, Adam-SPSA) were carried forward, replaced, or rendered moot in TensorRL-QAS's own (narrower) noisy experiments.
