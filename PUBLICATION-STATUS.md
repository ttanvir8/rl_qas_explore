# QAS Papers: Peer-Review & Venue Ranking Status

This document checks every paper in the [Papers](README.md#papers) section of the main [README](README.md) against its actual publication record — was it only ever posted to arXiv, or did it clear peer review at a journal or conference? Where it was published, the venue is ranked using **Scimago Journal Rank** (best quartile, Q1–Q4) for journals and the **CORE Conference Ranking** (A\*/A/B/C) for CS/AI conferences.

**Method:** each paper's arXiv record (`journal_ref`/`comment`/`doi` fields) was checked first, since authors update these on acceptance; where absent, a targeted web search was run for the paper title plus "accepted"/"proceedings"/"journal". Venue rankings were then looked up on Scimago and the CORE portal. Checked as of **2026-07-01**. This is a point-in-time snapshot — some "preprint only" papers may since have been accepted elsewhere; please verify before citing.

## Table of Contents
- [Summary](#summary)
- [Important Caveats](#important-caveats)
- [Q1 Journal Publications](#q1-journal-publications)
- [A* Conference Publications](#a-conference-publications)
- [Q2 Journal Publications](#q2-journal-publications)
- [Other Peer-Reviewed (Workshops, Companion Tracks, Non-CORE Conferences)](#other-peer-reviewed-workshops-companion-tracks-non-core-conferences)
- [Preprint-Only / Not Yet Published](#preprint-only--not-yet-published)

## Summary

| Status | Count |
|---|---|
| Q1 journal | 22 |
| A* conference | 6 |
| Q2 journal | 6 |
| Other peer-reviewed (workshop / companion track / non-CORE conference) | 12 |
| Preprint only (arXiv, no acceptance found) | 22 |
| **Total** | **68** |

Roughly **two-thirds** of the list (46/68) cleared some form of peer review; the remainder are arXiv preprints, which is expected for a fast-moving field where a large share of the list is from 2025–2026.

## Important Caveats
- **IEEE QCE (Quantum Computing and Engineering)** is not tiered by CORE — CORE2023 lists it as "TBR" (to be ranked) and ICORE2026 as "National/Regional." Papers accepted there are genuinely peer-reviewed conference papers, just without an A\*/A/B/C tier. Same treatment applies to **IEEE QAI** (Quantum Artificial Intelligence), which doesn't appear in CORE at all.
- **Workshop and companion/poster tracks** (e.g. GECCO Companion, GECCO-satellite QCCC, an ICML workshop) are reviewed but are **not** the main track of their parent conference, so they do not inherit the parent's CORE rank (e.g. GECCO main track is CORE A, but its Companion volume is not separately ranked).
- Row **#14** (*Evolutionary-based quantum architecture search*, 2022) — the Quantum Information Processing publication match was inferred from matching authors/content under a slightly retitled published version ("Evolutionary-based searching method for quantum circuit architecture"); worth a manual spot-check before relying on it.
- Row **#44** (*TensorRL-QAS*) — its NeurIPS 2025 acceptance was independently confirmed via the arXiv `comment` field ("Accepted at NeurIPS 2025"), not just the README's own "[NeurIPS 2025]" label.
- Scimago's own site (scimagojr.com) returned HTTP 403 to some automated fetches; those quartiles were cross-checked via mirrors/aggregators and should be considered reliable but not first-party-verified in every case.
- Papers from April–June 2026 (items #65–68) are recent enough that "preprint only" may simply mean review is still in progress, not that they were rejected or informal.

---

## Q1 Journal Publications
*(22 papers)*

| Paper (Year) | Venue |
|---|---|
| [Quantum optimization with a novel Gibbs objective function and ansatz architecture search (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023074) | Physical Review Research |
| [DQAS: Differentiable quantum architecture search (2021)](https://arxiv.org/abs/2010.08561) | Quantum Science and Technology (IOP), 7 045023 (2022) |
| [Neural predictor based quantum architecture search (2021)](https://arxiv.org/abs/2103.06524) | Machine Learning: Science and Technology (IOP), 2 045027 (2021) |
| [Quantum Architecture Search with Meta-Learning (2021)](https://arxiv.org/abs/2106.06248) | Advanced Quantum Technologies (Wiley), 5, 2100134 (2022) |
| [Quantum circuit architecture search for variational quantum algorithms (2022)](https://arxiv.org/abs/2010.10217) | npj Quantum Information, 8, 62 (2022) |
| [EQNAS: Evolutionary Quantum Neural Architecture Search for Image Classification (2023)](https://www.sciencedirect.com/science/article/pii/S0893608023005348) | Neural Networks (Elsevier) |
| [Quantum architecture search via truly proximal policy optimization (2023)](https://www.nature.com/articles/s41598-023-32349-2) | Scientific Reports, 13, 5157 |
| [Distributed quantum architecture search (2024)](https://arxiv.org/abs/2403.06214) | Physical Review A, 110, 022403 (2024) |
| [KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search (2024)](https://arxiv.org/abs/2406.17630) | EPJ Quantum Technology, 11, 76 |
| [Application of ZX-calculus to quantum architecture search (2024)](https://arxiv.org/abs/2406.01095) | Quantum Machine Intelligence, 7, 34 (2025) |
| [A quantum information theoretic analysis of reinforcement learning-assisted quantum architecture search (2024)](https://arxiv.org/abs/2404.06174) | Quantum Machine Intelligence, 6, 49 (2024) |
| [Reinforcement learning-based architecture search for quantum machine learning (2024)](https://arxiv.org/abs/2406.02717) | Machine Learning: Science and Technology (IOP) |
| [Quantum Circuit Optimization with AlphaTensor (2024)](https://arxiv.org/abs/2402.14396) | Nature Machine Intelligence (2025) |
| [Quantum circuit synthesis with diffusion models (2024)](https://arxiv.org/abs/2311.02041) | Nature Machine Intelligence (2024) |
| [Quantum Architecture Search with Unsupervised Representation Learning (2025)](https://arxiv.org/abs/2401.11576) | Quantum journal, 10, 1994 (2026) |
| [Topology-Driven Quantum Architecture Search Framework (2025)](https://arxiv.org/abs/2502.14265) | Science China Information Sciences, 68, 180507 (2025) |
| [Self-supervised representation learning for Bayesian quantum architecture search (2025)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.032403) | Physical Review A, 111, 032403 (2025) |
| [Reinforcement learning with learned gadgets to tackle hard quantum problems on real hardware (2025)](https://arxiv.org/abs/2411.00230) | Communications Physics, 9, 44 (2026) |
| [Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware (2025)](https://arxiv.org/abs/2501.11454) | Machine Learning: Science and Technology (IOP) |
| [Neural Architecture Search Algorithms for Quantum Autoencoders (2025)](https://arxiv.org/abs/2509.15451) | IEEE Transactions on Quantum Engineering |
| [Quantum-Based Self-Attention Mechanism for Hardware-Aware Differentiable Quantum Architecture Search (2025)](https://arxiv.org/abs/2512.02476) | Physical Review Applied |
| [Differentiable Architecture Search for Adversarially Robust Quantum Computer Vision (2026)](https://arxiv.org/abs/2601.18058) | Quantum Machine Intelligence (Springer) |

## A* Conference Publications
*(6 papers)*

| Paper (Year) | Venue |
|---|---|
| [Reinforcement learning for optimization of variational quantum circuit architectures (2021)](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html) | NeurIPS 2021 |
| [QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits (2022)](https://arxiv.org/abs/2107.10845) | HPCA 2022 |
| [QuantumDARTS: Differentiable Quantum Architecture Search for Variational Quantum Algorithms (2023)](https://proceedings.mlr.press/v202/wu23v.html) | ICML 2023 (PMLR v202) |
| [Training-Free Quantum Architecture Search (2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29135) | AAAI 2024 |
| [Curriculum reinforcement learning for quantum architecture search under hardware errors (2024)](https://arxiv.org/abs/2402.03500) | ICLR 2024 |
| [TensorRL-QAS: Reinforcement learning with tensor networks for scalable quantum architecture search (2025)](https://arxiv.org/abs/2505.09371) | NeurIPS 2025 |

## Q2 Journal Publications
*(6 papers)*

| Paper (Year) | Venue |
|---|---|
| [Search space pruning for quantum architecture search (2022)](https://link.springer.com/article/10.1140/epjp/s13360-022-02714-7) | European Physical Journal Plus |
| [Quantum Circuit Architecture Search on a Superconducting Processor (2022)](https://arxiv.org/abs/2201.00934) | Entropy (MDPI), 26(12), 1025 (2024) |
| [Evolutionary-based quantum architecture search (2022)](https://arxiv.org/abs/2212.00421) | Quantum Information Processing (retitled on publication; see caveat above) |
| [A GNN-based predictor for quantum architecture search (2023)](https://link.springer.com/article/10.1007/s11128-023-03881-x) | Quantum Information Processing |
| [GSQAS: Graph Self-supervised Quantum Architecture Search (2023)](https://arxiv.org/abs/2303.12381) | Physica A: Statistical Mechanics and its Applications, 630, 129286 |
| [An RNN-policy gradient approach for quantum architecture search (2024)](https://arxiv.org/abs/2405.05892) | Quantum Information Processing (2024) |

## Other Peer-Reviewed (Workshops, Companion Tracks, Non-CORE Conferences)
*(12 papers — genuinely reviewed, but not a CORE-tiered main track or Scimago-ranked journal)*

| Paper (Year) | Venue | Note |
|---|---|---|
| [Automated Design of Quantum Circuits (1998)](https://link.springer.com/chapter/10.1007/3-540-49208-9_8) | QCQC'98 (Quantum Computing and Quantum Communications), Springer LNCS 1509 | Predates/outside CORE's scope |
| [Evolutionary quantum architecture search for parametrized quantum circuits (2022)](https://arxiv.org/abs/2208.11167) | GECCO '22 Companion (ACM) | Companion/poster track, not GECCO's CORE-A main track |
| [Quantum Reinforcement Learning for Quantum Architecture Search (2023)](https://dl.acm.org/doi/abs/10.1145/3588983.3596692) | QCCC'23 workshop (satellite of GECCO 2023) | Workshop, not GECCO main track |
| [Differentiable Quantum Architecture Search for Quantum Reinforcement Learning (2023)](https://arxiv.org/abs/2309.10392) | IEEE QCE 2023, QML workshop | IEEE QCE not CORE-tiered |
| [Differentiable Quantum Architecture Search in Asynchronous Quantum Reinforcement Learning (2024)](https://arxiv.org/abs/2407.18202) | IEEE QCE 2024 | IEEE QCE not CORE-tiered |
| [Quantum Machine Learning Architecture Search via Deep Reinforcement Learning (2024)](https://arxiv.org/abs/2407.20147) | IEEE QCE 2024 | IEEE QCE not CORE-tiered |
| [SA-DQAS: Self-attention Enhanced Differentiable Quantum Architecture Search (2024)](https://arxiv.org/abs/2406.08882) | ICML 2024 "Differentiable Almost Everything" workshop (poster) | Workshop poster, not ICML main track |
| [Challenges for Reinforcement Learning in Quantum Circuit Design (2024)](https://arxiv.org/abs/2312.11337) | IEEE QCE 2024 | IEEE QCE not CORE-tiered |
| [Differentiable Quantum Architecture Search in Quantum-Enhanced Neural Network Parameter Generation (2025)](https://arxiv.org/abs/2505.09653) | IEEE QCE 2025 | IEEE QCE not CORE-tiered |
| [QAS-QTNs: Curriculum Reinforcement Learning-Driven Quantum Architecture Search for Quantum Tensor Networks (2025)](https://arxiv.org/abs/2507.12013) | IEEE QCE 2025 | IEEE QCE not CORE-tiered |
| [Quantum Long Short-term Memory with Differentiable Architecture Search (2025)](https://arxiv.org/abs/2508.14955) | IEEE QAI 2025 | IEEE QAI not present in CORE |
| [Neural Architecture Search for Quantum Autoencoders (2025)](https://arxiv.org/abs/2511.19246) | IEEE QCE 2025 | IEEE QCE not CORE-tiered |

## Preprint-Only / Not Yet Published
*(22 papers — no journal/conference acceptance found as of 2026-07-01)*

- [Quantum architecture search via deep reinforcement learning (2021)](https://arxiv.org/abs/2104.07715)
- [Quantum Architecture Search via Continual Reinforcement Learning (2021)](https://arxiv.org/abs/2112.05779)
- [Quantum circuit optimization with deep reinforcement learning (2021)](https://arxiv.org/abs/2103.07585)
- [Trainability maximization using estimation of distribution algorithms assisted by surrogate modelling for quantum architecture search (2024)](https://arxiv.org/abs/2407.20091)
- [Benchmarking Quantum Architecture Search with Surrogate Assistance (2025)](https://arxiv.org/abs/2506.06762)
- [Scalable Quantum Architecture Search via Landscape Analysis (2025)](https://arxiv.org/abs/2505.05380)
- [DeQompile: quantum circuit decompilation using genetic programming for explainable quantum architecture search (2025)](https://arxiv.org/abs/2504.08310)
- [RhoDARTS: Differentiable Quantum Architecture Search with Density Matrix Simulations (2025)](https://arxiv.org/abs/2506.03697)
- [Scaling the Automated Discovery of Quantum Circuits via Reinforcement Learning with Gadgets (2025)](https://arxiv.org/abs/2503.11638)
- [QuProFS: An Evolutionary Training-free Approach to Efficient Quantum Feature Map Search (2025)](https://arxiv.org/abs/2508.07104)
- [Quantum Architecture Search for Solving Quantum Machine Learning Tasks (2025)](https://arxiv.org/abs/2509.11198)
- [Hybrid action Reinforcement Learning for quantum architecture search (2025)](https://arxiv.org/abs/2511.04967)
- [QASER: Breaking the Depth vs. Accuracy Trade-Off for Quantum Architecture Search (2025)](https://arxiv.org/abs/2511.16272)
- [Distributed quantum architecture search using multi-agent reinforcement learning (2025)](https://arxiv.org/abs/2511.22708)
- [Noise-Aware Quantum Architecture Search Based on NSGA-II Algorithm (2026)](https://arxiv.org/abs/2601.10965)
- [Probabilistic Design of Parametrized Quantum Circuits through Local Gate Modifications (2026)](https://arxiv.org/abs/2602.12465)
- [AI Agents for Variational Quantum Circuit Design (2026)](https://arxiv.org/abs/2602.19387)
- [How to find expressible and trainable parameterized quantum circuits? (2026)](https://arxiv.org/abs/2603.14451)
- [Replay-buffer engineering for noise-robust quantum circuit optimization (2026)](https://arxiv.org/abs/2604.21863) — very recent (posted 2026-04-23), may still be in review
- [Magic-Informed Quantum Architecture Search (2026)](https://arxiv.org/abs/2605.03932) — very recent (posted 2026-05-05), may still be in review
- [Q-PhotoNAS: Hybrid Quantum Neural Architecture Search Framework on Photonic Devices (2026)](https://arxiv.org/abs/2605.22097) — very recent (posted 2026-05-21), may still be in review
- [Zero-shot Quantum Neural Architecture Search (2026)](https://arxiv.org/abs/2605.27410) — very recent (posted ~2026-06), may still be in review

---

*This file is a derived, publication-status view of [README.md](README.md). If you add a new paper to the README, please also verify its venue and file it into the appropriate section here.*
