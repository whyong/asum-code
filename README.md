# 🌌 Accumulated-State Universe Model (ASUM)

> **Non-Equilibrium Phase Transitions and Bistable Turing Oscillations Driven by Information-Thermodynamic Dualism**

**ASUM** (涌积态宇宙模型) is a purely algebraic, matrix-evolution engine for simulating non-equilibrium complex systems. Built on the core axiom of **Information-Thermodynamic Dualism**, it drives a ternary scalar field system (Substrate, Intermediate, High-order structure) to undergo spontaneous phase transitions and spatial differentiation under unsupervised conditions.

*This theory and architecture were independently formulated and developed by **Wang Hongyong** between **March 26, 2026, and April 15, 2026**.*

Through 21 days of continuous iteration and rigorous mathematical verification, ASUM successfully reproduced **Bistable Turing Oscillations** and **First-Order Phase Transitions** in a continuous 1,000,000-frame Deep Time numerical experiment, with an unprecedented mass-conservation drift of only `0.003%`.

## 🔬 Core Philosophy: The L/D Dualism

How does order emerge from chaos? ASUM hypothesizes that the evolution of complex systems is governed by an iron law: **"Precipitate Time through Loss, Create Space through Distortion."**

This is achieved via two adversarial operators interacting on a 2D periodic grid:

1. **Context-Adaptive Lossy Operator (**$\mathcal{L}$**)**: Acts as a low-pass filter on the time series. It uses a context-adaptive exponential decay to extract macroscopic topological trends, precipitating historical memory. *(Analogue: Dynamic KV-Cache eviction, Sparse Attention in LLMs)*.
2. **Generalized Distortion Operator (**$\mathcal{D}$**)**: Acts as a high-pass spatial perturbation. It injects Laplacian inertial flux and multiplicative Gaussian noise to force symmetry breaking and explore unknown phase spaces. *(Analogue: Forward noise injection in Diffusion Models, decoding temperature)*.

The final macro-structure (Turing Patterns) is the continuous residual of the eternal confrontation between Order ($\mathcal{L}$) and Chaos ($\mathcal{D}$).

## ✨ Key Discoveries & Features

- **Spontaneous Turing Pattern Emergence**: From a purely uniform random noise substrate, ASUM spontaneously forms spatial periodic patterns, perfectly aligning with Turing's (1952) Reaction-Diffusion theory.
- **Deep Time Ultra-Stability**: Engineered with a divergence-form diffusion solver (Half-grid flux method), guaranteeing strict global mass conservation. The 1-million-frame epic test (v8.0 T-010) maintained a cumulative float drift of only `0.003004%`.
- **Emergent Metric (Gravity-locked Patterns)**: Regions with high information density dynamically depress their local diffusion coefficients, creating a topological "black hole" effect that locks Turing patterns in space.
- **Bistable Phase Transitions**: Without supervised intervention, the system autonomously accumulates potential energy, triggering a first-order phase transition from a low-valley attractor ($f_{C2} \approx 0.17$) to an ultra-stable high-level Turing attractor ($f_{C2} \approx 0.33$).
- **Oscillation Counter & Dual-Threshold (v11.0)**: The final architecture broke the sub-valley stagnation bottleneck by treating oscillation history as a macroscopic topological feature, achieving the highest historical spatial differentiation index ($C_{var} = 0.106$).

## 🧬 System Ontology: The Ternary Field

The universe runs on a $128 \times 128$ periodic grid, defined by three interacting scalar fields:

- $P$ **(Substrate Field)**: The absolute foundational material / energy.
- $C_1$ **(Intermediate Field)**: The transitional carrier of information accumulation.
- $C_2$ **(High-Order Structure)**: The ultimate target state, the spatial carrier of Turing patterns.

The system rigorously obeys global mass conservation:

$$\frac{d}{dt} \int_\Omega (P + C_1 + C_2) \, dV = 0$$

## 🚀 Implications for AI and Complex Systems

While initially wrapped in physical metaphors (quantum fluctuations, gravity emergence), ASUM's underlying mathematical skeleton is highly isomorphic to **Self-Organizing Neural Networks**.

The dynamic routing of the $\mathcal{L}$ operator and the phase space exploration of the $\mathcal{D}$ operator provide a mathematically verified PDE framework for designing context-adaptive memory networks, ecological simulations, and multi-agent thermodynamic game models.

## 🛠 Installation & Usage

ASUM is built entirely on standard Python scientific computing libraries. No heavy GPU frameworks required.

### Requirements

- Python 3.13+
- NumPy
- SciPy

### Quick Start

```
git clone [https://github.com/whyong/asum.git](https://github.com/whyong/asum.git)
cd ASUM

# Run the v11.0 final baseline model (Standard 24,000 frames)
python universe_v11.py
```

*Note: The project contains archived versions (v5.0 to v11.0) along with their respective design documents, test logs, and ablation study reports.*

## 📜 Documentation & Architecture

For a deep dive into the mathematical proofs, parameter baselines, and historical iteration logs, please refer to the Chinese documentation included in the repository:

- `涌积态宇宙模型 — 完整研究封存报告.md` (Complete Research Archival Report)
- `涌积态宇宙模型 v11.0 — 架构设计文档.md` (v11.0 Architecture Design)
- `V8-T-010 百万帧长程测试归档报告.md` (1-Million Frame Test Log)

## ⚖️ License & Academic Archiving

### Open Source License

This project is open-sourced under the **GNU Affero General Public License v3.0 (AGPLv3)**. This ensures that any modifications, derivative works, or network-deployed services based on this mathematical engine remain fully open-source and accessible to the community.

### Zenodo Archiving

To guarantee scientific reproducibility and permanent academic record, the complete theoretical framework, code snapshots, ablation studies, and the 1-million-frame test logs are permanently archived on **Zenodo**.

## 👤 Author & Citation

- **Author & Theory Creator**: Wang Hongyong (王洪勇)
- **Theory Formulation Period**: March 26, 2026 – April 15, 2026

If you use ASUM's logic, operators, or code in your research regarding Non-equilibrium Thermodynamics, Artificial Life, or Generative AI Architectures, please formally cite the Zenodo archive:

```
@software{wang_hongyong_2026_asum,
  author       = {Wang Hongyong},
  title        = {The Accumulated-State Universe Model (ASUM): A Matrix-Evolution Engine for Complex Systems},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v11.0},
  doi          = {10.5281/zenodo.20149583},
  url          = {[https://doi.org/10.5281/zenodo.20149583](https://doi.org/10.5281/zenodo.20149583)}
}
```

*“Use loss to precipitate time, use distortion to create space.”*