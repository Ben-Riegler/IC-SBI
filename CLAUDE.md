# IC-SBI Project

## Paper
**"Encoding Inductive Biases in Simulation-based Inference"** — targeting NeurIPS 2026.
Method: **AICGM** (Amortized Implicit Copula Generative Models).

---

## Part 1: Current State of the Paper

### Core Idea
Two-step posterior approximation using Sklar's theorem:
1. **Marginal posteriors** q_φ(θ_j | x) per dimension, learned via MDNs with NLL loss. Mixture components are tailored to each parameter's support (Gaussian, Gamma, Beta, NegBin, etc.)
2. **Joint copula** C_φ(y|x) implicitly defined by a **latent Gaussian MDN p_φ(y|x) on ℝ^m**. The copula of this latent distribution is matched to the copula of the true posterior.

**Inference (Algorithm 1):**
1. Sample y ~ p_φ(·|x) from the latent GMM on ℝ^m
2. Map to copula space: v = (F_φ(y_j|x))_{j=1}^m ∈ [0,1]^m using the latent marginal CDFs
3. Map to parameter space: θ_j = F_φ^{-1}(v_j|x) by inverting the learned marginal MDN CDFs

### Key Design Points
- **Latent GMM p_φ(y|x)** lives on ℝ^m. Its copula C_φ(y|x) is implicitly defined and approximates the true posterior copula C(θ|x).
- **Marginal MDNs q_φ(θ_j|x)**: per-parameter, head family matched to support. SharedEmbedMultiMDN uses shared encoder + independent per-dim heads + constant variance as an inductive bias.
- **Discrete parameters**: randomized PIT (Eq. 11): u = F(θ_j-1|x) + α(F(θ_j|x) - F(θ_j-1|x)), α~U(0,1), which yields u ~ U(0,1) by Proposition 1.
- **Modular training**: marginal MDNs trained first via NLL; latent GMM trained second via energy distance + Gumbel-Softmax straight-through.
- **Inductive biases on the copula** can be encoded via the architecture and initialization of the latent GMM (e.g., diagonal covariance = independence prior, K=1 = Gaussian copula).

### Current Experiments (3, all synthetic)

| Paper row | Script | MDN | DGP |
|---|---|---|---|
| Gaussian (encoding structure) | `experiments/normal_exper_gmm_GS.py` | `models/multi_mdn.py` | Isotropic Normal-Normal conjugate |
| Bounded Support | `experiments/mixed_exper_gmm_GS.py` | `models/mixed_mdn.py` | N×Gamma×Beta prior, Bin+Normal simulator |
| Cont. + Discrete | `experiments/negbin_exper_gmm_GS.py` | `models/beta_negbin_gamma_mdn.py` | Beta×NegBin×Gamma prior, Bin+Normal sim |

### Known Weaknesses
- **No baselines** — results are uninterpretable without NPE/FMPE comparisons. Need at least FMPE (Wildberger et al.) and Pawsterior (`pdfs/Pawsterior.pdf`)
- **No real-world benchmarks** — must add ≥1 task from sbibm (Lueckmann et al. 2021)
- **No sample efficiency curves** — the efficiency claim needs LPP vs N plots with baselines
- **Gaussian experiment too weak** — isotropic covariance = independent marginals, trivial for any method
- **Introduction framing wrong** — "generative methods don't support inductive biases" is false. Correct argument: standard flow training is geometry-agnostic, wasting simulator budget on infeasible regions. Reframe around sample efficiency and encodability, not expressiveness. Lead with discrete parameters as the strongest case.
- **Pawsterior** needs direct engagement — variational flow matching with geometric confinement, handles bounded/discrete/mixed. Needs comparison or explicit differentiation (modularity, interpretability, simpler training).

### File Map

**Directory structure:**
```
CopSBI/
├── DGPs/
│   └── data.py
├── eval/
│   ├── abc_reference.py
│   └── metrics.py
├── models/
│   ├── beta_negbin_gamma_mdn.py
│   ├── gen_gmm_GS.py
│   ├── mdn_inv.py
│   ├── mixed_mdn.py
│   └── multi_mdn.py
├── experiments/
│   ├── normal_exper_gmm_GS.py
│   ├── mixed_exper_gmm_GS.py
│   └── negbin_exper_gmm_GS.py
├── plots.py
└── utils.py
```

**Experiments should be run from the project root** (e.g., `python -m experiments.normal_exper_gmm_GS`) so that `models/`, `DGPs/`, `eval/`, `plots.py`, and `utils.py` are all importable.

**Core (active):**
- `models/gen_gmm_GS.py` — latent GMM (p_φ(y|x) on ℝ^m). Key: `GMM`, `train_GMM_generator`, `sample_GMM_gen`, `gmm_marg_cdf`
- `models/multi_mdn.py` — `SharedEmbedMultiMDN` (Gaussian marginals). `get_multiMDN_cdf_vals` for PIT
- `models/mixed_mdn.py` — `MixedMarginalMDN` (Gaussian/Gamma/Beta heads). `get_mixed_cdf_vals`, `mixed_mdn_inv_marg` (bisection)
- `models/beta_negbin_gamma_mdn.py` — `BetaNegBinGammaMDN` (Beta/NegBin/Gamma). Randomized PIT for discrete NegBin. Also contains `gen_beta_negbin_gamma_data` and `abc_rejection` for that DGP.
- `models/mdn_inv.py` — bisection inversion for Gaussian MDN marginals
- `DGPs/data.py` — DGPs: `gen_mv_normal_normal_data`, `gen_mixed_prior_binomial_normal_data` (has `dependence` flag), `gen_two_moons_data`
- `eval/metrics.py` — `c2st_jax`, `sbc` (SBC with ECDF plots)
- `eval/abc_reference.py` — ABC rejection sampler (reference posterior for mixed experiment)
- `plots.py` — plotting helpers
- `utils.py` — save/load model params

### Tech Stack
- JAX + Flax (linen) + Optax (AdamW)
- `jax_enable_x64=False` in all experiment scripts; `True` in `DGPs/data.py` and `models/multi_mdn.py`
- Experiments save to `experiments/<folder>/` with `config.txt`, `DGP.npz`, `Pars.npz`, `metrics.txt`
- Env managed with `uv` (`pyproject.toml`, `uv.lock`)

---

## Part 2: Target Paper Structure and Experiments

### Structural Goal
Inspired by PRISM (Tueboys et al.) and RoPE (Wehenkel et al.): ~4.5 pages of background/method, then a dense experiments section progressing simple→complex and synthetic→real-world. Ablations embedded within experiments. Each experiment yields one clear, quotable insight. The visual identity of the paper is a paired **"no IB vs with IB"** comparison for each capability.

### Key Method Properties (anchors for experiment design)
1. **Arbitrary support handling** — randomized PIT handles continuous, discrete, and mixed parameters principally
2. **Per-parameter inductive bias** — MDN head family matched to each parameter's marginal (Gaussian/Gamma/Beta/NegBin)
3. **Inductive bias on dependence structure** — latent GMM architecture encodes known independence, correlation structure, or block structure in the copula
4. **Modularity** — swap or retrain individual marginal heads without touching the latent GMM / copula
5. **Sample efficiency** — all IB types reduce the simulations needed to reach target posterior quality (the unifying theme across all experiments)
6. **Interpretable marginals** — clean q(θ_j | x) per dimension as a byproduct; flows require expensive numerical marginalization

### Experiments

**Exp 1 — Per-parameter marginal IB (synthetic)**
DGP with visibly non-Gaussian marginals (skewed, bounded). Baseline: AICGM with all-Gaussian heads. Compare to AICGM with matched heads (Gamma, Beta, etc). Clean ablation isolating the value of matching head family to marginal family.
- Key figure: LPP vs N, paired posterior plots showing tail/boundary failures of wrong heads
- Insight: "Matching the head family to the marginal yields correct posteriors with fewer simulations; mismatched heads fail at support boundaries."

**Exp 2 — Arbitrary support vs unconstrained flow (synthetic)**
Same DGPs, but baseline is NPE/FMPE. Shows flows assign probability mass outside the support, degrading with fewer simulations.
- Key figure: LPP vs N with NPE/FMPE baselines, posterior sample plots showing out-of-support mass
- Insight: "Unconstrained flows violate support constraints; AICGM's support-aware marginals guarantee zero out-of-support mass and converge faster."

**Exp 3 — Dependence structure IB (synthetic, new)**
DGP where the true copula has known structure (e.g., near-Gaussian/elliptical, or known block-independence). Compare unconstrained latent GMM vs structure-constrained GMM (diagonal covariance = independence prior, K=1 = Gaussian copula). Constrained version should converge faster when structure matches reality.
- Key figure: LPP vs N for constrained vs unconstrained copula
- Insight: "Encoding partial knowledge of the dependence structure into the latent GMM reduces the simulation budget needed to learn the joint posterior."

**Exp 4 — Modularity (synthetic)**
Train a full AICGM. Change the simulator for one parameter (different marginal family). Retrain only that MDN head, freeze the latent GMM, show posterior quality is recovered. Compare cost to full retrain. No other SBI method supports this.
- Key figure: LPP before/after partial retrain vs full retrain as function of retrain budget
- Insight: "When one simulator component changes, only the affected marginal needs retraining — the copula transfers."

**Exp 5 — sbibm benchmark**
≥1 task from sbibm (Lueckmann et al. 2021): SIR or Lotka-Volterra (positive parameters, known marginal families, established baselines). Shows AICGM is competitive while being more sample-efficient.
- Key figure: LPP vs N with NPE/FMPE baselines, SBC ECDF plots
- Insight: "AICGM matches or exceeds NPE/FMPE while requiring fewer simulations when the prior family is known."

**Exp 6 (optional) — Interpretable marginals**
Show that AICGM gives clean marginal posteriors for free from the MDN heads. Compare to marginals obtained by sampling a flow and marginalizing. Compelling in higher dimensions.

### Metrics
- **LPP** (Log-Posterior Probability): (1/n) Σ log p̃(θⁱ|xⁱ) on labeled test set. Primary accuracy metric. Directly computable from AICGM's analytic density (see Part 3). Higher is better.
- **ACAUC** (Average Coverage AUC): scalar calibration metric. Zero = perfectly calibrated, positive = overconfident, negative = underconfident. Computable from posterior samples. Replaces SBC in tables.
- **SBC ECDF plots** — kept as visualization figures, not table metrics. ACAUC is its scalar summary.
- **LPP vs N curves** — the unifying visual across all experiments.
- **Posterior predictive RMSE** — for sbibm tasks where downstream metrics exist.

---

## Part 3: Analytic Density and NLL Training

### AICGM Has a Fully Analytic Joint Density

By Sklar's theorem, defining p(θ|x) via a copula and marginals always gives a valid normalized density:

> p(θ|x) = c(F₁(θ₁|x), ..., F_m(θ_m|x) | x) · Πⱼ q_φ(θⱼ|x)

where c is a valid copula density (uniform marginals on [0,1]^m) and q_φ(θⱼ|x) are the learned marginal MDN densities.

The copula c is implicitly defined by the **latent GMM p_φ(y|x) on ℝ^m**. The copula of a GMM is analytic:

> c_φ(u|x) = p_φ(y|x) / Πⱼ p_{φ,j}(yⱼ|x),   where y = (Φ_{φ,j}^{-1}(uⱼ))_{j=1}^m

where p_{φ,j}(yⱼ|x) are the 1D marginal densities of the latent GMM (themselves 1D GMMs with the same weights — tractable). This copula has uniform marginals by construction. The full posterior density is therefore analytic and properly normalized. This enables **LPP as an exact evaluation metric** with no kernel density estimation.

### NLL Training Avenue

Since p_φ(θ|x) is analytic, one can replace the current energy distance objective with the standard NPE objective:

> L(φ) = E_{p(θ,x)}[log p_φ(θ|x)]

This is strictly better justified than energy distance. Variants:

**Freeze marginals, train latent GMM with NLL:** With fixed marginal MDNs, the u = F_φ(θ|x) values are deterministic. Maximizing E[log c_φ(u|x)] w.r.t. GMM parameters is equivalent to MLE for the latent GMM — cleaner than energy distance and preserves modularity (frozen marginals means swapping one doesn't invalidate the copula).

**Pre-train marginals, then fine-tune jointly:** Natural initialization exploiting the modular structure, then allowing marginals and copula to jointly adapt via the full NLL.

**Discrete parameters:** The randomized PIT makes u stochastic given θ, complicating direct density evaluation. Options worth trying empirically:
- Deterministic upper CDF: u = F(θ|x) — simple, biased
- Stochastic PIT at training time — unbiased in expectation, adds noise; theoretically preferred

This NLL training direction is both a practical improvement (better loss) and a potential theoretical contribution: AICGM is a structured density model trained via NPE, where the copula decomposition is an architectural choice encoding inductive biases.
