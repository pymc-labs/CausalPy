# Bayesian synthetic control references

Provenance for later synthetic-control work. These pins record where the published descriptions were read. They are not a claim that CausalPy reproduces any published number, and they do not choose an estimator.

## Ridge augmented synthetic control

Ben-Michael, Feller, and Rothstein describe augmented synthetic control as a simplex synthetic-control fit plus an outcome model that estimates the bias from imperfect pre-treatment balance. Their main proposal uses ridge regression for that outcome model. The ridge penalty keeps the resulting weights near the simplex solution while allowing negative weights when extrapolation improves pre-treatment fit. The paper studies that estimator under a linear outcome model and under a latent factor model.

- Paper: Ben-Michael, Feller, and Rothstein, *The Augmented Synthetic Control Method*, arXiv:1811.04170. <https://arxiv.org/abs/1811.04170>. PDF: <https://arxiv.org/pdf/1811.04170.pdf>.
- Package vignettes, pin only: `ebenmichael/augsynth` commit `7e70072232fe75057aa8b5d7af08c7e1aef53ae8` (2026-09-04). [Single treated unit](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/vignettes/singlesynth-vignette.md) and [several treated units](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/vignettes/multisynth-vignette.md).

## Interactive fixed effects

Xu's generalized synthetic control method imputes a treated counterfactual from a linear interactive fixed-effects model. Factors and control loadings are estimated from control units. Treated loadings are estimated by projecting pre-treatment treated outcomes onto those factors. The counterfactual is then the factor product for the treated unit. Difference-in-differences is the case with no extra interactive factors.

- Paper: Xu, *Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models*, Political Analysis 25:57–76 (2017). <https://doi.org/10.1017/pan.2016.2>. Author PDF: <https://yiqingxu.org/papers/english/2016_Xu_gsynth/Xu_PA_2017.pdf>. Supplement: <https://yiqingxu.org/papers/english/2016_Xu_gsynth/Xu_2017_SM.pdf>.
- Package pages read for provenance, not as a numerical target: [gsynth](https://yiqingxu.org/packages/gsynth/) (saved page text identifies v1.5.0 as a wrapper of fect and points IFE and matrix-completion fits at fect) and [fect](https://yiqingxu.org/packages/fect/) (saved manual identifies v2.4.5).
