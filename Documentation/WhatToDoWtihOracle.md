# What To Do With Oracle

Assume the oracle is a function that takes three things: the labelled training data, a subject ancestry coordinate vector $a$, and a target variant $j$. Its job is to look at the training data, focus on the part of the data it considers relevant for ancestry position $a$, and return an estimated effect size for the target variant. Write that output as $\hat b_j(a)$. In the procedures below, the default interpretation is that this is a marginal effect estimate for variant $j$ at ancestry location $a$, unless stated otherwise.

Let the training data be

$$
\mathcal D_{train} = \{(y_i, a_i, c_i, g_i)\}_{i=1}^n
$$

where:

- $y_i \in \{0,1\}$ is the disease label.
- $a_i \in \mathbb R^{16}$ is the ancestry coordinate vector.
- $c_i$ is a vector of optional non-genetic covariates.
- $g_i = (g_{i1}, \dots, g_{ip})$ is the genotype dosage vector.

For a target subject $i$, the main personalized genetic score has the form

$$
S_i = \sum_{j} G_{ij} w_{ij}
$$

where the main question is how to define the per-variant weight $w_{ij}$ from the oracle output while controlling double counting from LD.

## Procedure 1: LD-Pruned Oracle Score

This is the simplest and safest starting point.

Assume LD pruning has already produced a set of approximately independent variants $S$. For a target subject $i$ with ancestry coordinate $a_i$, query the oracle for each retained variant:

$$
\hat b_j(a_i), \quad j \in S.
$$

Then form the personalized score:

$$
S_i^{(1)} = \sum_{j \in S} G_{ij} \hat b_j(a_i).
$$

Because the variants are already pruned, the risk of counting the same signal several times is reduced.

To turn this into a disease risk estimate, fit a calibration model on training or cross-fit data:

$$
\Pr(y_i = 1) = \sigma\left(\alpha + \gamma^\top c_i + \theta S_i^{(1)}\right)
$$

where $\sigma(x) = 1 / (1 + e^{-x})$.

Operational steps:

1. Start from an LD-pruned SNP set $S$.
2. For each subject $i$ and each variant $j \in S$, compute the oracle effect $\hat b_j(a_i)$.
3. Sum $G_{ij} \hat b_j(a_i)$ over the retained variants.
4. Fit a logistic calibration layer using the score and optional covariates.
5. Evaluate on held-out data.

This procedure assumes the oracle may be returning marginal effects, but pruning makes that less dangerous.

## Procedure 2: Oracle Contributions Plus Ridge Calibration

This procedure allows a denser SNP set while using a second-stage model to control redundancy.

For each subject $i$ and variant $j$, define the personalized oracle contribution

$$
X_{ij}^{oracle} = G_{ij} \hat b_j(a_i).
$$

Instead of summing these contributions directly with equal downstream weight, fit a penalized logistic model:

$$
\Pr(y_i = 1) = \sigma\left(\alpha + \gamma^\top c_i + \sum_{j=1}^p \lambda_j X_{ij}^{oracle}\right).
$$

Estimate the coefficients by minimizing

$$
-\ell(\alpha, \gamma, \lambda) + \rho \sum_{j=1}^p \lambda_j^2
$$

where $\ell$ is the logistic log-likelihood and $\rho > 0$ is the ridge penalty strength.

The final personalized score is then

$$
S_i^{(2)} = \sum_{j=1}^p \lambda_j G_{ij} \hat b_j(a_i).
$$

Why this helps: if several correlated variants are all tracking the same signal, the ridge penalty shrinks their combined influence and reduces inflation.

Operational steps:

1. For each subject $i$ and variant $j$, compute $\hat b_j(a_i)$ from the oracle.
2. Build the feature matrix with entries $X_{ij}^{oracle} = G_{ij} \hat b_j(a_i)$.
3. Fit a ridge-penalized logistic regression of disease status on these oracle-derived features and optional covariates.
4. Tune $\rho$ by validation.
5. Use the fitted model to obtain the final risk score or predicted probability.

This procedure still starts from marginal oracle effects, but the downstream ridge model compensates for multicollinearity.

## Procedure 3: LD-Matrix Correction of Marginal Oracle Effects

This procedure starts from marginal oracle effects and tries to convert them into approximate conditional effects before scoring.

For a target ancestry location $a$, define the vector of oracle marginal effects over a set of variants in one LD block:

$$
b_{marg}(a) = \begin{bmatrix}
\hat b_1(a) \\
\hat b_2(a) \\
\vdots \\
\hat b_m(a)
\end{bmatrix}.
$$

Let $\beta_{cond}(a)$ be the vector of conditional effects we would prefer to use. The working approximation is

$$
b_{marg}(a) \approx R(a)\beta_{cond}(a)
$$

where $R(a)$ is the ancestry-local LD correlation matrix for that block.

Therefore estimate conditional effects by

$$
\hat\beta_{cond}(a) = \left(R(a) + \rho I\right)^{-1} b_{marg}(a)
$$

with $\rho > 0$ and $I$ the identity matrix.

The reason for adding $\rho I$ instead of using $R(a)^{-1}$ directly is numerical stability. If two variants are nearly identical, $R(a)$ can be singular or nearly singular, and the inverse can explode. The ridge term stabilizes the inverse and shrinks implausibly large conditional coefficients.

To compute $R(a)$ for a block, standardize the genotype dosages in the training data and then estimate a weighted correlation matrix using weights based on ancestry proximity to $a$. For example, with training subject weights

$$
w_r(a) = \exp\left(-\frac{\|a_r - a\|^2}{2h^2}\right),
$$

the LD matrix entries are weighted correlations among the block genotypes across training individuals.

Once the conditional effect vector is estimated for the target ancestry location, score the subject by

$$
S_{i,block}^{(3)} = G_{i,block}^\top \hat\beta_{cond}(a_i).
$$

Then sum across LD blocks:

$$
S_i^{(3)} = \sum_{blocks} S_{i,block}^{(3)}.
$$

Operational steps:

1. Partition variants into LD blocks.
2. For each target subject ancestry location $a_i$ and each block, query the oracle to get the blockwise marginal effect vector $b_{marg}(a_i)$.
3. Estimate the ancestry-local LD matrix $R(a_i)$ for that block from weighted training genotypes.
4. Compute the stabilized conditional effect estimate $\hat\beta_{cond}(a_i) = (R(a_i) + \rho I)^{-1} b_{marg}(a_i)$.
5. Compute the block score and sum across blocks.
6. Optionally add a final calibration model for disease probability.

This procedure is the most direct way to use a marginal-effect oracle while correcting for multicollinearity mathematically.

## Recommended Order

For a first implementation, the most sensible order is:

1. Procedure 1 as the baseline.
2. Procedure 2 as the first denser and more flexible extension.
3. Procedure 3 as the more ambitious LD-aware correction.

That sequence keeps the first model stable and interpretable, then adds progressively stronger handling of correlated variants.