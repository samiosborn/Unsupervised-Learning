# Evidence Lower Bound (ELBO)

> **Goal:** For a latent-variable model, bound the log evidence \(\log p_\theta(x)\) from below by an expression that is tractable to optimise. Later, we’ll specialise this to Markov chains.

---

## Setup

- Observed data: \(x \in \mathcal{X}\)
- Latents: \(z \in \mathcal{Z}\) (single latent) or \(z_{1:T}\) (Markov chain)
- Model parameters: \(\theta\); variational parameters: \(\phi\)

---

## Model (joint & true posterior)

- **Joint model:** \(p_\theta(x,z) = p_\theta(x\mid z)\,p_\theta(z)\)
- **Evidence (marginal likelihood):** \(p_\theta(x) = \int p_\theta(x,z)\,dz\)
- **True/Exact posterior (typically intractable):** \(p_\theta(z\mid x) = \frac{p_\theta(x,z)}{p_\theta(x)}\)


---

## Variational family (approximate posterior)

- **Variational distribution:** \(q_\phi(z\mid x)\)
- **Goal:** maximise the ELBO w.r.t. \(\theta,\phi\) so that \(q_\phi(z\mid x)\) approximates \(p_\theta(z\mid x)\)

---

## Theorem

**ELBO (one-latent form):**
\[
\mathcal{L}(\theta,\phi; x)
\;=\;
\mathbb{E}_{q_\phi(z\mid x)}\!\left[ \log p_\theta(x,z) - \log q_\phi(z\mid x) \right]
\;\le\; \log p_\theta(x)
\]

Equivalent decompositions:
\[
\mathcal{L}(\theta,\phi;x)
= \underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x\mid z)\right]}_{\text{Reconstruction}} - \underbrace{\mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right)}_{\text{Regulariser}}
\]

-  Kullback–Leibler ("KL") divergence: \(\mathrm{KL}(q\|p) = \mathbb{E}_q\!\big[\log ( \tfrac{q}{p} ) \big]\)

Gap identity:
\[
\log p_\theta(x) - \mathcal{L}(\theta,\phi;x)
= \mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)\;\;\ge 0
\]

---


### Reminder: Jensen’s Inequality

Let \(X\) be an \(\mathbb{R}^d\)-valued random variable with \(\mathbb{E}\|X\|<\infty\), taking values in a convex set \(\mathcal{C}\subseteq\mathbb{R}^d\)

Let \(f:\mathcal{C}\to \mathbb{R}\cup\{+\infty\}\) be **convex** with \( \mathbb{E}\big[\,|f(X)|\,\big]<\infty \)

Then, 
\[
f\!\big(\mathbb{E}[X]\big)\; \le\; \mathbb{E}\!\left[f(X)\right]
\]

**Conditional form:** For a sub-\(\sigma\)-algebra \(\mathcal{G}\),
\[
f\!\big(\mathbb{E}[X\mid \mathcal{G}]\big)\; \le\; \mathbb{E}\!\left[f(X)\mid \mathcal{G}\right] \quad \text{a.s.}
\]

**Equality conditions:**
- If \(f\) is **strictly convex** and \(\operatorname{Var}(X)>0\), the inequality is **strict**
- Equality holds iff \(X\) is a.s. supported on a set where \(f\) is **affine** (in 1-D: essentially \(X\) is a.s. constant, unless \(f\) is linear on the support)


#### Concave corollary

If \(f\) is **concave**, apply the above to \(-f\) to get
\[
f\!\big(\mathbb{E}[X]\big)\; \ge\; \mathbb{E}\!\left[f(X)\right]
\]
In particular, with \(f=\log\) i.e. strictly concave on \((0,\infty)\),
\[
\mathbb{E}[\log Y] \;\le\; \log \mathbb{E}[Y],
\quad\text{for } Y>0 \text{ with } \mathbb{E}[Y]<\infty
\]
and equality iff \(Y\) is a.s. constant.

---

## Proof of ELBO

Integral over joint distribution:
\[
\log p_\theta(x) = \log \int p_\theta(x,z)\,dz
\]

Multiplying by 1: 
\[
\log \int p_\theta(x,z)\,dz = \log \int q_\phi(z\mid x)\,\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
\]

Converting to expectation: 
\[
\log \int q_\phi(z\mid x)\,\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
= \log \,\mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\]

Overall: 
\[
\log p_\theta(x) = \log \,\mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\]

Use Jensen's Inequality: 
\[
\log \,\mathbb{E}_{q_\phi}[Y] \;\ge\; \mathbb{E}_{q_\phi}[\log Y]
\quad\text{for}\quad Y=\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\]

Gives: 
\[
\log \,\mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \ge  \mathbb{E}_{q_\phi} \left[ \log{ \frac{p_\theta(x,z)}{q_\phi(z\mid x)} } \right]
\]
Hence:
\[
\log p_\theta(x)
\;\ge\;
\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x,z) - \log q_\phi(z\mid x)\right]
= \mathcal{L}(\theta,\phi;x)
\]

---

## Equivalent Forms

- **Entropy form:**
  \[
  \mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\log p_\theta(x\mid z)\right] + \mathbb{E}_{q_\phi}\!\left[\log p_\theta(z)\right] + \mathcal{H}\!\left[q_\phi(z\mid x)\right]
  \]
    - where \(\mathcal{H}[q] = -\mathbb{E}_q[\log q]\)

- **ELBO gap:**
  \[
  \log p_\theta(x) - \mathcal{L} = \mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
  \]
