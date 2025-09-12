# Principal Component Analysis (PCA) - Dimension Reduction

## 1) Intuition

We observe data points \(x_1,\dots,x_n \in \mathbb{R}^d\) 

Empirically, the cloud of points often lies near a much lower-dimensional subspace. 

PCA searches for orthonormal directions \(v_1,\dots,v_k\) that capture as much variance as possible, then projects data onto the span of these directions. 

First, we centre the data:
\[
\mu \;=\; \frac{1}{n}\sum_{i=1}^n x_i,
\qquad
\tilde{x}_i \;=\; x_i - \mu
\]

For a unit vector \(v \in \mathbb{R}^d\), the variance of the projection of the data onto the line spanned by \(v\) is:
\[
\operatorname{Var}(v)
\;=\; \frac{1}{n}\sum_{i=1}^n \big(v^\top \tilde{x}_i\big)^2
\]
Intuitively: project points onto \(v\), measure their spread; larger spread \(\Rightarrow\) more “signal” along \(v\).

---

## 2) Sample Covariance Matrix

Recall: Sample Variance (onto \(v\) )
\[
\operatorname{Var}(v)
\;=\; \frac{1}{n}\sum_{i=1}^n \big(v^\top \tilde{x}_i\big)^2 = v^\top \left(\frac{1}{n}\sum_{i=1}^n \tilde{x}_i \tilde{x}_i^\top\right) v
\]

Define the **sample covariance matrix**: 

\[
\Sigma =\; \frac{1}{n}\sum_{i=1}^n \tilde{x}_i \tilde{x}_i^\top
\;=\; \frac{1}{n} X^\top X
\]
- Using \(1/n\) makes the optimisation clean; some texts use \(1/(n-1)\) for an unbiased estimator, but the PCA directions are unaffected by that constant factor. 

Then, 
\[
\operatorname{Var}(v) = v^\top \Sigma \, v
\]

---

## 3) PCA for Dimensionality Reduction

The **first principal component** is the direction \( v \) of maximal variance:
\[
v_1 \;=\; \arg\max_{\|v\|=1} \; \operatorname{Var}(v) = \arg\max_{\|v\|=1} \;  v^\top \Sigma \, v
\]
Subsequent components are found similarly, but constrained to be orthogonal to the previous ones.

This functional \(R(v) = \frac{v^\top \Sigma v}{\|v\|}\) is the **Rayleigh quotient**.

---

## 4) Optimising

We have the optimisation problem: 
\[
\max_{v\in\mathbb{R}^d, \|v\|=1} \; v^\top \Sigma \; v
\]

Consider the Lagrangian:
\[
\mathcal{L}(v,\lambda) \;=\; v^\top \Sigma v - \lambda (v^\top v - 1)
\]
Stationary points satisfy: 
\[
\nabla_v \mathcal{L} = 2\Sigma v - 2\lambda v = 0
\quad\Longleftrightarrow\quad
\Sigma v = \lambda v
\]
Thus any optimiser is an **eigenvector** of \(\Sigma\). 

The Rayleigh quotient attains its **maximum** at the eigenvector associated with the **greatest eigenvalue**: \(\lambda_1\)

Orthogonality of eigenvectors of a symmetric matrix yields mutually orthogonal principal components \(v_1,\dots,v_d\) with variances \(\lambda_1 \ge \cdots \ge \lambda_d \ge 0\)

---

## 5) Projection and Reconstruction

Let \(V_k = [\,v_1\;\cdots\; v_k\,] \in \mathbb{R}^{d\times k}\) be the matrix of the top \(k\) eigenvectors (orthonormal columns).

Consider the orthogonal projecton: \(\operatorname{span}(V_k)\) by \(P_k = V_k V_k^\top \in \mathbb{R}^{d\times d}\)

**Encoder (to coordinates in \(\mathbb{R}^k\)):**
\[
E:\ \mathbb{R}^d \to \mathbb{R}^k,\qquad
z_i \;=\; V_k^\top \hat{x}_i  =\; V_k^\top (x_i - \mu) 
\]

Here \(z_i\) are the coordinates of \(\hat{x}_i\) in the basis: \(V_k\)


**Decoder (back to \(\mathbb{R}^d\) in the subspace):**
\[
D:\ \mathbb{R}^k \to \mathbb{R}^d,\qquad
\hat{x}_i \;=\; \mu + V_k z_i
\]

Equivalently, composing \(D \circ E\) gives the orthogonal projection:
\[
\hat{x}_i \;=\; \mu + V_k V_k^\top \hat{x}_i \;=\; \mu + P_k \hat{x}_i
\]

### Decomposition and Pythagoras

Let \(\tilde{x}_i = x_i - \mu\)
Let \(V_k \in \mathbb{R}^{d\times k}\) be the orthonormal basis of the top \(k\) principal components.  

Define the projection matrix:
\[
P_k = V_k V_k^\top, \qquad P_k^2 = P_k, \quad P_k^\top = P_k
\]

#### Residual Vector
The orthogonal **residual** (the part of \(\tilde{x}_i\) not captured by the top \(k\) components) is:
\[
r_i = (I - P_k)\tilde{x}_i
\]

#### Decomposition
By the orthogonal projection theorem, every data point decomposes uniquely as: 
\[
\tilde{x}_i = P_k \tilde{x}_i + (I - P_k)\tilde{x}_i 
= \underbrace{V_k z_i}_{\in\, \mathrm{span}(V_k)} 
\;+\; 
\underbrace{r_i}_{\perp\, \mathrm{span}(V_k)}
\]
Where we have the PCA term: 
\[
z_i = V_k^\top \tilde{x}_i \;\in\; \mathbb{R}^k
\]


#### Pythagoras identity
Since the two parts are orthogonal,
\[
\langle V_k z_i, \; r_i \rangle = 0,
\]
We have: 
\[
\|\tilde{x}_i\|^2 
= \|V_k z_i\|^2 + \|r_i\|^2
\]

Equivalently, because \(z_i\) are coordinates in an orthonormal basis: 
\[
\|V_k z_i\|^2 = \|z_i\|^2
\]
So, 
\[
\|\tilde{x}_i\|^2 = \|z_i\|^2 + \|r_i\|^2
\]

#### Variance Interpretation
- \(\|z_i\|^2\) measures how much of the sample’s "energy" lies within the principal subspace.  
- \(\|r_i\|^2\) is the reconstruction error for sample: \(i\)  

Averaging over all samples gives
\[
\frac{1}{n}\sum_{i=1}^n \|r_i\|^2 \;=\; \sum_{j=k+1}^d \lambda_j,
\]
Which is the sum of the eigenvalues of the discarded components. 

This shows PCA minimises reconstruction error: no other \(k\)-dimensional subspace can make the residual smaller.
