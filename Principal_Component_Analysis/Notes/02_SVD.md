# Singular Value Decomposition (SVD)

## 1) Motivation

In PCA, we saw that the directions of maximal variance are given by the eigenvectors of the **sample covariance matrix**:
\[
\Sigma = \frac{1}{n} X^\top X
\]

Computing eigenvectors of \(\Sigma\) directly is possible, but in practice we use a more powerful and numerically stable tool - **Singular Value Decomposition (SVD)**.

---

## 2) SVD Theorem

Let \(X \in \mathbb{R}^{n\times d}\) 

The SVD theorem states there exists a factorisation: 
\[
X = U\,S\,V^\top
\]

Where: 
- \(U \in \mathbb{R}^{n\times n}\) orthogonal (\(U^\top U = I_n\)). The columns \(u_1,\dots,u_n\) are called the **left singular vectors**.
- \(V \in \mathbb{R}^{d\times d}\) orthogonal (\(V^\top V = I_d\)). The columns \(v_1,\dots,v_d\) are called the **right singular vectors**.
- \(S \in \mathbb{R}^{n\times d}\) diagonal (rectangular) with non-negative entries \(s_1 \ge s_2 \ge \cdots \ge s_r > 0\) on the main diagonal (and zeros elsewhere), where \(r=\mathrm{rank}(X)\)

#### Singular Vectors and Scalars

- **Singular values** \( s_j \) are the non-negative scalar entries on the diagonal of the matrix \( S \).
- **Right singular vectors** (columns of \(V\)) satisfy:
\[
X^\top X\,v_j = s_j^2\,v_j,\qquad v_j \in \mathbb{R}^d
\]
    - They live in the input/domain space \(\mathbb{R}^d\).
    - The \( X \) is on the right. 

- **Left singular vectors** (columns of \(U\)) satisfy: 
\[
X X^\top\,u_j = s_j^2\,u_j,\qquad u_j \in \mathbb{R}^n
\]
    - They live in the output/codomain space \(\mathbb{R}^n\).
    - The \( X \) is on the left. 

- The triplets link as:
\[
X\,v_j = s_j\,u_j,\qquad X^\top u_j = s_j\,v_j
\]

Note: Singular does not mean “noninvertible” here.

---

## 3) Proof of SVD

Let \(X \in \mathbb{R}^{n \times d}\)

#### Step 1: Symmetric Gram matrices

Consider the two symmetric, positive semi-definite matrices:
\[
X^\top X \in \mathbb{R}^{d \times d}
\qquad
X X^\top \in \mathbb{R}^{n \times n}
\]

- Both are symmetric: \((X^\top X)^\top = X^\top X\)
- Both are positive semi-definite: 
\[
v^\top (X^\top X) v = \|X v\|^2 \ge 0
\]

Thus, by the **spectral theorem**, each has an orthonormal eigenbasis.

#### Step 2: Eigen-decomposition of \(X^\top X\)

Presume the eigen-decomposition is: 
\[
X^\top X = V \Lambda V^\top
\]

Where: 
- \(V \in \mathbb{R}^{d \times d}\) is orthogonal, with column vectors \(v_1,\dots,v_d\) 
- \(\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_d)\) with \(\lambda_j \ge 0\)

#### Step 3: Define singular values

Define: 
\[
s_j = \sqrt{\lambda_j}, \quad j=1,\dots,d
\]

These are non-negative scalars, the **singular values** of \(X\).  

Reorder them such that: 
\[
s_1 \ge s_2 \ge \cdots \ge s_r > 0, \quad s_{r+1} = \cdots = 0
\]
Where:  \(r = \mathrm{rank}(X)\)

#### Step 4: Construct left singular vectors

Define \( u_j \) as: 
\[
u_j = \frac{1}{s_j} X v_j
\]

- For each non-zero \(s_j\) with \( j = 1, \cdots, r \) 

Claim: \( u_j \) are orthonormal.
Proof: 
\[
u_i^\top u_j = \frac{1}{s_i s_j} v_i^\top X^\top X v_j
= \frac{1}{s_i s_j} v_i^\top \lambda_j v_j
= \delta_{ij}
\]

Thus \(\{u_1,\dots,u_r\}\) form an orthonormal set in \(\mathbb{R}^n\).

Extend this to a full orthonormal basis \(\{u_1,\dots,u_n\}\) of \(\mathbb{R}^n\). 

Let \(U\) be the matrix with these columns.

#### Step 5: Define the diagonal matrix \(S\)

Let: 
\[
S = \begin{bmatrix}
\mathrm{diag}(s_1,\dots,s_r) & 0 \\
0 & 0
\end{bmatrix} \in \mathbb{R}^{n \times d}
\]

#### Step 6: Verify the decomposition

For each \(j \le r\),
\[
X v_j = s_j u_j
\]

Thus, if \(V = [v_1,\dots,v_d]\), \(U = [u_1,\dots,u_n]\), and \(S\) as above, we have:
\[
X = U S V^\top
\]
Where: 
- Columns of \(U\) are left singular vectors  
- Columns of \(V\) are right singular vectors  
- Diagonal entries of \(S\) are the singular values

---

## 4) Geometry of SVD

Think of \(X\) as a linear map:
\[
X: \mathbb{R}^d \to \mathbb{R}^n.
\]

The SVD shows that \(X\) acts by:
1. Rotating to align with axes (via \(V^\top\)) 
2. Stretching/squashing along those axes (by factors \(s_j\))
3. Rotating again (via \(U\)) into the output space

So SVD decomposes any linear transformation into “rotate, then scale, and rotate”.

---

## 5) Relation to Eigen-decomposition

From \(X = U S V^\top\), we can form:

\[
X^\top X = V S^\top S V^\top = V \,\text{diag}(s_1^2,\dots,s_r^2)\, V^\top
\]

Thus:
- Right singular vectors of \(X\) (columns of \(V\)) are eigenvectors of \(X^\top X\).
- Squared singular values \(s_j^2\) are the eigenvalues of \(X^\top X\).

This connects SVD directly to PCA:  
- \(\Sigma = \frac{1}{n} X^\top X\) has eigenvalues \(\lambda_j = s_j^2 / n\)

---

## 6) Reduced (Thin) SVD

If rank\((X) = r\), then only the first \(r\) singular values are non-zero.  

We can write the **thin SVD**:
\[
X = U_{[:,1:r]} \, S_{1:r,1:r} \, V_{[:,1:r]}^\top
\]

Where: 
- \(U_{[:,1:r]} \in \mathbb{R}^{n \times r}\)
- \(S_{1:r,1:r} \in \mathbb{R}^{r \times r}\)
- \(V_{[:,1:r]} \in \mathbb{R}^{d \times r}\)

This is computationally easier to compute and is used in PCA algorithms. 

---

## 7) Eckart–Young Theorem

Let \(X \in \mathbb{R}^{n \times d}\) with singular values
\[
s_1 \ge s_2 \ge \cdots \ge s_r > 0, \quad r = \mathrm{rank}(X)
\]

For any \(k < r\), define
\[
X_k = U_{[:,1:k]} \, S_{1:k,1:k} \, V_{[:,1:k]}^\top
\]
As the truncation of the SVD keeping only the first \(k\) singular values/vectors, also called thin SVD.

Then for every matrix \(Y \in \mathbb{R}^{n \times d}\) with \(\mathrm{rank}(Y) \le k\), we have the inequality under the Frobenius norm: 
\[
\|X - X_k\|_F \;\le\; \|X - Y\|_F
\]

Where:
\[
\|X - X_k\|_F = \left( \sum_{j=k+1}^r s_j^2 \right)^{1/2}
\]

---

## 8) PCA via Thin SVD

Let \(X \in \mathbb{R}^{n \times d}\) be the **centred** data matrix. 
Let \(r = \mathrm{rank}(X)\)

Compute its thin SVD:
\[
X = U_r S_r V_r^\top
\]
With: 
- \(U_r \in \mathbb{R}^{n \times r}\)
- \(S_r \in \mathbb{R}^{r \times r}\)
- \(V_r \in \mathbb{R}^{d \times r}\)

Recall the sample covariance matrix:
\[
\Sigma = \tfrac{1}{n} X^\top X
\]

Substitute the SVD:
\[
\Sigma = \tfrac{1}{n} V_r \, S_r^2 V_r^\top
\]

Thus:
- Columns of \(V_r\) are eigenvectors of \(\Sigma\), called the **principal components**
- Eigenvalues are \(\lambda_j = \tfrac{s_j^2}{n}\)
- Variance explained by component \(j\) is proportional to \(s_j^2\)

---

## 9) Encoding (Projection)

We want to represent each sample \(x_i \in \mathbb{R}^d\) in a lower-dimensional space.

**Step 1. Order by variance**  
From the SVD, the eigenvalues of the covariance matrix are
\[
\lambda_j = \tfrac{1}{n} s_j^2, \quad j=1,\dots,r
\]
- Larger \(\lambda_j\) means component \(v_j\) explains more variance in the data.
- Hence, we rank the directions by decreasing eigenvalues \(\lambda_j\) (equivalently, by decreasing singular values \(s_j\))

**Step 2. Select top \(k\) principal components**  
Pick the first \(k\) right singular vectors:
\[
V_k = [v_1, \dots, v_k] \in \mathbb{R}^{d \times k}
\]
Where:  \(s_1^2 \ge s_2^2 \ge \cdots \ge s_k^2\)

**Step 3. Encode samples**  
Each centred data point column vector \(x_i\) is projected onto this basis:
\[
z_i = V_k^\top x_i \;\;\in \mathbb{R}^k
\]
For the entire dataset (stacking rows of \(X\)):
\[
Z = X V_k \;\;\in \mathbb{R}^{n \times k}
\]

- \(z_i\) are the **PCA coordinates** of sample \(x_i\)

Therefore, \(Z\) is the reduced representation of all \(n\) samples.

---

## 10) Reconstruction (Decoding)

Given \(z_i\), reconstruct in the original feature space:
\[
\hat{x}_i = V_k z_i \;\in \mathbb{R}^d
\]

For all samples:
\[
\hat{X} = Z V_k^\top = X V_k V_k^\top
\]

This is the **orthogonal projection** of \(X\) onto the span of the first \(k\) principal components.

---

## 11) Optimality

By Eckart–Young:
\[
\|X - \hat{X}\|_F^2 = \sum_{j=k+1}^r s_j^2
\]

So:
- Keeping the first \(k\) singular vectors minimises the total reconstruction error.  
- No other \(k\)-dimensional subspace achieves a smaller error. 
