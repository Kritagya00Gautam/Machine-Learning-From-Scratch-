# Basis and Rank

In the previous section we asked whether a few vectors can build **every**
vector in a space. This section answers it:

- A **generating set** can build every vector.
- A **basis** is a generating set with **no redundancy**, the smallest possible
  set of building blocks.
- The **dimension** is how many building blocks a basis has.
- The **rank** of a matrix counts how many independent directions its columns
  (or rows) really contain.

> Based on Section 2.6 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> examples here are written for these notes.

---

## 1. Generating set and span

**Definition.** Let $V$ be a vector space and $\mathcal{A} = \{\mathbf{x}_1, \dots, \mathbf{x}_k\} \subseteq V$.

- If **every** $\mathbf{v} \in V$ can be written as a linear combination of
  $\mathbf{x}_1, \dots, \mathbf{x}_k$, then $\mathcal{A}$ is a **generating set** of $V$.
- The set of **all** linear combinations of the vectors in $\mathcal{A}$ is the
  **span** of $\mathcal{A}$. If $\mathcal{A}$ generates $V$, we write

$$
V = \operatorname{span}[\mathcal{A}] = \operatorname{span}[\mathbf{x}_1, \dots, \mathbf{x}_k]
$$

**Intuition: everywhere you can reach.** Imagine you can only move along the
directions $\mathbf{x}_1, \dots, \mathbf{x}_k$, any distance forwards or backwards.
The span is every place you can reach.

| Vectors in $\mathbb{R}^3$ | Their span |
|---|---|
| $[1, 0, 0]^\top$ | the $x$-axis (a line) |
| $[1, 0, 0]^\top, [0, 1, 0]^\top$ | the $xy$-plane |
| $[1, 0, 0]^\top, [2, 0, 0]^\top$ | still just the $x$-axis: the second adds nothing |
| $[1, 0, 0]^\top, [0, 1, 0]^\top, [0, 0, 1]^\top$ | all of $\mathbb{R}^3$ |

A span is always a **subspace**: it contains $\mathbf{0}$, and sums and multiples
of linear combinations are again linear combinations.

---

## 2. Basis

A generating set may contain redundant vectors. A basis is a generating set with
the redundancy removed.

**Definition.** A generating set $\mathcal{A}$ of $V$ is **minimal** if no
smaller subset of $\mathcal{A}$ still spans $V$. Every **linearly independent**
generating set is minimal, and is called a **basis** of $V$.

### Four ways to say "basis"

For a vector space $V$ and a nonempty set $\mathcal{B} \subseteq V$, these
statements mean the same thing:

1. $\mathcal{B}$ is a **basis** of $V$.
2. $\mathcal{B}$ is a **minimal generating set**: remove any vector and it no
   longer spans $V$.
3. $\mathcal{B}$ is a **maximal linearly independent set**: add any other vector
   from $V$ and it becomes dependent.
4. Every $\mathbf{x} \in V$ is a linear combination of vectors in $\mathcal{B}$,
   and that combination is **unique**. If
   $\mathbf{x} = \sum \lambda_i \mathbf{b}_i = \sum \psi_i \mathbf{b}_i$, then
   $\lambda_i = \psi_i$ for all $i$.

So a basis sits exactly in the sweet spot: **enough** vectors to reach
everything, and **no more** than needed.

**Analogy: an alphabet.** A basis is like an alphabet for a vector space. Every
"word" (vector) can be spelled with it, each word has exactly **one** spelling,
and no letter can be dropped without losing words.

### Examples

**Standard (canonical) basis of $\mathbb{R}^3$:**

$$
\mathcal{B} = \left\{
\begin{bmatrix}1\\0\\0\end{bmatrix},
\begin{bmatrix}0\\1\\0\end{bmatrix},
\begin{bmatrix}0\\0\\1\end{bmatrix}
\right\}
$$

**Bases are not unique.** For $\mathbb{R}^2$, both
$\left\{\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}0\\1\end{bmatrix}\right\}$
and
$\left\{\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}1\\1\end{bmatrix}\right\}$
are bases. The same vector simply gets different "spellings":

$$
\begin{bmatrix}3\\2\end{bmatrix}
= 3\begin{bmatrix}1\\0\end{bmatrix} + 2\begin{bmatrix}0\\1\end{bmatrix}
= 1\begin{bmatrix}1\\0\end{bmatrix} + 2\begin{bmatrix}1\\1\end{bmatrix}
$$

**Independent but not a basis.**
$\left\{[1,0,0]^\top, [0,1,0]^\top\right\}$ is independent, but it doesn't span
$\mathbb{R}^3$: you can never reach $[0, 0, 1]^\top$. Too few vectors.

**Spanning but not a basis.**
$\left\{[1,0]^\top, [0,1]^\top, [1,1]^\top\right\}$ spans $\mathbb{R}^2$, but it is
dependent. Too many vectors.

---

## 3. Dimension

**Every vector space has a basis.** There are many different bases, but **all
bases of a space have the same number of vectors**. That number is the
**dimension**:

$$
\dim(V) = \text{number of basis vectors of } V
$$

(These notes only consider finite-dimensional spaces.)

**Intuition:** the dimension is the number of **independent directions** in the
space. It counts how many separate numbers you need to describe a position in it.

| Space | Dimension |
|---|---|
| a line through the origin | 1 |
| a plane through the origin | 2 |
| $\mathbb{R}^n$ | $n$ |
| $\mathbb{R}^{m \times n}$ | $mn$ |
| $\{\mathbf{0}\}$ | 0 |

**Subspaces.** If $U \subseteq V$ is a subspace, then

$$
\dim(U) \leq \dim(V), \qquad \text{with } \dim(U) = \dim(V) \iff U = V
$$

> **Dimension ≠ number of entries.** $U = \operatorname{span}\left[\begin{bmatrix}1\\1\end{bmatrix}\right]$
> lives in $\mathbb{R}^2$ and its vectors have 2 entries, but $U$ is a line, so
> $\dim(U) = 1$.

---

## 4. Finding a basis of a span

Given $U = \operatorname{span}[\mathbf{x}_1, \dots, \mathbf{x}_m] \subseteq \mathbb{R}^n$:

1. Write $\mathbf{x}_1, \dots, \mathbf{x}_m$ as the **columns** of a matrix $A$.
2. Bring $A$ to **row-echelon form**.
3. The **original** vectors belonging to the **pivot columns** form a basis of $U$.

### Example

Find a basis of
$U = \operatorname{span}\left[
\begin{bmatrix}1\\2\\1\end{bmatrix},
\begin{bmatrix}2\\4\\2\end{bmatrix},
\begin{bmatrix}0\\1\\1\end{bmatrix},
\begin{bmatrix}1\\3\\2\end{bmatrix}
\right] \subseteq \mathbb{R}^3$.

$$
\begin{bmatrix} 1 & 2 & 0 & 1 \\ 2 & 4 & 1 & 3 \\ 1 & 2 & 1 & 2 \end{bmatrix}
\begin{array}{l} \\ -2R_1 \\ -R_1 \end{array}
\rightsquigarrow
\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix}
\begin{array}{l} \\ \\ -R_2 \end{array}
\rightsquigarrow
\begin{bmatrix} \boxed{1} & 2 & 0 & 1 \\ 0 & 0 & \boxed{1} & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

Pivots are in columns 1 and 3, so a basis is

$$
\left\{ \begin{bmatrix}1\\2\\1\end{bmatrix}, \begin{bmatrix}0\\1\\1\end{bmatrix} \right\},
\qquad \dim(U) = 2
$$

The other two vectors were redundant: column 2 $= 2 \times$ column 1, and
column 4 $=$ column 1 $+$ column 3. Four vectors were given, but they only fill
a **plane**.

> Use the **original** columns, not the row-reduced ones. Row operations
> change the columns themselves, but keep the *relationships* between them.

---

## 5. Rank

**Definition.** The **rank** of $A \in \mathbb{R}^{m \times n}$, written
$\operatorname{rk}(A)$, is the number of linearly independent **columns** of $A$.
Remarkably, this always equals the number of linearly independent **rows**.

**How to compute it:** row-reduce $A$ to REF and **count the pivots**.

**Intuition: the real amount of information.** A matrix may have many columns,
but the rank tells you how many of them are **genuinely different directions**.
Picture a spreadsheet with 10 columns. If 7 of them are just sums or multiples
of the other 3, the rank is 3. Only 3 columns carry new information.

### Examples

$$
\operatorname{rk}\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = 1
\quad\text{(row 2 = 2 × row 1)},
\qquad
\operatorname{rk}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 2
$$

The $3 \times 4$ matrix in Section 4 has 2 pivots, so its rank is 2.

### Properties of the rank

| Property | Meaning |
|---|---|
| $\operatorname{rk}(A) = \operatorname{rk}(A^\top)$ | column rank = row rank |
| The columns of $A$ span a subspace $U \subseteq \mathbb{R}^m$ with $\dim(U) = \operatorname{rk}(A)$ | this is the **column space** (later: **image** or **range**). Its basis is the pivot columns |
| The rows of $A$ span a subspace $W \subseteq \mathbb{R}^n$ with $\dim(W) = \operatorname{rk}(A)$ | the **row space**. A basis comes from Gaussian elimination on $A^\top$ |
| $A \in \mathbb{R}^{n \times n}$ is invertible $\iff \operatorname{rk}(A) = n$ | square matrices are invertible exactly when they have full rank |
| $A\mathbf{x} = \mathbf{b}$ is solvable $\iff \operatorname{rk}(A) = \operatorname{rk}([A \mid \mathbf{b}])$ | adding $\mathbf{b}$ as a column must not add a new independent direction, so $\mathbf{b}$ is reachable from the columns |
| Solutions of $A\mathbf{x} = \mathbf{0}$ form a subspace of dimension $n - \operatorname{rk}(A)$ | the **kernel** / **null space**: one free direction per free variable |
| $\operatorname{rk}(A) \leq \min(m, n)$ | you can't have more independent columns than columns, or than rows |

**Full rank and rank deficient.** $A \in \mathbb{R}^{m \times n}$ has **full rank**
if $\operatorname{rk}(A) = \min(m, n)$, the largest possible. Otherwise it is
**rank deficient**: some information is redundant.

**Solvability example.** For
$A = \begin{bmatrix}1 & 2\\2 & 4\end{bmatrix}$ (rank 1):

- $\mathbf{b} = [3, 6]^\top$: $[A \mid \mathbf{b}]$ still has rank 1, so the system is **solvable**.
- $\mathbf{b} = [3, 7]^\top$: $[A \mid \mathbf{b}]$ has rank 2, so there is **no solution**.
  $\mathbf{b}$ is not on the line spanned by the columns.

---

## 6. Why this matters in machine learning

| ML situation | Connection |
|---|---|
| Choosing features | a basis is the smallest set of features that loses no information |
| Dimensionality reduction (PCA) | choose a new basis in which only a few directions matter |
| Low-rank matrices (recommender systems) | a huge user × movie rating matrix is well approximated by a low-rank one: a few hidden "taste" factors explain everything |
| Linear regression | the normal equations have a unique solution when $X$ has full column rank |
| Embeddings | each coordinate is the weight on a basis direction |

---

## Summary

- The **span** of a set is everything you can build from it. A **generating set**
  spans the whole space.
- A **basis** is an **independent generating set**: a minimal generating set,
  a maximal independent set, and it gives every vector a **unique** representation.
- Bases are not unique, but they all have the same size, the **dimension**.
- To find a basis of a span: put the vectors in columns, row-reduce, and keep
  the original vectors at the pivot columns.
- **Rank** = number of pivots = number of independent columns = number of
  independent rows.
- $n \times n$ matrices are invertible exactly when $\operatorname{rk}(A) = n$.
  $A\mathbf{x} = \mathbf{b}$ is solvable exactly when
  $\operatorname{rk}(A) = \operatorname{rk}([A \mid \mathbf{b}])$.

**Previous:** [Linear Independence](../05-linear%20independence/) &nbsp;·&nbsp;
**Next:** [Linear Mappings](../07-Linear%20mapping/)
