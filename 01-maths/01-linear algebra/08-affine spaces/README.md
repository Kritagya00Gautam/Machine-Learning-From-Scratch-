# Affine Spaces

Every vector subspace passes through the **origin**. But many shapes we care
about don't: the line $y = 2x + 1$, a plane floating above the floor, the
solution set of $A\mathbf{x} = \mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$.
These are **affine spaces**, subspaces that have been **shifted away from the
origin**. This section also covers the matching **affine mappings**, which are
linear maps plus a shift.

> In machine learning texts the words "linear" and "affine" are often used
> loosely, and affine things are frequently called "linear". Here we keep them
> separate.

> Based on Section 2.8 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> examples here are written for these notes.

---

## 1. Affine subspaces

**Definition.** Let $V$ be a vector space, $\mathbf{x}_0 \in V$, and $U \subseteq V$
a subspace. The set

$$
L = \mathbf{x}_0 + U := \{\mathbf{x}_0 + \mathbf{u} : \mathbf{u} \in U\} \subseteq V
$$

is an **affine subspace** (or **linear manifold**) of $V$.

- $U$ is the **direction space**. It gives the shape and orientation.
- $\mathbf{x}_0$ is the **support point**. It says where the shape is placed.

**Intuition: sliding a subspace.** Take a line through the origin, which is a
subspace, and **slide** it without turning it until it passes through
$\mathbf{x}_0$. The result is an affine subspace. It points the same way as
before, but it's in a new place.

**Analogy: train tracks.** Two parallel train tracks have the same direction
(the same $U$) but different positions (different $\mathbf{x}_0$). Only a track
that runs through the origin is a vector subspace.

### Affine is not a vector subspace (usually)

If $\mathbf{x}_0 \notin U$, then $\mathbf{0} \notin L$, so $L$ is **not** a vector
subspace. Adding two points of $L$ also takes you off $L$: you would be adding
the shift twice.

If $\mathbf{x}_0 \in U$, then $L = U$. A vector subspace is just the special case
of an affine subspace with the shift chosen as $\mathbf{x}_0 = \mathbf{0}$.

**Examples:** points, lines and planes in $\mathbb{R}^3$ that do **not**
(necessarily) pass through the origin.

### The support point is not unique

Any point of $L$ can serve as the support point: if $\mathbf{x}_1 \in L$, then
$L = \mathbf{x}_1 + U$ as well. Just like you can describe a train track by
naming **any** station on it.

### When is one affine subspace inside another?

For $L = \mathbf{x}_0 + U$ and $\tilde{L} = \tilde{\mathbf{x}}_0 + \tilde{U}$:

$$
L \subseteq \tilde{L}
\iff
U \subseteq \tilde{U}
\;\text{ and }\;
\mathbf{x}_0 - \tilde{\mathbf{x}}_0 \in \tilde{U}
$$

The directions of $L$ must fit inside those of $\tilde{L}$, **and** you must be
able to get from $\tilde{L}$'s support point to $L$'s by moving within
$\tilde{L}$'s directions.

---

## 2. Parametric equations: lines, planes, hyperplanes

If $L = \mathbf{x}_0 + U$ is $k$-dimensional and
$(\mathbf{b}_1, \dots, \mathbf{b}_k)$ is an ordered basis of $U$, then every
$\mathbf{x} \in L$ can be written **uniquely** as

$$
\mathbf{x} = \mathbf{x}_0 + \lambda_1\mathbf{b}_1 + \cdots + \lambda_k\mathbf{b}_k,
\qquad \lambda_1, \dots, \lambda_k \in \mathbb{R}
$$

This is the **parametric equation** of $L$. The $\mathbf{b}_i$ are the
**directional vectors** and the $\lambda_i$ are the **parameters**.

**Intuition: a GPS route.** "Start at $\mathbf{x}_0$, then travel $\lambda_1$
units along $\mathbf{b}_1$ and $\lambda_2$ units along $\mathbf{b}_2$." Each
choice of parameters is one location.

| Name | Dimension | Parametric equation | Needs |
|---|---|---|---|
| **Point** | 0 | $\mathbf{x} = \mathbf{x}_0$ | just the point |
| **Line** | 1 | $\mathbf{y} = \mathbf{x}_0 + \lambda\mathbf{b}_1$ | a support point and 1 direction |
| **Plane** | 2 | $\mathbf{y} = \mathbf{x}_0 + \lambda_1\mathbf{b}_1 + \lambda_2\mathbf{b}_2$ | a support point and 2 independent directions |
| **Hyperplane** in $\mathbb{R}^n$ | $n-1$ | $\mathbf{y} = \mathbf{x}_0 + \sum_{i=1}^{n-1}\lambda_i\mathbf{b}_i$ | a support point and $n-1$ independent directions |

A **hyperplane** is one dimension less than the whole space. In $\mathbb{R}^2$
a line is a hyperplane, and in $\mathbb{R}^3$ a plane is a hyperplane.
Hyperplanes split the space into two halves, which is the key idea behind
linear classifiers such as support vector machines (Chapter 12 of the book).

### Example: a line in $\mathbb{R}^2$

The line through $\mathbf{x}_0 = [1, 2]^\top$ in direction $\mathbf{b} = [3, 1]^\top$:

$$
\mathbf{y} = \begin{bmatrix}1\\2\end{bmatrix} + \lambda\begin{bmatrix}3\\1\end{bmatrix}
$$

$\lambda = 0$ gives $[1, 2]^\top$, $\lambda = 1$ gives $[4, 3]^\top$, and
$\lambda = -1$ gives $[-2, 1]^\top$. The origin is not on this line: solving
$1 + 3\lambda = 0$ gives $\lambda = -\tfrac{1}{3}$, but then the second entry is
$2 - \tfrac{1}{3} \neq 0$. So the line is affine, not a subspace.

### Example: a plane in $\mathbb{R}^3$

The plane $x_1 + x_2 + x_3 = 1$ contains $\mathbf{x}_0 = [1, 0, 0]^\top$. Its
direction space consists of the solutions of $x_1 + x_2 + x_3 = 0$, which are
spanned by $[1, -1, 0]^\top$ and $[1, 0, -1]^\top$:

$$
\mathbf{y} = \begin{bmatrix}1\\0\\0\end{bmatrix}
+ \lambda_1\begin{bmatrix}1\\-1\\0\end{bmatrix}
+ \lambda_2\begin{bmatrix}1\\0\\-1\end{bmatrix}
$$

(Check: the entries of $\mathbf{y}$ add up to $1 + 0 + 0 = 1$ for every
$\lambda_1, \lambda_2$. ✓) It is a hyperplane in $\mathbb{R}^3$, parallel to the
plane $x_1 + x_2 + x_3 = 0$ through the origin.

---

## 3. Affine subspaces and systems of linear equations

This is where everything from the earlier sections connects.

For $A \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^m$, the
solution set of $A\mathbf{x} = \mathbf{b}$ is

- **empty**, or
- an **affine subspace of $\mathbb{R}^n$ of dimension $n - \operatorname{rk}(A)$**.

Its support point is any **particular solution**, and its direction space is
the **kernel** (the solutions of $A\mathbf{x} = \mathbf{0}$):

$$
\{\text{solutions of } A\mathbf{x} = \mathbf{b}\}
= \underbrace{\mathbf{x}_p}_{\text{support point}}
+ \underbrace{\ker(A)}_{\text{direction space}}
$$

That is exactly the recipe "general solution = particular solution + homogeneous
solutions", now seen geometrically.

**Example (from "Solving Systems of Linear Equations").** We found

$$
\mathbf{x} = \begin{bmatrix}4\\0\\1\end{bmatrix} + \lambda\begin{bmatrix}2\\-1\\0\end{bmatrix}
$$

This is a **line** in $\mathbb{R}^3$: support point $[4, 0, 1]^\top$, direction
$[2, -1, 0]^\top$. The matrix had $n = 3$ and $\operatorname{rk}(A) = 2$, so
the dimension is $3 - 2 = 1$. ✓

**More facts.**

- A single equation $a_1x_1 + \cdots + a_nx_n = b$ with not all $a_i = 0$ has a
  **hyperplane** in $\mathbb{R}^n$ as its solution set.
- Conversely, every $k$-dimensional affine subspace of $\mathbb{R}^n$ is the
  solution set of some $A\mathbf{x} = \mathbf{b}$ with $\operatorname{rk}(A) = n - k$.
- For homogeneous systems ($\mathbf{b} = \mathbf{0}$), the solution set is a
  vector subspace, which is an affine subspace with support point $\mathbf{0}$.

---

## 4. Affine mappings

Just as affine subspaces are shifted subspaces, affine mappings are **linear
mappings followed by a shift**.

**Definition.** Let $V, W$ be vector spaces, $\Phi : V \to W$ a linear mapping,
and $\mathbf{a} \in W$. The mapping

$$
\phi : V \to W, \qquad \phi(\mathbf{x}) = \mathbf{a} + \Phi(\mathbf{x})
$$

is an **affine mapping**. The vector $\mathbf{a}$ is its **translation vector**.

**Intuition.** First transform the graph paper linearly (stretch, rotate,
shear), then **slide** the whole sheet by $\mathbf{a}$. The origin no longer has
to stay put.

### Example

$$
\phi(\mathbf{x}) = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}\mathbf{x} + \begin{bmatrix}1\\-1\end{bmatrix}
$$

This stretches horizontally by 2, then shifts right by 1 and down by 1. Since
$\phi(\mathbf{0}) = [1, -1]^\top \neq \mathbf{0}$, it is **not linear**. It is
affine.

### Properties

- Every affine mapping is a **linear mapping $\Phi$ followed by a translation**
  $\tau(\mathbf{w}) = \mathbf{w} + \mathbf{a}$, so $\phi = \tau \circ \Phi$.
  Both $\Phi$ and $\tau$ are **uniquely** determined by $\phi$.
- The **composition** of affine mappings is affine: shifting and transforming,
  then doing it again, is still "a linear map plus a shift".
- **Bijective** affine mappings keep the geometric structure intact: they
  preserve **dimension** and **parallelism**. Points stay points, lines stay
  lines, and parallel lines stay parallel.

---

## 5. Why this matters in machine learning

| ML situation | Affine connection |
|---|---|
| A neural-network layer $W\mathbf{x} + \mathbf{b}$ | an **affine** mapping: linear weights plus a bias (translation) |
| Linear regression with an intercept, $y = \boldsymbol{\theta}^\top\mathbf{x} + \theta_0$ | an affine function; often loosely called "linear" |
| Classification with a decision boundary $\mathbf{w}^\top\mathbf{x} + b = 0$ | the boundary is a **hyperplane** (support vector machines, logistic regression) |
| Data normalisation $(\mathbf{x} - \boldsymbol{\mu}) / \sigma$ | an affine transformation of the data |
| Solution sets of under-determined systems | affine subspaces: many parameter settings fit the data equally well |

---

## Summary

- An **affine subspace** $L = \mathbf{x}_0 + U$ is a vector subspace $U$ shifted
  by a support point $\mathbf{x}_0$. Unless $\mathbf{x}_0 \in U$, it does not
  contain $\mathbf{0}$ and is not a vector subspace.
- **Parametric form:** $\mathbf{x} = \mathbf{x}_0 + \sum \lambda_i \mathbf{b}_i$
  describes points (0-dimensional), lines (1), planes (2) and hyperplanes ($n-1$).
- The solutions of $A\mathbf{x} = \mathbf{b}$ are empty or an affine subspace:
  particular solution + kernel, of dimension $n - \operatorname{rk}(A)$.
- An **affine mapping** is $\phi(\mathbf{x}) = \mathbf{a} + \Phi(\mathbf{x})$:
  a linear map plus a translation. Compositions of affine maps are affine, and
  bijective ones preserve dimension and parallelism.

This completes the linear algebra foundations: from systems of equations,
through matrices, vector spaces, independence, bases and linear maps, to affine
geometry.

**Previous:** [Linear Mappings](../07-Linear%20mapping/) &nbsp;·&nbsp;
**Back to:** [Linear Algebra overview](../LinearAlgebra.md)
