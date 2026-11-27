# Linear Mappings

So far we have studied vector spaces and their structure. Now we study
**functions between vector spaces that respect that structure**. These are
**linear mappings**, and they are exactly what matrices really *are*.

> Based on Section 2.7 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> examples here are written for these notes.

---

## 1. What is a linear mapping?

**Definition.** For vector spaces $V$ and $W$, a mapping $\Phi : V \to W$ is a
**linear mapping** (also called a *linear transformation* or *vector space
homomorphism*) if

$$
\forall \mathbf{x}, \mathbf{y} \in V,\; \forall \lambda, \psi \in \mathbb{R}:\quad
\Phi(\lambda\mathbf{x} + \psi\mathbf{y}) = \lambda\Phi(\mathbf{x}) + \psi\Phi(\mathbf{y})
$$

That one rule packs together two simpler ones:

- **Additivity:** $\Phi(\mathbf{x} + \mathbf{y}) = \Phi(\mathbf{x}) + \Phi(\mathbf{y})$
- **Homogeneity:** $\Phi(\lambda\mathbf{x}) = \lambda\Phi(\mathbf{x})$

In words: **it doesn't matter whether you add and scale first, then map, or map
first, then add and scale.** The result is the same.

### Intuition: what linear maps look like

Picture a sheet of graph paper being transformed. A linear map:

- keeps the **origin fixed**, since $\Phi(\mathbf{0}) = \mathbf{0}$ always,
- keeps **straight lines straight**,
- keeps **grid lines parallel and evenly spaced**.

It may stretch, rotate, reflect, shear or flatten the sheet, but it never bends
it and never slides it away from the origin.

### Examples and non-examples (on $\mathbb{R}^2$)

| Mapping | Linear? | Why |
|---|---|---|
| Scaling: $\Phi(\mathbf{x}) = 3\mathbf{x}$ | ✅ | stretches everything equally |
| Rotation by $90°$: $\Phi\left(\begin{bmatrix}x_1\\x_2\end{bmatrix}\right) = \begin{bmatrix}-x_2\\x_1\end{bmatrix}$ | ✅ | turns the sheet |
| Projection onto the $x$-axis: $\Phi\left(\begin{bmatrix}x_1\\x_2\end{bmatrix}\right) = \begin{bmatrix}x_1\\0\end{bmatrix}$ | ✅ | like a shadow when light shines straight down |
| Any matrix: $\Phi(\mathbf{x}) = A\mathbf{x}$ | ✅ | $A(\lambda\mathbf{x} + \psi\mathbf{y}) = \lambda A\mathbf{x} + \psi A\mathbf{y}$ |
| Shift: $\Phi(\mathbf{x}) = \mathbf{x} + \begin{bmatrix}1\\0\end{bmatrix}$ | ❌ | $\Phi(\mathbf{0}) \neq \mathbf{0}$ (this is *affine*, see the next section) |
| Squaring: $\Phi\left(\begin{bmatrix}x_1\\x_2\end{bmatrix}\right) = \begin{bmatrix}x_1^2\\x_2\end{bmatrix}$ | ❌ | $\Phi(2\mathbf{x}) \neq 2\Phi(\mathbf{x})$: it bends lines |

> **Two meanings of a matrix.** A matrix can store a **collection of vectors**
> (its columns) or represent a **linear mapping**. Always ask which one you're
> looking at.

---

## 2. Injective, surjective, bijective

These terms apply to any mapping $\Phi : \mathcal{V} \to \mathcal{W}$ between sets:

| Property | Definition | Intuition |
|---|---|---|
| **Injective** (one-to-one) | $\Phi(\mathbf{x}) = \Phi(\mathbf{y}) \Rightarrow \mathbf{x} = \mathbf{y}$ | no two inputs land on the same output: nothing gets merged |
| **Surjective** (onto) | $\Phi(\mathcal{V}) = \mathcal{W}$ | every output is reached by some input |
| **Bijective** | injective **and** surjective | a perfect pairing: every output comes from exactly one input |

A bijective $\Phi$ can be **undone**: there is an inverse
$\Phi^{-1} : \mathcal{W} \to \mathcal{V}$ with $\Phi^{-1}(\Phi(\mathbf{x})) = \mathbf{x}$.

**Analogy: coat check.** Each coat gets a ticket. *Injective* means no two coats
share a ticket. *Surjective* means every ticket has a coat. *Bijective* means
both, so every ticket can be traded back for exactly one coat.

The projection onto the $x$-axis above is **not injective**: $[1, 5]^\top$ and
$[1, -2]^\top$ both map to $[1, 0]^\top$. The rotation **is bijective**: rotate
back by $-90°$ to undo it.

### Special names for linear mappings

| Name | Meaning |
|---|---|
| **Isomorphism** | $\Phi : V \to W$, linear and bijective |
| **Endomorphism** | $\Phi : V \to V$, linear (maps a space into itself) |
| **Automorphism** | $\Phi : V \to V$, linear and bijective |
| **Identity** $\operatorname{id}_V$ | $\operatorname{id}_V(\mathbf{x}) = \mathbf{x}$ |

**Example: $\mathbb{R}^2$ and $\mathbb{C}$.** The map
$\Phi\left(\begin{bmatrix}x_1\\x_2\end{bmatrix}\right) = x_1 + i\,x_2$ is linear:
adding pairs in $\mathbb{R}^2$ matches adding complex numbers, and scaling
matches too. This is why complex numbers can be drawn as points in the plane.

### Same dimension means "the same space"

**Theorem.** Finite-dimensional vector spaces $V$ and $W$ are isomorphic (there
is a linear bijection between them) **if and only if** $\dim(V) = \dim(W)$.

**Intuition:** spaces of the same dimension are "the same thing in different
clothing". You can translate between them without losing anything. This is why
we can treat an $m \times n$ matrix as a vector of length $mn$ (flattening an
image, for example): $\mathbb{R}^{m \times n}$ and $\mathbb{R}^{mn}$ are both
$mn$-dimensional.

**More facts.**

- If $\Phi : V \to W$ and $\Psi : W \to X$ are linear, so is their composition
  $\Psi \circ \Phi : V \to X$ ("do $\Phi$, then $\Psi$").
- If $\Phi$ is an isomorphism, so is $\Phi^{-1}$.
- Sums and scalar multiples of linear maps $V \to W$ are again linear.

---

## 3. Coordinates

To turn a linear map into a matrix, we first need **coordinates**.

**Ordered basis.** A basis where the **order** matters is written with round
brackets: $B = (\mathbf{b}_1, \dots, \mathbf{b}_n)$. The bases
$(\mathbf{b}_1, \mathbf{b}_2)$ and $(\mathbf{b}_2, \mathbf{b}_1)$ are different
ordered bases.

**Definition (Coordinates).** Let $B = (\mathbf{b}_1, \dots, \mathbf{b}_n)$ be
an ordered basis of $V$. Every $\mathbf{x} \in V$ has a unique representation

$$
\mathbf{x} = \alpha_1\mathbf{b}_1 + \cdots + \alpha_n\mathbf{b}_n
$$

The numbers $\alpha_1, \dots, \alpha_n$ are the **coordinates** of $\mathbf{x}$
with respect to $B$, and

$$
\boldsymbol{\alpha} = \begin{bmatrix}\alpha_1\\ \vdots \\ \alpha_n\end{bmatrix} \in \mathbb{R}^n
$$

is the **coordinate vector**.

**Intuition: the same place, different directions.** "The library is 3 blocks
east and 2 blocks north" and "the library is 1 block east, then 2 blocks
north-east" describe the **same** place in two different coordinate systems.
The vector doesn't change. Only its description does.

**Example.** For $\mathbf{x} = [3, 2]^\top$:

- In the standard basis $(\mathbf{e}_1, \mathbf{e}_2)$, the coordinates are $[3, 2]^\top$.
- In $B = \left(\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}1\\1\end{bmatrix}\right)$,
  we have $\mathbf{x} = 1\begin{bmatrix}1\\0\end{bmatrix} + 2\begin{bmatrix}1\\1\end{bmatrix}$,
  so the coordinates are $[1, 2]^\top$.

Choosing a basis gives an isomorphism between any $n$-dimensional space $V$ and
$\mathbb{R}^n$. That's why we can always work with plain columns of numbers.

---

## 4. The transformation matrix

**Definition.** Let $\Phi : V \to W$ be linear, with ordered bases
$B = (\mathbf{b}_1, \dots, \mathbf{b}_n)$ of $V$ and
$C = (\mathbf{c}_1, \dots, \mathbf{c}_m)$ of $W$. Write the image of each basis
vector of $V$ in the basis $C$:

$$
\Phi(\mathbf{b}_j) = \alpha_{1j}\mathbf{c}_1 + \cdots + \alpha_{mj}\mathbf{c}_m,
\qquad j = 1, \dots, n
$$

The $m \times n$ matrix $A_\Phi$ with entries $A_\Phi(i, j) = \alpha_{ij}$ is the
**transformation matrix** of $\Phi$ with respect to $B$ and $C$.

> **The key recipe:** **column $j$ of $A_\Phi$ = the coordinates of
> $\Phi(\mathbf{b}_j)$.** "Where does each basis vector go?" Write the answers
> down as columns, and you have the matrix.

If $\hat{\mathbf{x}}$ is the coordinate vector of $\mathbf{x}$ in $B$, and
$\hat{\mathbf{y}}$ is the coordinate vector of $\Phi(\mathbf{x})$ in $C$, then

$$
\hat{\mathbf{y}} = A_\Phi\,\hat{\mathbf{x}}
$$

**Why this works:** a linear map is completely decided by what it does to a
basis. Every $\mathbf{x}$ is a combination of basis vectors, and linearity
carries that combination straight through $\Phi$.

### Example: rotation by $90°$

Using the standard basis, where do $\mathbf{e}_1$ and $\mathbf{e}_2$ go?

$$
\Phi(\mathbf{e}_1) = \begin{bmatrix}0\\1\end{bmatrix}, \qquad
\Phi(\mathbf{e}_2) = \begin{bmatrix}-1\\0\end{bmatrix}
\qquad\Longrightarrow\qquad
A_\Phi = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

Check with $\mathbf{x} = [2, 1]^\top$:
$A_\Phi\mathbf{x} = [-1, 2]^\top$, which is $[2, 1]^\top$ turned a quarter-turn
anticlockwise. ✓

### Composition is matrix multiplication

If $\Phi : V \to W$ has matrix $A_\Phi$ and $\Psi : W \to X$ has matrix $A_\Psi$,
then "first $\Phi$, then $\Psi$" has matrix

$$
A_{\Psi \circ \Phi} = A_\Psi A_\Phi
$$

The matrix of the map applied **first** sits on the **right**, next to the
vector. This is the real reason matrix multiplication is defined the way it is,
and why $AB \neq BA$: rotating then projecting is not the same as projecting
then rotating.

---

## 5. Basis change

A linear map stays the same map, but its **matrix depends on the bases** we
choose. How does the matrix change when we change basis?

**Setup.** $\Phi : V \to W$ has matrix $A$ with respect to bases $B$ (of $V$)
and $C$ (of $W$). We switch to new bases $\tilde{B}$ and $\tilde{C}$.

- Let $S \in \mathbb{R}^{n \times n}$ be the matrix whose columns are the
  **new basis vectors $\tilde{\mathbf{b}}_j$ written in the old basis $B$**.
- Let $T \in \mathbb{R}^{m \times m}$ be the matrix whose columns are the new
  basis vectors $\tilde{\mathbf{c}}_k$ written in the old basis $C$.

**Theorem (Basis change).** The matrix of $\Phi$ with respect to $\tilde{B}$
and $\tilde{C}$ is

$$
\tilde{A} = T^{-1} A S
$$

**Read it right to left:** $S$ converts new $V$-coordinates into old ones, $A$
applies the map in the old bases, and $T^{-1}$ converts the result from old
$W$-coordinates into new ones.

### Example: the right basis makes a map simple

Let $\Phi : \mathbb{R}^2 \to \mathbb{R}^2$ have matrix
$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ in the standard basis. Change
to the new basis $\tilde{B} = \left(\begin{bmatrix}1\\1\end{bmatrix}, \begin{bmatrix}1\\-1\end{bmatrix}\right)$
(the same basis for input and output, so $T = S$):

$$
S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad
S^{-1} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

$$
AS = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix}, \qquad
\tilde{A} = S^{-1}AS = \frac{1}{2}\begin{bmatrix} 6 & 0 \\ 0 & 2 \end{bmatrix}
= \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

In the new basis the map is **diagonal**: it simply stretches the
$[1, 1]^\top$ direction by $3$ and leaves the $[1, -1]^\top$ direction alone.
(Check: $A[1,1]^\top = [3,3]^\top$ and $A[1,-1]^\top = [1,-1]^\top$. ✓)
Finding such "natural" bases is the idea behind **eigenvectors** in matrix
decompositions.

### Equivalent and similar matrices

- $A, \tilde{A} \in \mathbb{R}^{m \times n}$ are **equivalent** if
  $\tilde{A} = T^{-1}AS$ for some invertible $S$ and $T$. They describe the same
  map in different bases.
- $A, \tilde{A} \in \mathbb{R}^{n \times n}$ are **similar** if
  $\tilde{A} = S^{-1}AS$ for some invertible $S$. This is the same idea for maps
  from a space to itself, with the same basis change on both sides.
- Similar matrices are always equivalent, but not the other way round.

---

## 6. Image and kernel

Two subspaces describe what a linear map does.

**Definition.** For a linear map $\Phi : V \to W$:

$$
\ker(\Phi) := \{\mathbf{v} \in V : \Phi(\mathbf{v}) = \mathbf{0}_W\}
\qquad\text{(kernel / null space)}
$$

$$
\operatorname{Im}(\Phi) := \{\mathbf{w} \in W : \exists\,\mathbf{v} \in V,\; \Phi(\mathbf{v}) = \mathbf{w}\}
\qquad\text{(image / range)}
$$

$V$ is called the **domain** and $W$ the **codomain**.

**Intuition: the shadow.** Shine a light straight down onto the floor. The
projection takes each point in the room to its shadow.

- The **image** is the floor: every place a shadow can land.
- The **kernel** is the vertical line through the origin: everything squashed
  into the single point $\mathbf{0}$. Moving a point up or down doesn't change
  its shadow, so that information is **lost**.

### Properties

- $\Phi(\mathbf{0}_V) = \mathbf{0}_W$ always, so $\mathbf{0}_V \in \ker(\Phi)$
  and the kernel is never empty.
- $\ker(\Phi)$ is a subspace of $V$, and $\operatorname{Im}(\Phi)$ is a subspace of $W$.
- **$\Phi$ is injective $\iff \ker(\Phi) = \{\mathbf{0}\}$.** If only $\mathbf{0}$
  is squashed to zero, no information is lost.

### For a matrix $A \in \mathbb{R}^{m \times n}$, $\Phi(\mathbf{x}) = A\mathbf{x}$

With columns $A = [\mathbf{a}_1, \dots, \mathbf{a}_n]$:

- **Image = column space:**
  $\operatorname{Im}(\Phi) = \{A\mathbf{x}\} = \operatorname{span}[\mathbf{a}_1, \dots, \mathbf{a}_n] \subseteq \mathbb{R}^m$,
  and $\dim(\operatorname{Im}(\Phi)) = \operatorname{rk}(A)$.
- **Kernel = null space:** the solutions of the homogeneous system
  $A\mathbf{x} = \mathbf{0}$, a subspace of $\mathbb{R}^n$. It captures every
  combination of columns that produces $\mathbf{0}$, which are exactly the
  column dependencies.

### Example

$$
A = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix},
\qquad \Phi : \mathbb{R}^3 \to \mathbb{R}^2
$$

- **Image:** columns 1 and 3 are $\mathbf{e}_1$ and $\mathbf{e}_2$, so the image
  is all of $\mathbb{R}^2$ and $\operatorname{rk}(A) = 2$. So $\Phi$ is surjective.
- **Kernel:** $A$ is in RREF with column 2 free. The Minus-1 Trick gives
  $\ker(\Phi) = \operatorname{span}\left[[2, -1, 0]^\top\right]$, a line of dimension 1.
  So $\Phi$ is **not** injective.

---

## 7. The rank-nullity theorem

**Theorem (Rank-nullity, the fundamental theorem of linear mappings).** For a
linear map $\Phi : V \to W$:

$$
\dim(\ker(\Phi)) + \dim(\operatorname{Im}(\Phi)) = \dim(V)
$$

For a matrix $A \in \mathbb{R}^{m \times n}$ this reads
$\dim(\text{null space}) + \operatorname{rk}(A) = n$.

**Intuition: nothing disappears without a trace.** Every input dimension
either **survives** into the image or is **crushed** into the kernel. In the
example above, $1 + 2 = 3$. ✓ In the shadow example, $1$ (vertical) $+ 2$
(floor) $= 3$ (the room).

### Consequences

- If $\dim(\operatorname{Im}(\Phi)) < \dim(V)$, the kernel is non-trivial,
  $\dim(\ker(\Phi)) \geq 1$. In matrix terms, $A\mathbf{x} = \mathbf{0}$ has
  **infinitely many** solutions.
- If $\dim(V) = \dim(W)$, then
  $$
  \Phi \text{ injective} \iff \Phi \text{ surjective} \iff \Phi \text{ bijective}
  $$
  For square matrices, "no information lost", "every output reachable" and
  "invertible" are all the same condition.

---

## 8. Why this matters in machine learning

| ML situation | Connection |
|---|---|
| A neural-network layer $W\mathbf{x}$ | a linear map from input features to hidden features (plus a bias, which makes it affine) |
| Deep networks | composition of layers: the matrices multiply |
| PCA | a change of basis to directions of maximum variance, then projection |
| Projection and compression | the kernel is the information thrown away |
| Linear regression | predictions $X\boldsymbol{\theta}$ live in the column space of $X$ |

---

## Summary

- A **linear map** preserves addition and scaling:
  $\Phi(\lambda\mathbf{x} + \psi\mathbf{y}) = \lambda\Phi(\mathbf{x}) + \psi\Phi(\mathbf{y})$.
  It fixes the origin and keeps grid lines straight, parallel and evenly spaced.
- **Injective**: nothing is merged. **Surjective**: everything is reached.
  **Bijective**: both, and the map can be undone.
- Spaces of equal (finite) dimension are **isomorphic**.
- **Coordinates** describe a vector relative to an ordered basis.
- The **transformation matrix** has column $j$ = coordinates of $\Phi(\mathbf{b}_j)$.
  Composition of maps = matrix multiplication.
- **Basis change:** $\tilde{A} = T^{-1}AS$. A good basis can make a map diagonal.
- **Kernel** = what maps to $\mathbf{0}$ (lost information). **Image** = what can
  be reached (the column space). **Rank-nullity:** $\dim\ker + \dim\operatorname{Im} = \dim V$.

**Previous:** [Basis and Rank](../06-basis%20and%20rank/) &nbsp;·&nbsp;
**Next:** [Affine Spaces](../08-affine%20spaces/)
