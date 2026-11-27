# Matrices

Matrices are the central object of linear algebra. They let us:

- write a whole **system of linear equations** as one compact equation, $A\mathbf{x} = \mathbf{b}$;
- store **data**: each row is one example, each column is one feature;
- represent **linear transformations**: functions that rotate, scale, shear or project vectors.

Almost every machine learning model, from linear regression to deep neural
networks, is built from matrix operations.

---

## 1. What is a matrix?

**Definition.** For $m, n \in \mathbb{N}$, a real-valued **$(m, n)$ matrix** $A$
is a rectangular grid of $m \cdot n$ real numbers $a_{ij}$, arranged in $m$ rows
and $n$ columns:

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix},
\qquad a_{ij} \in \mathbb{R}
$$

- $a_{ij}$ is the entry in **row $i$** and **column $j$** (row first, then column).
- $\mathbb{R}^{m \times n}$ is the set of all real $(m, n)$ matrices.
- A $(1, n)$ matrix is a **row vector**; an $(m, 1)$ matrix is a **column vector**.
  In this repository, "vector" means a column vector unless stated otherwise.

> **Shape is everything.** Before doing any matrix operation, write down the
> shapes. Most mistakes in linear algebra come from ignoring them.

### Matrices as stacked vectors

A matrix $A \in \mathbb{R}^{m \times n}$ can also be seen as a long vector
$\mathbf{a} \in \mathbb{R}^{mn}$ by stacking its columns one below the other.
This "reshaping" view is used often in practice, for example when flattening an
image into a vector before feeding it to a model.

---

## 2. Matrix addition

Two matrices can be added **only if they have the same shape**. Addition is done
entry by entry.

For $A, B \in \mathbb{R}^{m \times n}$:

$$
A + B =
\begin{bmatrix}
a_{11} + b_{11} & \cdots & a_{1n} + b_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$

**Example.**

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
+
\begin{bmatrix} 5 & 0 \\ -1 & 2 \end{bmatrix}
=
\begin{bmatrix} 6 & 2 \\ 2 & 6 \end{bmatrix}
$$

Addition is **commutative** ($A + B = B + A$) and **associative**
($(A + B) + C = A + (B + C)$). The **zero matrix** $\mathbf{0}$ (all entries
zero) acts like the number 0: $A + \mathbf{0} = A$.

---

## 3. Matrix multiplication

### The shape rule

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times k}$, the
product $C = AB$ is an $(m, k)$ matrix:

$$
\underbrace{A}_{m \times n} \;\; \underbrace{B}_{n \times k} \;=\; \underbrace{C}_{m \times k}
$$

The **inner dimensions must match** ($n$ and $n$), and the **outer dimensions
give the result's shape** ($m$ and $k$). If the inner dimensions differ, the
product is not defined.

### The entry rule

Each entry of $C$ is a sum of products:

$$
c_{ij} = \sum_{l=1}^{n} a_{il}\, b_{lj}, \qquad i = 1, \dots, m, \quad j = 1, \dots, k
$$

In words: **$c_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$.**
Walk along the row of $A$, walk down the column of $B$, multiply matching pairs,
and add them up.

### Worked example

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3},
\qquad
B = \begin{bmatrix} 0 & 2 \\ 1 & -1 \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

$AB$ has shape $(2 \times 3)(3 \times 2) = 2 \times 2$:

$$
AB =
\begin{bmatrix}
1\cdot 0 + 2\cdot 1 + 3\cdot 0 & 1\cdot 2 + 2\cdot(-1) + 3\cdot 1 \\
3\cdot 0 + 2\cdot 1 + 1\cdot 0 & 3\cdot 2 + 2\cdot(-1) + 1\cdot 1
\end{bmatrix}
=
\begin{bmatrix} 2 & 3 \\ 2 & 5 \end{bmatrix}
$$

$BA$ has shape $(3 \times 2)(2 \times 3) = 3 \times 3$:

$$
BA =
\begin{bmatrix}
6 & 4 & 2 \\
-2 & 0 & 2 \\
3 & 2 & 1
\end{bmatrix}
$$

### ⚠️ Multiplication is not commutative

The example above shows that in general

$$
AB \neq BA
$$

Here $AB$ and $BA$ don't even have the same shape. Even for square matrices,
where both products exist and have the same shape, they usually differ. **Order
matters**: "multiply $A$ by $B$ on the left" and "on the right" are different
operations.

### Not the same as element-wise multiplication

Matrix multiplication is **not** multiplying entry by entry. The entry-by-entry
product is a different operation called the **Hadamard product**, written
$A \odot B$, with $(A \odot B)_{ij} = a_{ij} b_{ij}$. It requires equal shapes.

---

## 4. The identity matrix

The **identity matrix** $I_n \in \mathbb{R}^{n \times n}$ has 1s on the main
diagonal and 0s everywhere else:

$$
I_n =
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

It is the matrix version of the number 1: multiplying by it changes nothing.

$$
I_m A = A I_n = A \qquad \text{for all } A \in \mathbb{R}^{m \times n}
$$

Note that $I_m \neq I_n$ when $m \neq n$. The identity on the left must match
the number of rows, and the one on the right must match the number of columns.

---

## 5. Properties of matrix operations

Assume all shapes are compatible.

| Property | Statement |
|---|---|
| **Associativity** | $(AB)C = A(BC)$ |
| **Distributivity** | $(A + B)C = AC + BC$ and $A(C + D) = AC + AD$ |
| **Identity** | $I_m A = A I_n = A$ |
| **Not commutative** | $AB \neq BA$ in general |

Associativity means you can drop the brackets and write $ABC$. However, the
*order in which you compute* still affects speed. For example, computing
$A(B\mathbf{x})$ (two matrix-vector products) is usually much cheaper than
$(AB)\mathbf{x}$ (a full matrix-matrix product first).

---

## 6. Inverse of a matrix

### Definition

Let $A \in \mathbb{R}^{n \times n}$ be a **square** matrix. If there is a matrix
$B \in \mathbb{R}^{n \times n}$ with

$$
AB = BA = I_n
$$

then $B$ is the **inverse** of $A$, written $A^{-1}$.

- If $A^{-1}$ exists, $A$ is called **invertible**, **regular** or **nonsingular**.
- If it does not exist, $A$ is called **singular** or **noninvertible**.
- When the inverse exists, it is **unique**.

The inverse "undoes" $A$: if $A$ maps $\mathbf{x}$ to $\mathbf{y}$, then
$A^{-1}$ maps $\mathbf{y}$ back to $\mathbf{x}$. Not every matrix can be undone.
A matrix that squashes space onto a line, for example, loses information, so it
has no inverse.

### The 2 × 2 case

For

$$
A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
$$

the inverse is

$$
A^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}}
\begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix}
$$

The recipe: **swap** the diagonal entries, **negate** the off-diagonal entries,
and **divide** by $a_{11}a_{22} - a_{12}a_{21}$. That number is the
**determinant** of $A$. $A$ is invertible exactly when the determinant is not
zero.

**Example.**

$$
A = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix},
\qquad \det(A) = 2\cdot 3 - 1\cdot 5 = 1,
\qquad A^{-1} = \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix}
$$

Check:

$$
AA^{-1} =
\begin{bmatrix} 2\cdot 3 + 1\cdot(-5) & 2\cdot(-1) + 1\cdot 2 \\ 5\cdot 3 + 3\cdot(-5) & 5\cdot(-1) + 3\cdot 2 \end{bmatrix}
=
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
= I_2 \;\checkmark
$$

For larger matrices, the inverse is computed with Gaussian elimination, which is
covered in the next section, *Solving Systems of Linear Equations*.

---

## 7. Transpose

### Definition

For $A \in \mathbb{R}^{m \times n}$, the **transpose** $A^\top \in \mathbb{R}^{n \times m}$
is the matrix with entries

$$
(A^\top)_{ij} = a_{ji}
$$

Rows become columns and columns become rows. Geometrically, you flip the matrix
across its main diagonal.

**Example.**

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\quad \Longrightarrow \quad
A^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

Transposing turns a column vector into a row vector. This gives a neat way to
write the dot product: $\mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$.

### Properties of the inverse and transpose

| Property | Note |
|---|---|
| $AA^{-1} = I = A^{-1}A$ | definition of the inverse |
| $(AB)^{-1} = B^{-1}A^{-1}$ | the order **reverses** |
| $(A + B)^{-1} \neq A^{-1} + B^{-1}$ | inverting does **not** split over a sum |
| $(A^\top)^\top = A$ | transposing twice returns the original |
| $(AB)^\top = B^\top A^\top$ | the order **reverses** |
| $(A + B)^\top = A^\top + B^\top$ | transposing **does** split over a sum |
| $(A^{-1})^\top = (A^\top)^{-1}$ | for invertible $A$; often written $A^{-\top}$ |

**Why does the order reverse?** Think of putting on socks and then shoes. To
undo it, you take off the shoes first, then the socks. Applying $B$ and then $A$
is undone by applying $A^{-1}$ and then $B^{-1}$, which gives $B^{-1}A^{-1}$.

**Why is $(A+B)^{-1} \neq A^{-1} + B^{-1}$?** Numbers already show it:
$\frac{1}{2 + 4} = \frac{1}{6}$, but $\frac{1}{2} + \frac{1}{4} = \frac{3}{4}$.

### Symmetric matrices

A square matrix is **symmetric** if

$$
A = A^\top
$$

meaning $a_{ij} = a_{ji}$: the matrix is a mirror image across its diagonal.

$$
\begin{bmatrix} 4 & 1 & 7 \\ 1 & 3 & 0 \\ 7 & 0 & 5 \end{bmatrix}
\quad \text{is symmetric.}
$$

Useful facts:

- Only square matrices can be symmetric.
- If $A$ is invertible, then $A^\top$ is also invertible, and $(A^{-1})^\top = (A^\top)^{-1}$.
- The **sum** of two symmetric matrices is symmetric.
- The **product** of two symmetric matrices is generally **not** symmetric.
  For example:

$$
\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
=
\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
$$

- For **any** matrix $A$, both $A^\top A$ and $AA^\top$ are symmetric. These
  products appear all the time in machine learning, for example in covariance
  matrices and in the normal equations of linear regression.

---

## 8. Multiplication by a scalar

For $A \in \mathbb{R}^{m \times n}$ and a scalar $\lambda \in \mathbb{R}$,
$\lambda A$ multiplies **every entry** by $\lambda$:

$$
(\lambda A)_{ij} = \lambda\, a_{ij}
$$

**Example.**

$$
3 \begin{bmatrix} 1 & -2 \\ 0 & 4 \end{bmatrix}
=
\begin{bmatrix} 3 & -6 \\ 0 & 12 \end{bmatrix}
$$

For $\lambda, \psi \in \mathbb{R}$:

| Property | Statement |
|---|---|
| **Associativity** | $(\lambda\psi)A = \lambda(\psi A)$ |
| | $\lambda(AB) = (\lambda A)B = A(\lambda B) = (AB)\lambda$ |
| | $(\lambda A)^\top = \lambda A^\top$ |
| **Distributivity** | $(\lambda + \psi)A = \lambda A + \psi A$ |
| | $\lambda(A + B) = \lambda A + \lambda B$ |

Scalars, unlike matrices, can be moved freely anywhere in a product.

---

## 9. Compact representation of systems of linear equations

This is where matrices pay off. Consider the system

$$
\begin{aligned}
2x_1 + 3x_2 + 5x_3 &= 1 \\
4x_1 - 2x_2 - 7x_3 &= 8 \\
9x_1 + 5x_2 - 3x_3 &= 2
\end{aligned}
$$

Using the rules of matrix multiplication, the whole system becomes one equation:

$$
\underbrace{\begin{bmatrix} 2 & 3 & 5 \\ 4 & -2 & -7 \\ 9 & 5 & -3 \end{bmatrix}}_{A}
\underbrace{\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}}_{\mathbf{x}}
=
\underbrace{\begin{bmatrix} 1 \\ 8 \\ 2 \end{bmatrix}}_{\mathbf{b}}
\qquad \Longleftrightarrow \qquad
A\mathbf{x} = \mathbf{b}
$$

Multiplying row $i$ of $A$ by $\mathbf{x}$ gives back the left-hand side of
equation $i$.

There is a second way to read the same product that is just as important.
**$A\mathbf{x}$ is a linear combination of the columns of $A$**, weighted by the
entries of $\mathbf{x}$:

$$
A\mathbf{x} =
x_1 \begin{bmatrix} 2 \\ 4 \\ 9 \end{bmatrix}
+ x_2 \begin{bmatrix} 3 \\ -2 \\ 5 \end{bmatrix}
+ x_3 \begin{bmatrix} 5 \\ -7 \\ -3 \end{bmatrix}
$$

So solving $A\mathbf{x} = \mathbf{b}$ means asking: **how much of each column of
$A$ do we need to build $\mathbf{b}$?** This "column view" is the key to the
upcoming topics of vector spaces, linear independence, basis and rank.

If $A$ is square and invertible, multiplying both sides by $A^{-1}$ on the left
gives the solution directly:

$$
A^{-1}A\mathbf{x} = A^{-1}\mathbf{b} \quad \Longrightarrow \quad \mathbf{x} = A^{-1}\mathbf{b}
$$

In practice, the explicit inverse is rarely computed. Elimination methods are
faster and more numerically stable.

---

## 10. Why this matters in machine learning

| ML concept | Matrix form |
|---|---|
| A dataset of $N$ examples with $D$ features | data matrix $X \in \mathbb{R}^{N \times D}$ |
| Linear regression predictions | $\hat{\mathbf{y}} = X\boldsymbol{\theta}$ |
| Linear regression closed-form solution | $\boldsymbol{\theta} = (X^\top X)^{-1} X^\top \mathbf{y}$ |
| One neural-network layer | $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$ |
| A batch of inputs through a layer | $H = \sigma(XW^\top + \mathbf{b}^\top)$ |
| Covariance of centred data | $\frac{1}{N} X^\top X$ (symmetric) |
| Grayscale image | a matrix of pixel intensities |

Every idea in this section appears in the closed-form regression solution: a
**transpose**, a **matrix product**, a **symmetric matrix** $X^\top X$, and an
**inverse**.

---

## Summary

- An $(m, n)$ matrix has $m$ rows and $n$ columns. Always track shapes.
- **Addition** and **scalar multiplication** work entry by entry.
- **Matrix multiplication** combines rows of the first matrix with columns of
  the second. It needs matching inner dimensions, and $AB \neq BA$ in general.
- The **identity** $I$ is the "1" of matrices. The **inverse** $A^{-1}$ undoes
  $A$ and exists only for some square matrices.
- The **transpose** swaps rows and columns. Inverting or transposing a product
  **reverses the order**.
- **Symmetric** matrices satisfy $A = A^\top$. The matrix $A^\top A$ is always symmetric.
- A system of linear equations is written compactly as $A\mathbf{x} = \mathbf{b}$,
  and $A\mathbf{x}$ is a linear combination of the columns of $A$.

**Next:** [Solving Systems of Linear Equations](../03-solving%20systems%20of%20linear%20equations/)
