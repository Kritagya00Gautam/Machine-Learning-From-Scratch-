# Solving Systems of Linear Equations

In the previous sections we wrote a system of $m$ linear equations in $n$
unknowns,

$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
&\;\;\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
$$

compactly as one matrix equation:

$$
A\mathbf{x} = \mathbf{b}, \qquad A \in \mathbb{R}^{m \times n},\;
\mathbf{x} \in \mathbb{R}^{n},\; \mathbf{b} \in \mathbb{R}^{m}
$$

Here $a_{ij}$ and $b_i$ are known numbers and $x_1, \dots, x_n$ are the
unknowns. This section answers the next question: **how do we actually find all
solutions?** Along the way we also get an algorithm for computing the inverse of
a matrix.

> Based on Section 2.3 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> worked examples here are written for these notes.

---

## 1. Particular and general solution

### The column view

Recall from the matrices section that $A\mathbf{x}$ is a linear combination of
the columns $\mathbf{c}_1, \dots, \mathbf{c}_n$ of $A$:

$$
A\mathbf{x} = x_1\mathbf{c}_1 + x_2\mathbf{c}_2 + \cdots + x_n\mathbf{c}_n
$$

So solving $A\mathbf{x} = \mathbf{b}$ means finding weights $x_1, \dots, x_n$
that build $\mathbf{b}$ out of the columns of $A$.

### An easy example

Take a system with **2 equations and 4 unknowns**:

$$
\begin{bmatrix} 1 & 0 & 3 & -2 \\ 0 & 1 & 1 & 5 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}
=
\begin{bmatrix} 6 \\ 4 \end{bmatrix}
$$

With fewer equations than unknowns, we usually expect **infinitely many**
solutions. This system is especially easy because the first two columns are the
unit vectors $\begin{bmatrix}1\\0\end{bmatrix}$ and $\begin{bmatrix}0\\1\end{bmatrix}$.

**Step 1: one solution.** We can build $\mathbf{b}$ directly from the first two
columns:

$$
\begin{bmatrix} 6 \\ 4 \end{bmatrix}
= 6\begin{bmatrix} 1 \\ 0 \end{bmatrix} + 4\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

So $\mathbf{x}_p = [6, 4, 0, 0]^\top$ is a solution. This is called a
**particular solution** (or **special solution**).

**Step 2: ways of making zero.** It is not the only solution. Adding $\mathbf{0}$
to a solution does not change $A\mathbf{x}$, so we look for non-trivial ways to
produce $\mathbf{0}$ from the columns. Write each remaining column in terms of
the first two:

- $\mathbf{c}_3 = \begin{bmatrix}3\\1\end{bmatrix} = 3\mathbf{c}_1 + 1\mathbf{c}_2$,
  so $3\mathbf{c}_1 + 1\mathbf{c}_2 - 1\mathbf{c}_3 + 0\mathbf{c}_4 = \mathbf{0}$.
  This gives the vector $[3, 1, -1, 0]^\top$.
- $\mathbf{c}_4 = \begin{bmatrix}-2\\5\end{bmatrix} = -2\mathbf{c}_1 + 5\mathbf{c}_2$,
  so $-2\mathbf{c}_1 + 5\mathbf{c}_2 + 0\mathbf{c}_3 - 1\mathbf{c}_4 = \mathbf{0}$.
  This gives the vector $[-2, 5, 0, -1]^\top$.

Any multiple of these vectors also produces $\mathbf{0}$:
$A(\lambda_1[3, 1, -1, 0]^\top) = \lambda_1 \mathbf{0} = \mathbf{0}$ for every
$\lambda_1 \in \mathbb{R}$, and likewise for the second.

**Step 3: combine.** Adding everything together gives **all** solutions, the
**general solution**:

$$
\left\{ \mathbf{x} \in \mathbb{R}^4 :\;
\mathbf{x} =
\begin{bmatrix} 6 \\ 4 \\ 0 \\ 0 \end{bmatrix}
+ \lambda_1 \begin{bmatrix} 3 \\ 1 \\ -1 \\ 0 \end{bmatrix}
+ \lambda_2 \begin{bmatrix} -2 \\ 5 \\ 0 \\ -1 \end{bmatrix},
\;\; \lambda_1, \lambda_2 \in \mathbb{R} \right\}
$$

### The general recipe

1. Find **one particular solution** of $A\mathbf{x} = \mathbf{b}$.
2. Find **all solutions of the homogeneous system** $A\mathbf{x} = \mathbf{0}$.
3. **Add** them: general solution = particular solution + every homogeneous solution.

Why this works: if $A\mathbf{x}_p = \mathbf{b}$ and $A\mathbf{x}_h = \mathbf{0}$,
then $A(\mathbf{x}_p + \mathbf{x}_h) = \mathbf{b} + \mathbf{0} = \mathbf{b}$.
Conversely, any two solutions of $A\mathbf{x} = \mathbf{b}$ differ by a solution
of $A\mathbf{x} = \mathbf{0}$.

> **Neither is unique.** Any solution can serve as the particular solution, and
> the vectors describing the homogeneous part can be chosen in many ways. The
> *set* of all solutions is what is unique.

The example was easy because the matrix already had a convenient shape. General
systems don't, but there is an algorithm that transforms **any** system into this
shape: **Gaussian elimination**. It is built from *elementary transformations*.

---

## 2. Elementary transformations

An **elementary transformation** changes a system of equations into a simpler
one **without changing its solution set**. There are three of them:

| Transformation | On the matrix |
|---|---|
| Swap two equations | swap two rows |
| Multiply an equation by a constant $\lambda \neq 0$ | scale a row by $\lambda \in \mathbb{R}\setminus\{0\}$ |
| Add (a multiple of) one equation to another | add a multiple of one row to another row |

Each operation can be undone (swap back, multiply by $1/\lambda$, subtract
again), so no solutions are gained or lost. That is also why scaling by $0$ is
not allowed: it destroys an equation and cannot be undone.

### The augmented matrix

Writing $x_1, x_2, \dots$ over and over is tedious. Instead we store the
coefficients and the right-hand side together in the **augmented matrix**

$$
\left[\, A \;\middle|\; \mathbf{b} \,\right]
$$

The vertical bar separates the left-hand side from the right-hand side. Each row
is one equation. We write $\rightsquigarrow$ to mean "transformed by elementary
row operations into".

---

## 3. Row-echelon form

### Pivots and the staircase

The **pivot** (or **leading coefficient**) of a row is its first nonzero entry
from the left.

**Definition (Row-echelon form, REF).** A matrix is in row-echelon form if

1. all rows that contain only zeros are at the bottom, and
2. among the nonzero rows, the pivot of each row is **strictly to the right** of
   the pivot of the row above it.

This gives every REF matrix a **staircase** shape:

$$
\begin{bmatrix}
\boxed{\ast} & \ast & \ast & \ast & \ast \\
0 & 0 & \boxed{\ast} & \ast & \ast \\
0 & 0 & 0 & \boxed{\ast} & \ast \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

(Boxed entries are pivots, so they are nonzero. $\ast$ can be any number. Some
textbooks also require each pivot to be $1$ in REF. The MML book does not.)

### Basic and free variables

- Variables whose columns contain a pivot are **basic variables**.
- All other variables are **free variables**. They can take any value, and the
  basic variables are then determined by them.

In the staircase above, $x_1, x_3, x_4$ are basic and $x_2, x_5$ are free.

### Reading off a particular solution

In REF, write $\mathbf{b}$ as a combination of the **pivot columns** only:

$$
\mathbf{b} = \sum_{i=1}^{P} \lambda_i \mathbf{p}_i
$$

where $\mathbf{p}_1, \dots, \mathbf{p}_P$ are the pivot columns. Solve for the
$\lambda_i$ starting from the **rightmost** pivot column (the bottom equation)
and work leftwards (back-substitution). Put the $\lambda_i$ into the positions of
the basic variables and set every **free variable to $0$**. This gives a
particular solution.

### How many solutions? Read it from the REF

| What the REF of $[A \mid \mathbf{b}]$ shows | Solutions |
|---|---|
| A row $[\,0 \;\cdots\; 0 \mid c\,]$ with $c \neq 0$ (it says $0 = c$) | **none**: the system is inconsistent |
| No such row, and every variable is basic | **exactly one** |
| No such row, and at least one free variable | **infinitely many**: one free parameter per free variable |

This matches the geometric picture from the first section: lines (or planes)
that don't meet, meet at one point, or overlap along a whole line or plane.

---

## 4. Reduced row-echelon form and Gaussian elimination

**Definition (Reduced row-echelon form, RREF).** A matrix is in reduced
row-echelon form (also called *row-reduced echelon form* or *row canonical form*)
if

1. it is in row-echelon form,
2. every pivot is $1$, and
3. each pivot is the **only nonzero entry in its column**.

$$
\begin{bmatrix}
\boxed{1} & \ast & 0 & 0 & \ast \\
0 & 0 & \boxed{1} & 0 & \ast \\
0 & 0 & 0 & \boxed{1} & \ast
\end{bmatrix}
$$

In RREF the pivot columns are exactly the standard unit vectors
$\mathbf{e}_1, \mathbf{e}_2, \dots$. That makes the general solution easy to read
off, as the example in Section 1 showed.

**Gaussian elimination** is the algorithm that applies elementary
transformations to bring a system into (reduced) row-echelon form:

1. **Forward pass:** working left to right, pick a pivot in the current column
   (swap rows if needed), then add multiples of the pivot row to clear every
   entry **below** it. This gives REF.
2. **Backward pass:** scale each pivot row so the pivot is $1$, then clear every
   entry **above** each pivot. This gives RREF.

### Worked example: a system with a parameter

For $a \in \mathbb{R}$, find all solutions of

$$
\begin{aligned}
x_1 + 2x_2 - \phantom{2}x_3 &= 3 \\
2x_1 + 4x_2 + \phantom{2}x_3 &= 9 \\
x_1 + 2x_2 + 2x_3 &= a
\end{aligned}
$$

**Augmented matrix:**

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
2 & 4 & 1 & 9 \\
1 & 2 & 2 & a
\end{array}\right]
\begin{array}{l} \\ -2R_1 \\ -R_1 \end{array}
\rightsquigarrow
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 0 & 3 & 3 \\
0 & 0 & 3 & a-3
\end{array}\right]
\begin{array}{l} \\ \\ -R_2 \end{array}
\rightsquigarrow
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 0 & 3 & 3 \\
0 & 0 & 0 & a-6
\end{array}\right]
$$

This is REF. The last row reads $0 = a - 6$.

- If $a \neq 6$, it says $0 = $ (something nonzero), so there is **no solution**.
- If $a = 6$, the last row is $0 = 0$ and can be dropped.

**Continue to RREF (with $a = 6$):** scale $R_2$ by $\tfrac{1}{3}$, then add
$R_2$ to $R_1$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0
\end{array}\right]
\begin{array}{l} +R_2 \\ \\ \\ \end{array}
\rightsquigarrow
\left[\begin{array}{ccc|c}
1 & 2 & 0 & 4 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 0
\end{array}\right]
$$

- **Pivots** are in columns 1 and 3, so $x_1, x_3$ are basic and $x_2$ is free.
- **Particular solution:** set $x_2 = 0$, then read off $x_1 = 4$, $x_3 = 1$.
  So $\mathbf{x}_p = [4, 0, 1]^\top$.
- **Homogeneous solutions:** column 2 is $2 \times$ column 1, so
  $2\mathbf{c}_1 - \mathbf{c}_2 = \mathbf{0}$, which gives $[2, -1, 0]^\top$.

**General solution (for $a = 6$):**

$$
\left\{ \mathbf{x} \in \mathbb{R}^3 :\;
\mathbf{x} =
\begin{bmatrix} 4 \\ 0 \\ 1 \end{bmatrix}
+ \lambda \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix},
\;\; \lambda \in \mathbb{R} \right\}
$$

*Check:* $\mathbf{x} = [4 + 2\lambda,\; -\lambda,\; 1]^\top$ gives
$4 + 2\lambda - 2\lambda - 1 = 3$, $\;8 + 4\lambda - 4\lambda + 1 = 9$, and
$4 + 2\lambda - 2\lambda + 2 = 6$. ✓

---

## 5. The Minus-1 Trick

The Minus-1 Trick is a quick way to **read off all solutions of the homogeneous
system** $A\mathbf{x} = \mathbf{0}$ when $A \in \mathbb{R}^{k \times n}$ is in
RREF.

**Setup.** Assume $A$ is in RREF and has **no all-zero rows** (delete them first;
they say $0 = 0$). Then $A$ has $k$ pivots, and its pivot columns are the unit
vectors $\mathbf{e}_1, \dots, \mathbf{e}_k$.

**The trick.**

1. Extend $A$ to a square $n \times n$ matrix $\tilde{A}$ by inserting rows of
   the form
   $$
   \begin{bmatrix} 0 & \cdots & 0 & -1 & 0 & \cdots & 0 \end{bmatrix}
   $$
   wherever a pivot is missing on the diagonal, so that the diagonal of
   $\tilde{A}$ contains only $1$s and $-1$s.
2. The columns of $\tilde{A}$ that have $-1$ on the diagonal are solutions of
   $A\mathbf{x} = \mathbf{0}$. Every homogeneous solution is a linear
   combination of them.

These columns form a **basis** of the solution space of $A\mathbf{x} = \mathbf{0}$.
Later this space is called the **kernel** or **null space** of $A$.

### Example

$$
A =
\begin{bmatrix}
1 & 0 & 2 & 0 & -1 \\
0 & 1 & -3 & 0 & 4 \\
0 & 0 & 0 & 1 & 2
\end{bmatrix}
\in \mathbb{R}^{3 \times 5}
$$

$A$ is in RREF with pivots in columns 1, 2 and 4. Pivots are missing on the
diagonal in positions 3 and 5, so we insert $-1$ rows there:

$$
\tilde{A} =
\begin{bmatrix}
1 & 0 & 2 & 0 & -1 \\
0 & 1 & -3 & 0 & 4 \\
0 & 0 & \mathbf{-1} & 0 & 0 \\
0 & 0 & 0 & 1 & 2 \\
0 & 0 & 0 & 0 & \mathbf{-1}
\end{bmatrix}
$$

Columns 3 and 5 contain the $-1$s, so

$$
\left\{ \mathbf{x} \in \mathbb{R}^5 :\;
\mathbf{x} =
\lambda_1 \begin{bmatrix} 2 \\ -3 \\ -1 \\ 0 \\ 0 \end{bmatrix}
+ \lambda_2 \begin{bmatrix} -1 \\ 4 \\ 0 \\ 2 \\ -1 \end{bmatrix},
\;\; \lambda_1, \lambda_2 \in \mathbb{R} \right\}
$$

are all solutions of $A\mathbf{x} = \mathbf{0}$.

*Check the first vector:* row 1 gives $2 - 2 = 0$, row 2 gives $-3 + 3 = 0$,
row 3 gives $0$. ✓

**Why it works.** A non-pivot column $\mathbf{c}_j$ of an RREF matrix is a
combination of the pivot columns to its left, and its entries are exactly the
weights. So "those weights, followed by $-1$ in position $j$" produces
$\mathbf{0}$. That is the same reasoning as Step 2 in Section 1. The trick just
lines the vectors up neatly.

The same trick on the earlier example: the RREF without its zero row is
$\begin{bmatrix}1 & 2 & 0\\0 & 0 & 1\end{bmatrix}$. Inserting a $-1$ row in
position 2 gives column 2 $= [2, -1, 0]^\top$, the homogeneous solution we found
by inspection.

---

## 6. Calculating the inverse with Gaussian elimination

To find $A^{-1}$ for a square $A \in \mathbb{R}^{n \times n}$, we need a matrix
$X$ with

$$
AX = I_n
$$

Writing $X = [\mathbf{x}_1 \mid \cdots \mid \mathbf{x}_n]$ column by column, this
is really $n$ systems at once: $A\mathbf{x}_i = \mathbf{e}_i$. All of them share
the same $A$, so we solve them together using one augmented matrix:

$$
\left[\, A \;\middle|\; I_n \,\right]
\;\rightsquigarrow\; \cdots \;\rightsquigarrow\;
\left[\, I_n \;\middle|\; A^{-1} \,\right]
$$

Bring the augmented matrix to RREF. If the left block becomes $I_n$, the right
block is $A^{-1}$. **Computing an inverse is the same as solving systems of
linear equations.**

If the left block **cannot** be turned into $I_n$ (a zero row appears on the
left), then $A$ is **singular** and has no inverse.

### Example

$$
A = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 1 & 0 & 1 & 0 & 0 \\
1 & 2 & 1 & 0 & 1 & 0 \\
0 & 1 & 2 & 0 & 0 & 1
\end{array}\right]
\begin{array}{l} \\ -R_1 \\ \\ \end{array}
\rightsquigarrow
\left[\begin{array}{ccc|ccc}
1 & 1 & 0 & 1 & 0 & 0 \\
0 & 1 & 1 & -1 & 1 & 0 \\
0 & 1 & 2 & 0 & 0 & 1
\end{array}\right]
\begin{array}{l} \\ \\ -R_2 \end{array}
$$

$$
\rightsquigarrow
\left[\begin{array}{ccc|ccc}
1 & 1 & 0 & 1 & 0 & 0 \\
0 & 1 & 1 & -1 & 1 & 0 \\
0 & 0 & 1 & 1 & -1 & 1
\end{array}\right]
\begin{array}{l} \\ -R_3 \\ \\ \end{array}
\rightsquigarrow
\left[\begin{array}{ccc|ccc}
1 & 1 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & -2 & 2 & -1 \\
0 & 0 & 1 & 1 & -1 & 1
\end{array}\right]
\begin{array}{l} -R_2 \\ \\ \\ \end{array}
$$

$$
\rightsquigarrow
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & 3 & -2 & 1 \\
0 & 1 & 0 & -2 & 2 & -1 \\
0 & 0 & 1 & 1 & -1 & 1
\end{array}\right]
\qquad \Longrightarrow \qquad
A^{-1} = \begin{bmatrix} 3 & -2 & 1 \\ -2 & 2 & -1 \\ 1 & -1 & 1 \end{bmatrix}
$$

*Check:* the first row of $A$ times the columns of $A^{-1}$ gives
$[3 - 2,\; -2 + 2,\; 1 - 1] = [1, 0, 0]$. The other rows give $[0, 1, 0]$ and
$[0, 0, 1]$, so $AA^{-1} = I_3$. ✓

---

## 7. Algorithms for solving $A\mathbf{x} = \mathbf{b}$ in practice

Here we assume a solution exists. If none exists, we look for an
**approximate** solution instead, for example with linear regression (Chapter 9
of the book).

### Using the inverse

If $A$ is **square and invertible**, the solution is

$$
\mathbf{x} = A^{-1}\mathbf{b}
$$

This is often not available, because $A$ may not be square, or may be singular.

### Using the Moore-Penrose pseudo-inverse

If $A$ has **linearly independent columns**, multiply both sides by $A^\top$:

$$
A\mathbf{x} = \mathbf{b}
\;\Longleftrightarrow\;
A^\top A\mathbf{x} = A^\top \mathbf{b}
\;\Longleftrightarrow\;
\mathbf{x} = (A^\top A)^{-1} A^\top \mathbf{b}
$$

- $A^\top A$ is square (and invertible under this assumption), even if $A$ is not.
- The matrix $(A^\top A)^{-1} A^\top$ is the **Moore-Penrose pseudo-inverse** of $A$.
- The resulting $\mathbf{x}$ is the **least-squares solution**. This is exactly
  the closed-form solution of linear regression seen in the matrices section.

**Drawbacks:** the matrix-matrix product $A^\top A$ and the inverse are expensive
to compute. Explicit inverses and pseudo-inverses are also **numerically
unstable**, so in practice they are generally avoided.

### Gaussian elimination

Gaussian elimination is intuitive and constructive, and it is a workhorse behind
many other tasks:

- computing **determinants**,
- checking whether vectors are **linearly independent**,
- computing the **inverse** of a matrix,
- computing the **rank** of a matrix,
- finding a **basis** of a vector space.

It works well for systems with **thousands** of variables. But its cost grows
**cubically** with the number of equations (roughly $n^3$ operations), so it
becomes impractical for **millions** of variables.

### Iterative methods

Very large systems are solved **indirectly**, by iterative methods:

- **Stationary iterative methods:** the Richardson method, the Jacobi method,
  the Gauss-Seidel method, and successive over-relaxation (SOR).
- **Krylov subspace methods:** conjugate gradients (CG), generalized minimal
  residual (GMRES), and biconjugate gradients (BiCG).

Let $\mathbf{x}_\ast$ be the true solution. These methods start from a guess and
repeatedly improve it with an update of the form

$$
\mathbf{x}^{(k+1)} = C\mathbf{x}^{(k)} + \mathbf{d}
$$

for a suitably chosen matrix $C$ and vector $\mathbf{d}$. These are chosen so
that the error $\lVert \mathbf{x}^{(k+1)} - \mathbf{x}_\ast \rVert$ shrinks at every step,
and the iterates converge to $\mathbf{x}_\ast$. (The norm $\lVert \cdot \rVert$,
a way to measure the size of a vector, is introduced in Chapter 3.)

---

## 8. Why this matters in machine learning

| ML situation | Connection |
|---|---|
| Linear regression | the normal equations $X^\top X\boldsymbol{\theta} = X^\top \mathbf{y}$ are a linear system |
| Least-squares fitting | the pseudo-inverse gives the best approximate solution when no exact one exists |
| Large-scale models (Gaussian processes, kernel methods) | huge linear systems solved with iterative methods such as conjugate gradients |
| Rank, basis, null space | all computed with Gaussian elimination, and needed for PCA and dimensionality reduction |

---

## Summary

- **General solution = particular solution + all solutions of $A\mathbf{x} = \mathbf{0}$.**
  Neither part is unique, but the full solution set is.
- **Elementary transformations** (swap rows, scale a row by $\lambda \neq 0$,
  add a multiple of one row to another) never change the solution set.
- The **augmented matrix** $[A \mid \mathbf{b}]$ stores a whole system compactly.
- **REF** has a staircase of pivots. Pivot columns give **basic variables** and
  the rest are **free variables**.
- **RREF** additionally has pivots equal to $1$, which are the only nonzero
  entries in their columns. **Gaussian elimination** produces it.
- A row $0 = c$ with $c \neq 0$ means **no solution**. Free variables mean
  **infinitely many**. Otherwise the solution is **unique**.
- The **Minus-1 Trick** reads a basis of the solutions of $A\mathbf{x} = \mathbf{0}$
  straight from the RREF.
- **Inverse:** $[A \mid I] \rightsquigarrow [I \mid A^{-1}]$.
- In practice: the inverse and pseudo-inverse are costly and unstable, Gaussian
  elimination is $O(n^3)$, and huge systems use **iterative methods**.

**Previous:** [Matrices](../02-matrices/) &nbsp;·&nbsp;
**Next:** [Vector Spaces](../04-vector%20spaces/)
