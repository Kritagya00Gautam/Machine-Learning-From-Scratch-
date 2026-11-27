# Linear Independence

In a vector space we can **add** vectors and **scale** them, and closure
guarantees we stay inside the space. A natural question follows:

> Can we pick a small set of vectors from which **every** vector in the space
> can be built by adding and scaling?

Such a set is called a **basis** (next section). Before we get there, we need two
ideas: **linear combinations** and **linear independence**.

> Based on Section 2.5 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> examples here are written for these notes.

---

## 1. Linear combinations

**Definition.** Let $V$ be a vector space and $\mathbf{x}_1, \dots, \mathbf{x}_k \in V$.
Every vector of the form

$$
\mathbf{v} = \lambda_1\mathbf{x}_1 + \lambda_2\mathbf{x}_2 + \cdots + \lambda_k\mathbf{x}_k
= \sum_{i=1}^{k} \lambda_i \mathbf{x}_i,
\qquad \lambda_1, \dots, \lambda_k \in \mathbb{R}
$$

is a **linear combination** of $\mathbf{x}_1, \dots, \mathbf{x}_k$.

**Intuition: a recipe.** Think of the vectors as *ingredients* and the scalars
$\lambda_i$ as *amounts*. A linear combination is a recipe: "take $2$ of
$\mathbf{x}_1$, $-1$ of $\mathbf{x}_2$, and $\tfrac{1}{2}$ of $\mathbf{x}_3$".

**Example.**

$$
2\begin{bmatrix} 1 \\ 0 \end{bmatrix} + 3\begin{bmatrix} 0 \\ 1 \end{bmatrix}
= \begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

### The trivial combination

The zero vector can **always** be built from any vectors by using zero of each:

$$
\mathbf{0} = 0\cdot\mathbf{x}_1 + 0\cdot\mathbf{x}_2 + \cdots + 0\cdot\mathbf{x}_k
$$

This is the **trivial** linear combination. It tells us nothing interesting. The
interesting question is whether $\mathbf{0}$ can be built in a **non-trivial**
way, with at least one $\lambda_i \neq 0$.

---

## 2. Linear (in)dependence

**Definition.** Let $\mathbf{x}_1, \dots, \mathbf{x}_k \in V$.

- If there is a **non-trivial** combination
  $\sum_{i=1}^{k} \lambda_i \mathbf{x}_i = \mathbf{0}$ (at least one $\lambda_i \neq 0$),
  the vectors are **linearly dependent**.
- If the **only** way to get $\mathbf{0}$ is the trivial one,
  $\lambda_1 = \cdots = \lambda_k = 0$, the vectors are **linearly independent**.

### Intuition: no redundancy

Linear independence is one of the most important ideas in linear algebra.
**Independent vectors carry no redundant information.** Remove any one of them,
and you lose something you cannot rebuild from the rest.

**Analogy 1: giving directions.** Suppose you're explaining how to reach a café:
*"Walk 3 blocks east, then 4 blocks north."* That fully describes the trip on a
flat city grid. If you add *"...it's also 5 blocks away diagonally north-east,"*
that's true, but **redundant**: the diagonal move is already a combination of
"east" and "north". The set {east, north} is independent. The set
{east, north, diagonal} is dependent.

**Analogy 2: mixing light.** On a screen, colours are made from red, green and
blue light. Represent them as vectors in $\mathbb{R}^3$:

$$
\text{red} = \begin{bmatrix}1\\0\\0\end{bmatrix},\quad
\text{green} = \begin{bmatrix}0\\1\\0\end{bmatrix},\quad
\text{yellow} = \begin{bmatrix}1\\1\\0\end{bmatrix}
$$

Yellow is red + green, so $\{\text{red}, \text{green}, \text{yellow}\}$ is
**dependent**:

$$
1\cdot\text{red} + 1\cdot\text{green} - 1\cdot\text{yellow} = \mathbf{0}
$$

With red, green and yellow you can never produce any blue, because yellow added
nothing new. Replace yellow with blue, $[0, 0, 1]^\top$, and the three vectors are
independent and can mix every colour.

### Geometric picture

| Vectors | Dependent when... |
|---|---|
| 2 vectors in $\mathbb{R}^2$ or $\mathbb{R}^3$ | they lie on the **same line** through the origin (one is a multiple of the other) |
| 3 vectors in $\mathbb{R}^3$ | they lie in the **same plane** through the origin |
| more than $n$ vectors in $\mathbb{R}^n$ | **always** dependent: there is not enough room |

---

## 3. Useful facts

- A set of vectors is **either** dependent **or** independent. There is no third option.
- If one of the vectors is $\mathbf{0}$, the set is **dependent**: take
  $\lambda = 1$ for the zero vector and $0$ for the rest.
- If two vectors are **identical**, the set is **dependent**: $1\cdot\mathbf{x} - 1\cdot\mathbf{x} = \mathbf{0}$.
- For $k \geq 2$ nonzero vectors: they are dependent **if and only if at least
  one of them is a linear combination of the others**. In particular, if one is
  a multiple of another ($\mathbf{x}_i = \lambda\mathbf{x}_j$), the set is dependent.

---

## 4. Testing independence with Gaussian elimination

The practical test:

1. Write the vectors as the **columns** of a matrix $A$.
2. Use Gaussian elimination to bring $A$ to **row-echelon form** (REF is
   enough here, RREF is not needed).
3. Read the result:
   - **Pivot columns** are independent of the columns to their left.
   - **Non-pivot columns** are linear combinations of the pivot columns to their left.

> **All columns are pivot columns $\iff$ the vectors are linearly independent.**
> If even one column has no pivot, the vectors are dependent.

Why this works: we are solving $A\boldsymbol{\lambda} = \mathbf{0}$ for the
weights $\boldsymbol{\lambda}$. A non-pivot column means a free variable, so
there are infinitely many solutions, including non-trivial ones.

Note that the order of the columns matters for *which* vectors get labelled
redundant, but not for *whether* the set is dependent.

### Example 1: independent

Are $\begin{bmatrix}1\\1\\0\end{bmatrix}, \begin{bmatrix}0\\1\\1\end{bmatrix}, \begin{bmatrix}1\\0\\1\end{bmatrix}$ independent?

$$
\begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}
\begin{array}{l} \\ -R_1 \\ \\ \end{array}
\rightsquigarrow
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & -1 \\ 0 & 1 & 1 \end{bmatrix}
\begin{array}{l} \\ \\ -R_2 \end{array}
\rightsquigarrow
\begin{bmatrix} \boxed{1} & 0 & 1 \\ 0 & \boxed{1} & -1 \\ 0 & 0 & \boxed{2} \end{bmatrix}
$$

Every column has a pivot, so the only solution is $\lambda_1 = \lambda_2 = \lambda_3 = 0$.
The vectors are **independent**.

### Example 2: dependent

Are $\mathbf{x}_1 = \begin{bmatrix}1\\2\\3\end{bmatrix}, \mathbf{x}_2 = \begin{bmatrix}0\\1\\4\end{bmatrix}, \mathbf{x}_3 = \begin{bmatrix}2\\5\\10\end{bmatrix}$ independent?

$$
\begin{bmatrix} 1 & 0 & 2 \\ 2 & 1 & 5 \\ 3 & 4 & 10 \end{bmatrix}
\begin{array}{l} \\ -2R_1 \\ -3R_1 \end{array}
\rightsquigarrow
\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & 4 & 4 \end{bmatrix}
\begin{array}{l} \\ \\ -4R_2 \end{array}
\rightsquigarrow
\begin{bmatrix} \boxed{1} & 0 & 2 \\ 0 & \boxed{1} & 1 \\ 0 & 0 & 0 \end{bmatrix}
$$

Column 3 has no pivot, so the vectors are **dependent**. The REF even tells us
the recipe: column 3 $= 2 \times$ column 1 $+ 1 \times$ column 2, so

$$
\mathbf{x}_3 = 2\mathbf{x}_1 + \mathbf{x}_2
\qquad\text{(check: } 2[1,2,3] + [0,1,4] = [2,5,10] \;✓\text{)}
$$

---

## 5. Linear combinations of independent vectors

Suppose $\mathbf{b}_1, \dots, \mathbf{b}_k$ are **independent**, and we build $m$
new vectors from them:

$$
\mathbf{x}_j = \sum_{i=1}^{k} \lambda_{ij}\mathbf{b}_i, \qquad j = 1, \dots, m
$$

Collect the $\mathbf{b}_i$ as columns of $B = [\mathbf{b}_1, \dots, \mathbf{b}_k]$,
and the recipe for $\mathbf{x}_j$ as the vector
$\boldsymbol{\lambda}_j = [\lambda_{1j}, \dots, \lambda_{kj}]^\top$. Then

$$
\mathbf{x}_j = B\boldsymbol{\lambda}_j
\qquad\Longrightarrow\qquad
\sum_{j=1}^{m} \psi_j \mathbf{x}_j = B\sum_{j=1}^{m} \psi_j \boldsymbol{\lambda}_j
$$

Since $B$ has independent columns, $B\mathbf{v} = \mathbf{0}$ only when
$\mathbf{v} = \mathbf{0}$. So:

> $\{\mathbf{x}_1, \dots, \mathbf{x}_m\}$ are independent **if and only if** their
> recipe vectors $\{\boldsymbol{\lambda}_1, \dots, \boldsymbol{\lambda}_m\}$ are independent.

This lets us test complicated vectors by testing small coefficient vectors
instead.

**Example.** Let $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ be independent and

$$
\mathbf{x}_1 = \mathbf{b}_1 + \mathbf{b}_2, \qquad
\mathbf{x}_2 = \mathbf{b}_2 + \mathbf{b}_3, \qquad
\mathbf{x}_3 = \mathbf{b}_1 + 2\mathbf{b}_2 + \mathbf{b}_3
$$

The recipes are $[1,1,0]^\top$, $[0,1,1]^\top$ and $[1,2,1]^\top$. The third is
the sum of the first two, so the recipes are dependent, and so are the
$\mathbf{x}_j$. Indeed, $\mathbf{x}_1 + \mathbf{x}_2 = \mathbf{x}_3$.

**Counting rule.** If you make $m$ linear combinations of $k$ vectors and
$m > k$, the results are **always dependent**. You cannot get more independent
"directions" out than you put in.

---

## 6. Why this matters in machine learning

| ML situation | Connection to independence |
|---|---|
| Redundant features (e.g. price in USD **and** price in EUR) | dependent columns in the data matrix: one adds no new information |
| Multicollinearity in linear regression | dependent (or nearly dependent) columns make $X^\top X$ non-invertible or unstable |
| Feature selection | keep an independent subset of features |
| Dimensionality reduction | find a few independent directions that explain the data |

---

## Summary

- A **linear combination** is a weighted sum $\sum \lambda_i \mathbf{x}_i$, a
  "recipe" using the vectors as ingredients.
- Vectors are **independent** if the only way to build $\mathbf{0}$ is with all
  weights zero. Otherwise they are **dependent**.
- Independent means **no redundancy**: no vector can be built from the others.
- Any set containing $\mathbf{0}$ or a repeated vector is dependent. So is any
  set of more than $n$ vectors in $\mathbb{R}^n$.
- **Test:** put the vectors in the columns of a matrix and row-reduce. They are
  independent exactly when every column is a pivot column.
- Combinations of independent vectors are independent exactly when their
  coefficient vectors are.

**Previous:** [Vector Spaces](../04-vector%20spaces/) &nbsp;·&nbsp;
**Next:** [Basis and Rank](../06-basis%20and%20rank/)
