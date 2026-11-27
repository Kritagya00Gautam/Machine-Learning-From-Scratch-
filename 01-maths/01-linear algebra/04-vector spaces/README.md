# Vector Spaces

So far we have worked with systems of linear equations, written compactly as
$A\mathbf{x} = \mathbf{b}$, and learned how to solve them. Now we step back and
ask: **what exactly is the space that vectors live in?**

At the start of linear algebra, vectors were described informally as objects
that can be **added** together and **multiplied by a scalar**, and the result is
again an object of the same kind. A **vector space** turns this idea into a
precise definition.

To get there we build up in three steps:

1. **Groups:** a set with one operation that behaves well.
2. **Vector spaces:** a group with an extra operation, scaling by numbers.
3. **Vector subspaces:** smaller vector spaces sitting inside a bigger one.

> Based on Section 2.4 of *Mathematics for Machine Learning* (Deisenroth,
> Faisal and Ong), `mml-book.pdf` in the `01-maths` folder. The explanations and
> examples here are written for these notes.

---

## 1. Groups

A **group** is a set together with one operation that combines two elements
into a third, while keeping some useful structure intact. Groups are used a lot
in computer science, for example in cryptography, coding theory and computer
graphics.

### Definition

Let $\mathcal{G}$ be a set and $\otimes : \mathcal{G} \times \mathcal{G} \to \mathcal{G}$
an operation on it. Then $G := (\mathcal{G}, \otimes)$ is a **group** if these
four rules hold:

| # | Rule | Statement | Meaning |
|---|---|---|---|
| 1 | **Closure** | $\forall x, y \in \mathcal{G}:\; x \otimes y \in \mathcal{G}$ | combining two elements never leaves the set |
| 2 | **Associativity** | $\forall x, y, z \in \mathcal{G}:\; (x \otimes y) \otimes z = x \otimes (y \otimes z)$ | brackets don't matter |
| 3 | **Neutral element** | $\exists e \in \mathcal{G}\; \forall x \in \mathcal{G}:\; x \otimes e = e \otimes x = x$ | there is a "do nothing" element |
| 4 | **Inverse element** | $\forall x \in \mathcal{G}\; \exists y \in \mathcal{G}:\; x \otimes y = y \otimes x = e$ | every element can be undone |

The inverse of $x$ is often written $x^{-1}$.

> **Careful:** $x^{-1}$ means "the inverse *with respect to the operation
> $\otimes$*". It is **not** necessarily $\frac{1}{x}$. For example, under
> addition the inverse of $5$ is $-5$.

### Abelian groups

If, in addition, the order doesn't matter,

$$
\forall x, y \in \mathcal{G}:\; x \otimes y = y \otimes x
$$

then the group is called **Abelian** (or **commutative**).

### Examples: group or not?

| Set and operation | Group? | Reason |
|---|---|---|
| $(\mathbb{Z}, +)$ | ✅ Abelian group | neutral element $0$, the inverse of $z$ is $-z$ |
| $(\mathbb{N}_0, +)$ with $\mathbb{N}_0 = \{0, 1, 2, \dots\}$ | ❌ | has neutral element $0$, but no inverses: there is no $y \in \mathbb{N}_0$ with $3 + y = 0$ |
| $(\mathbb{Z}, \cdot)$ | ❌ | neutral element $1$, but only $\pm 1$ have inverses: $\frac{1}{2} \notin \mathbb{Z}$ |
| $(\mathbb{R}, \cdot)$ | ❌ | $0$ has no inverse, since $0 \cdot y = 1$ is impossible |
| $(\mathbb{R} \setminus \{0\}, \cdot)$ | ✅ Abelian group | remove $0$ and every $x$ has inverse $\frac{1}{x}$ |
| $(\mathbb{R}^n, +)$, $(\mathbb{Z}^n, +)$ | ✅ Abelian group | addition component by component (see below) |
| $(\mathbb{R}^{m \times n}, +)$ | ✅ Abelian group | matrices added entry by entry |

For $\mathbb{R}^n$, addition is done component by component:

$$
(x_1, \dots, x_n) + (y_1, \dots, y_n) = (x_1 + y_1, \dots, x_n + y_n)
$$

The neutral element is $\mathbf{0} = (0, \dots, 0)$, and the inverse of
$(x_1, \dots, x_n)$ is $(-x_1, \dots, -x_n)$.

### Square matrices under multiplication

Now consider $(\mathbb{R}^{n \times n}, \cdot)$, square matrices with matrix
multiplication:

- **Closure** and **associativity** follow from the definition of matrix multiplication.
- **Neutral element:** the identity matrix $I_n$, since $AI_n = I_nA = A$.
- **Inverse element:** only exists when $A$ is invertible (regular). Singular
  matrices have no inverse.

So all square matrices together do **not** form a group. But if we keep only the
invertible ones, we do get a group.

**Definition (General linear group).** The set of all invertible matrices
$A \in \mathbb{R}^{n \times n}$, with matrix multiplication, is a group called
the **general linear group** $GL(n, \mathbb{R})$.

Because matrix multiplication is **not commutative** ($AB \neq BA$ in general),
$GL(n, \mathbb{R})$ is **not Abelian**.

---

## 2. Vector spaces

A group has only one operation, an **inner operation** that combines two
elements of the set. A vector space adds a second, **outer operation**:
multiplying an element of the set by a scalar from outside the set, a real
number $\lambda \in \mathbb{R}$.

- Inner operation $+$: think of it as **addition**.
- Outer operation $\cdot$: think of it as **scaling**.

> "Inner" and "outer" operation here have **nothing to do** with the inner
> product or outer product of vectors.

### Definition

A **real-valued vector space** $V = (\mathcal{V}, +, \cdot)$ is a set
$\mathcal{V}$ with two operations

$$
+ : \mathcal{V} \times \mathcal{V} \to \mathcal{V}
\qquad\qquad
\cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}
$$

such that:

1. **$(\mathcal{V}, +)$ is an Abelian group.** Addition is closed, associative
   and commutative, there is a zero vector, and every vector has a negative.
2. **Distributivity:**
   - $\forall \lambda \in \mathbb{R},\; \mathbf{x}, \mathbf{y} \in \mathcal{V}:\;
     \lambda \cdot (\mathbf{x} + \mathbf{y}) = \lambda \cdot \mathbf{x} + \lambda \cdot \mathbf{y}$
   - $\forall \lambda, \psi \in \mathbb{R},\; \mathbf{x} \in \mathcal{V}:\;
     (\lambda + \psi) \cdot \mathbf{x} = \lambda \cdot \mathbf{x} + \psi \cdot \mathbf{x}$
3. **Associativity of scaling:**
   $\forall \lambda, \psi \in \mathbb{R},\; \mathbf{x} \in \mathcal{V}:\;
   \lambda \cdot (\psi \cdot \mathbf{x}) = (\lambda\psi) \cdot \mathbf{x}$
4. **Neutral element of scaling:**
   $\forall \mathbf{x} \in \mathcal{V}:\; 1 \cdot \mathbf{x} = \mathbf{x}$

### Vocabulary

| Term | Meaning |
|---|---|
| **vectors** | the elements $\mathbf{x} \in \mathcal{V}$ |
| **zero vector** $\mathbf{0}$ | the neutral element of $(\mathcal{V}, +)$; in $\mathbb{R}^n$ it is $[0, \dots, 0]^\top$ |
| **vector addition** | the inner operation $+$ |
| **scalars** | the numbers $\lambda \in \mathbb{R}$ |
| **multiplication by scalars** | the outer operation $\cdot$ |

> **Scalar multiplication ≠ scalar product.** Multiplying a vector by a scalar
> gives a vector. The *scalar product* (dot product) of two vectors gives a
> number and is a different thing, covered in analytic geometry (Chapter 3).

### The key idea

A vector space is any collection of objects that you can **add** and **scale**,
where the result always stays in the collection and the usual algebra rules
hold. The objects don't have to be arrows or lists of numbers. That is why the
same theory applies to matrices, polynomials, functions and more.

### There is no "vector times vector"

For $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$, a product $\mathbf{a}\mathbf{b}$
is **not defined** as a vector space operation.

- One *could* define element-wise multiplication, $c_j = a_j b_j$. This
  "array multiplication" is common in programming, but it makes little sense
  mathematically under the matrix multiplication rules.
- If we treat vectors as $n \times 1$ matrices, then $\mathbf{a}\mathbf{b}$
  means $(n \times 1)(n \times 1)$, and the inner dimensions don't match.

Only two vector-vector products fit the matrix rules:

| Product | Shapes | Result |
|---|---|---|
| **outer product** $\mathbf{a}\mathbf{b}^\top$ | $(n \times 1)(1 \times n)$ | a matrix in $\mathbb{R}^{n \times n}$ |
| **inner / scalar / dot product** $\mathbf{a}^\top\mathbf{b}$ | $(1 \times n)(n \times 1)$ | a number in $\mathbb{R}$ |

**Example.** For $\mathbf{a} = [1, 2]^\top$ and $\mathbf{b} = [3, 4]^\top$:

$$
\mathbf{a}^\top\mathbf{b} = 1\cdot 3 + 2\cdot 4 = 11,
\qquad
\mathbf{a}\mathbf{b}^\top =
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
\begin{bmatrix} 3 & 4 \end{bmatrix}
=
\begin{bmatrix} 3 & 4 \\ 6 & 8 \end{bmatrix}
$$

### Examples of vector spaces

**1. $V = \mathbb{R}^n$.** The most important one. Operations are done
component by component:

- Addition: $\mathbf{x} + \mathbf{y} = (x_1 + y_1, \dots, x_n + y_n)$
- Scaling: $\lambda\mathbf{x} = (\lambda x_1, \dots, \lambda x_n)$

For example, in $\mathbb{R}^3$:

$$
\begin{bmatrix} 1 \\ -2 \\ 0 \end{bmatrix} + \begin{bmatrix} 4 \\ 1 \\ 3 \end{bmatrix}
= \begin{bmatrix} 5 \\ -1 \\ 3 \end{bmatrix},
\qquad
3\begin{bmatrix} 1 \\ -2 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ -6 \\ 0 \end{bmatrix}
$$

**2. $V = \mathbb{R}^{m \times n}$.** Matrices are vectors too. Addition is
entry by entry,

$$
A + B =
\begin{bmatrix}
a_{11} + b_{11} & \cdots & a_{1n} + b_{1n} \\
\vdots & & \vdots \\
a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn}
\end{bmatrix}
$$

and scaling multiplies every entry by $\lambda$, as in the matrices section. An
$m \times n$ matrix is just $mn$ numbers arranged in a grid, so
$\mathbb{R}^{m \times n}$ behaves exactly like $\mathbb{R}^{mn}$.

**3. $V = \mathbb{C}$.** The complex numbers, with the usual addition of complex
numbers and scaling by real numbers.

**4. Polynomials (an extra example).** All polynomials of degree at most 2,
$p(t) = a_0 + a_1 t + a_2 t^2$. Adding two of them or scaling one gives another
polynomial of degree at most 2, and the zero polynomial is the zero vector. So
this is a vector space, even though its "vectors" are functions.

### Notation conventions

- When $+$ and $\cdot$ are the standard operations, we simply write $V$ for the
  vector space $(\mathcal{V}, +, \cdot)$ and $\mathbf{x} \in V$ for its vectors.
- $\mathbb{R}^n$, $\mathbb{R}^{n \times 1}$ and $\mathbb{R}^{1 \times n}$
  differ only in **how we write** the vectors. We treat $\mathbb{R}^n$ and
  $\mathbb{R}^{n \times 1}$ as the same thing, so an $n$-tuple is written as a
  **column vector**:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}
$$

- We **do** distinguish column vectors ($\mathbb{R}^{n \times 1}$) from **row
  vectors** ($\mathbb{R}^{1 \times n}$), so that matrix multiplication stays
  unambiguous. By default $\mathbf{x}$ is a column vector, and the row vector is
  its **transpose** $\mathbf{x}^\top$.

---

## 3. Vector subspaces

### Intuition

A **vector subspace** is a set **inside** a vector space that is a vector space
in its own right. If you add or scale vectors that are in the subspace, you
**never leave it**. In this sense a subspace is **closed**.

Subspaces are a key idea in machine learning. For example, dimensionality
reduction methods such as PCA (Chapter 10) project data onto a lower-dimensional
subspace.

### Definition

Let $V = (\mathcal{V}, +, \cdot)$ be a vector space and let
$\mathcal{U} \subseteq \mathcal{V}$ with $\mathcal{U} \neq \emptyset$. Then
$U = (\mathcal{U}, +, \cdot)$ is a **vector subspace** (or **linear subspace**)
of $V$ if $U$ is itself a vector space, using the same operations $+$ and $\cdot$
restricted to $\mathcal{U}$. We write

$$
U \subseteq V
$$

### How to check whether something is a subspace

Since $\mathcal{U}$ sits inside $\mathcal{V}$, it automatically **inherits** most
vector space rules from $V$: commutativity, associativity, distributivity and
the neutral element of scaling. These hold for every vector in $V$, so they hold
for the vectors in $U$ too.

That leaves only **three things to check**:

| # | Check | Statement |
|---|---|---|
| 1 | **Not empty, and contains zero** | $\mathbf{0} \in \mathcal{U}$ |
| 2a | **Closed under scaling** | $\forall \lambda \in \mathbb{R},\; \forall \mathbf{x} \in \mathcal{U}:\; \lambda\mathbf{x} \in \mathcal{U}$ |
| 2b | **Closed under addition** | $\forall \mathbf{x}, \mathbf{y} \in \mathcal{U}:\; \mathbf{x} + \mathbf{y} \in \mathcal{U}$ |

If any one fails, $U$ is **not** a subspace.

> **Quick test:** if a set does not contain $\mathbf{0}$, it is **not** a
> subspace. Done.

### Examples in $\mathbb{R}^2$

| Set | Subspace? | Why |
|---|---|---|
| $\{\mathbf{0}\}$ | ✅ | the **trivial** subspace |
| $\mathbb{R}^2$ itself | ✅ | every space is a subspace of itself (also trivial) |
| a line through the origin, e.g. $\{(x, 2x) : x \in \mathbb{R}\}$ | ✅ | contains $\mathbf{0}$; sums and multiples stay on the line |
| a line **not** through the origin, e.g. $\{(x, 2x + 1)\}$ | ❌ | does not contain $\mathbf{0}$ |
| the first quadrant $\{(x, y) : x \geq 0, y \geq 0\}$ | ❌ | not closed under scaling: $-1 \cdot (1, 1) = (-1, -1)$ leaves the set |
| two lines through the origin, e.g. the $x$-axis $\cup$ the $y$-axis | ❌ | not closed under addition: $(1, 0) + (0, 1) = (1, 1)$ is on neither axis |
| a disc $\{(x, y) : x^2 + y^2 \leq 1\}$ | ❌ | not closed under scaling: $(1, 0) \cdot 5$ leaves the disc |

In $\mathbb{R}^2$, the only subspaces are $\{\mathbf{0}\}$, the lines through the
origin, and $\mathbb{R}^2$ itself. In $\mathbb{R}^3$, they are $\{\mathbf{0}\}$,
the lines and planes through the origin, and $\mathbb{R}^3$ itself.

### Worked check

Is $U = \{(x_1, x_2, x_3) \in \mathbb{R}^3 : x_1 + x_2 - x_3 = 0\}$ a subspace of
$\mathbb{R}^3$?

1. **Zero:** $0 + 0 - 0 = 0$, so $\mathbf{0} \in U$. ✓
2. **Scaling:** if $x_1 + x_2 - x_3 = 0$, then
   $\lambda x_1 + \lambda x_2 - \lambda x_3 = \lambda \cdot 0 = 0$, so $\lambda\mathbf{x} \in U$. ✓
3. **Addition:** if $\mathbf{x}, \mathbf{y} \in U$, then
   $(x_1 + y_1) + (x_2 + y_2) - (x_3 + y_3) = 0 + 0 = 0$, so $\mathbf{x} + \mathbf{y} \in U$. ✓

So $U$ is a subspace. Geometrically it is a plane through the origin.

Now change the equation to $x_1 + x_2 - x_3 = 1$. Then $0 + 0 - 0 = 0 \neq 1$, so
$\mathbf{0} \notin U$, and $U$ is **not** a subspace.

### Subspaces and systems of linear equations

These two examples point to a general fact connecting this section to the
previous one:

- The solution set of a **homogeneous** system $A\mathbf{x} = \mathbf{0}$ with
  $\mathbf{x} \in \mathbb{R}^n$ **is a subspace** of $\mathbb{R}^n$. If
  $A\mathbf{x} = \mathbf{0}$ and $A\mathbf{y} = \mathbf{0}$, then
  $A(\mathbf{x} + \mathbf{y}) = \mathbf{0}$ and $A(\lambda\mathbf{x}) = \mathbf{0}$,
  and $A\mathbf{0} = \mathbf{0}$.
- The solution set of an **inhomogeneous** system $A\mathbf{x} = \mathbf{b}$ with
  $\mathbf{b} \neq \mathbf{0}$ is **not a subspace**, because
  $A\mathbf{0} = \mathbf{0} \neq \mathbf{b}$, so $\mathbf{0}$ is not a solution.
  (It is a *shifted* subspace, an **affine space**, covered later.)
- The **intersection** of any number of subspaces is again a subspace.
  (A **union** generally is not, as the two-axes example shows.)
- **Converse:** every subspace $U \subseteq \mathbb{R}^n$ is the solution set of
  some homogeneous system $A\mathbf{x} = \mathbf{0}$.

This is why the "general solution = particular solution + homogeneous solutions"
recipe works so well. The homogeneous solutions form a subspace, and the
particular solution shifts it into place.

---

## 4. Why this matters in machine learning

| ML idea | Vector space connection |
|---|---|
| Feature vectors | each data point is a vector in $\mathbb{R}^D$ |
| Images, weight matrices | elements of $\mathbb{R}^{m \times n}$, which is itself a vector space |
| Dimensionality reduction (PCA) | find a low-dimensional **subspace** that captures most of the data |
| Embeddings (words, images) | meaning is encoded by positions and directions in a vector space |
| Linear models | predictions are linear combinations, which stay inside a vector space |

---

## Summary

- A **group** $(\mathcal{G}, \otimes)$ has closure, associativity, a neutral
  element and inverses. If it is also commutative, it is **Abelian**.
- The invertible $n \times n$ matrices form the **general linear group**
  $GL(n, \mathbb{R})$, which is **not** Abelian.
- A **vector space** is an Abelian group under addition, plus scaling by real
  numbers that satisfies distributivity, associativity and $1 \cdot \mathbf{x} = \mathbf{x}$.
- $\mathbb{R}^n$, $\mathbb{R}^{m \times n}$ and $\mathbb{C}$ are vector spaces.
  "Vectors" can be any objects you can add and scale.
- There is no "vector times vector" operation in a vector space. Only the
  **inner product** $\mathbf{a}^\top\mathbf{b}$ (a number) and the **outer product**
  $\mathbf{a}\mathbf{b}^\top$ (a matrix) fit the matrix rules.
- A **subspace** is a subset that contains $\mathbf{0}$ and is closed under
  addition and scaling.
- Solutions of $A\mathbf{x} = \mathbf{0}$ form a subspace. Solutions of
  $A\mathbf{x} = \mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$ do not.

**Previous:** [Solving Systems of Linear Equations](../03-solving%20systems%20of%20linear%20equations/) &nbsp;·&nbsp;
**Next:** [Linear Independence](../05-linear%20independence/)
