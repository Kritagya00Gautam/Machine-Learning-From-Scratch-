# Systems of Linear Equations

A **linear equation** is an equation in which every variable has degree one.
For example, a linear equation in two variables can be written as:

$$ax + by = c$$

A **system of linear equations** is a collection of linear equations that are
considered together. Finding a solution means finding values that satisfy every
equation in the system.

Systems of linear equations are used throughout machine learning. Many
problems can be expressed in this form and then solved using linear algebra
tools.

## Solutions in two dimensions

For a system with two variables, \(x_1\) and \(x_2\), each equation describes a
line in the \(x_1x_2\)-plane. The solution set is the intersection of all the
lines.

There are three possible cases:

1. **One solution:** the lines intersect at a single point.
2. **Infinitely many solutions:** every equation describes the same line, so
   every point on that line is a solution.
3. **No solution:** the lines are parallel and never intersect.

In a two-dimensional plane, the intersection of a system of lines can therefore
be a point, a line, or the empty set.

## Matrix representation

To solve a system systematically, collect its coefficients into vectors and
matrices. Consider the system:

$$
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1
$$

$$
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2
$$

$$
\vdots
$$

$$
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
$$

This system can be written compactly as a matrix-vector equation:

$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{bmatrix}
$$

In this notation, the coefficient matrix is \(A\), the variable vector is
\(\mathbf{x}\), and the constant vector is \(\mathbf{b}\). The system becomes:

$$A\mathbf{x} = \mathbf{b}$$

Matrix notation allows us to solve systems using a consistent set of
algebraic rules. The following sections cover matrices and the methods used to
solve systems of linear equations.
