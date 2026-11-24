# Linear Algebra

Algebra gives us a way to describe objects and the rules for working with
them. **Linear algebra** studies vectors, the operations we can perform on
them, and the transformations that preserve their linear structure.

## What is a vector?

A vector is not limited to an arrow or a list of numbers. More generally, it
is an element of a **vector space**: a collection of objects that can be added
together and multiplied by scalars, with these operations obeying consistent
rules. In this introduction, scalars are real numbers.

Different kinds of objects can be vectors:

- **Geometric vectors** can be drawn as arrows. Adding two arrows gives
  another arrow, and multiplying an arrow by a real number changes its length
  and, for a negative scalar, reverses its direction.
- **Polynomials** can be added and scaled, with the result still a
  polynomial. For example, polynomials of degree at most two form a vector
  space: adding or scaling them cannot produce a term of degree greater than
  two.
- **Sampled audio signals** can be represented as finite lists of numbers.
  Adding two signals sample by sample, or scaling every sample by the same
  number, produces another signal of the same length.

These objects may look very different, but their addition and scalar
multiplication follow the same underlying rules. This shared structure lets
us use ideas developed for one kind of vector with other kinds as well.

## Why it matters in machine learning

Machine learning data is often represented as vectors: a row of measurements,
a collection of image values, or a numerical representation of a word. Linear
algebra provides tools for combining these vectors and transforming them.
Matrices are one way to describe such transformations, and they are used
throughout machine learning to represent data and compute model outputs.

The key idea is to look beyond what the objects represent and understand the
operations they support. That viewpoint makes linear algebra useful across
geometry, signals, and machine learning.