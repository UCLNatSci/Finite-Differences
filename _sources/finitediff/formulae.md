---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Additional formulae

We can truncate and manipulate Taylor's series to obtain algebraic approximation for first and second (and higher) derivatives involving weighted averages of neighbouring points. These approximations may allow us to compute derivatives with greater accuracy for a given step size.

## Taylor series recap

We can compute a power series representation for a function $f(x)$ by Taylor expansion about a chosen point $x=a$. If we keep only a finite number of terms then the *truncated* polynomial gives a good approximation in a local neighbourhood of the expansion point.

The "goodness" of the approximation can be anticipated using the Lagrange remainder theorem, which places an upper bound on the size of the error. The error in the truncated expansion is found to be proportional to the next power of $(x-a)$ that was discarded in the expansion.

A mathematical summary of these results is given below, and further details can be found in [NSCI0005 Lecture Notes, Chapter 7 on Taylor Series](https://uclnatsci.github.io/Mathematics-for-Natural-Sciences/sequences_and_series/taylor_series.html)

### Definition

The truncated Taylor expansion of $f(x)$ about point $x=a$, up to and including the term in $(x-a)^n$, is given by

```{math}
:label: finitetaylor
\begin{align}
p_n(x) &= \sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k \\
& = f(a)+f^{\prime}(a)(x-a)+\frac{f^{\prime\prime}(a)}{2}(x-a)^2+\dots +\frac{f^{(n)}(a)}{n!}(x-a)^n.
\end{align}
```
where $f^{(k)}$ denotes the $k$-th derivative of $f$.

### Remainder

According to the Lagrange remainder theorem the error $R_n$ in the truncated expansion satisfies

```{math}
:label: remainder
f(x) = p_n(x) + R_n(x), \qquad R_n(x) = \frac{f^{(n+1)}(\xi)(x-a)^{n+1}}{(n+1)!}, \qquad \xi\in(a,x).
```

```{exercise} (Hand derivation)
Use the Lagrange remainder theorem to compute an upper bound for the size of the error in the quadratic expansion of $\sin(x)$ about $x=\frac{\pi}{3}$, at a nearby point $x=\frac{\pi}{2}$.
```

```{toggle}
The series expansion is found to be:

\begin{equation*}p_2(x)=\frac{\sqrt{3}}{2}+\frac{1}{2}\biggr(x-\frac{\pi}{3}\biggr)-\frac{\sqrt{3}}{4}\biggr(x-\frac{\pi}{3}\biggr)^2\end{equation*}

And from Lagrange's formula, the error in the expansion at $x=\frac{\pi}{2}$ is given by:

\begin{equation*}R(\xi)= \biggr|\frac{\cos(\xi)(\frac{\pi}{2}-\frac{\pi}{3})^3}{6}\biggr| \end{equation*}

Since $|\cos(\xi)|$ is bounded above by $1$ on the given domain, Lagrange's remainder theorem gives an upper bound of

\begin{equation*}\frac{\pi^3}{6^4}=0.0239246\end{equation*}

The exact error is

\begin{equation*}\biggr|\sin\left(\frac{\pi}{2}\right)-p_2\left(\frac{\pi}{2}\right)\biggr| = 0.0091119\end{equation*}

```

## Order of accuracy

Assuming that the Taylor series expansion is to be used in the neighbourhood of the expansion point means that $(x-a)$ is a small quantity. We say that the degree $n$ series has "order $(x-a)^n$ accuracy" and we may write

\begin{equation}f(x)=p_n(x)+\mathcal{O}((x-a))^{n+1}),\end{equation}

where the "big-O" notation describes the size of the error terms.

This finding can be used to describe the size of the error in the finite difference approximation presented in the last section in equation {eq}`finite1`.

Discarding all terms of degree greater than one in Taylor's expansion about $x=x_k$ gives:

\begin{equation}y(x)= y(x_k)+y^{\prime}(x_k)(x-x_k)+\mathcal{O}((x-x_k)^2)\end{equation}

If we label $h=(x-x_k)$, then we may rewrite the expression as shown below. The result is known as Euler's forward difference formula or the explicit Euler method.

\begin{equation}y(x_k+h)= y(x_k)+y^{\prime}(x_k)h +\mathcal{O}(h^2), \qquad h\ll 1\end{equation}

A simple rearrangement of Euler's forward difference formula gives the previously obtained expression for the derivative at $x_k$

\begin{equation}y^{\prime}(x_k)= \frac{y(x_k+h)-y(x_k)}{h}+\mathcal{O}(h)\end{equation}

 The result also shows that the error in this expression is order $h$, as discovered when we plotted the error against the step size.

 The error occurs due to working with finite approximations to infinitesimal quantities, and is a result of truncating the Taylor expansion after just two terms. We should distinguish **truncation error** from the previously seen **roundoff errors** that occur as a computational artefact due to working with finite precision arithmetic.

## Finite difference formulae

We can derive other finite difference formulas that from Taylor's series, by taking weighted sums of expansions about different points to eliminate unwanted terms in the series. Some of them are favoured because they exhibit higher order accuracy than Euler's formula (reduced truncation error), whilst others may be favoured in practical applications for their numeric stability, computational efficiency or ease of practical implementation. A few examples of finite difference formulas are given below, though this list is far from exhaustive.

You may notice that the forward, backward and central differences formulae are simply location-shifted versions of each other. However, this property does not extend to other finite difference formulae.

### First derivative formulae

<br>

**Forward difference** (order $h$ accuracy) :

The result is obtained by expanding taking $x=(x_k+h)$,  $a=x_k$ in the Taylor expansion. It is called forward differences because it uses a forward step (from $x_k$ to $x_{k+1}$) to estimate the derivative.

```{math}
:label: forwards1
\begin{align}
y_{k+1}&=y_k + h y^{\prime}_k + \frac{h^2}{2!}y^{\prime\prime}_k +\frac{h^3}{3!}y^{\prime\prime\prime}_k + \frac{h^4}{4!}y^{(4)}_k + \frac{h^5}{5!}y^{(5)}_k + \dots \qquad \text{where }  y_{k+1}=y(x_k+h)\\
&\Rightarrow \quad y^{\prime}_k= \frac{y_{k+1}-y_k}{h} + h\left[-\frac{1}{2}y^{\prime\prime}_k - \frac{h}{3!}y^{\prime\prime\prime}_k - \frac{h^2}{4!}y^{(4)}_k - \frac{h^3}{5!}y^{(5)}_k + \dots\right]
\end{align}
```

The forward difference formula gives an estimate of the derivative at the interior points $[x_1,x_2,x_3,\dots,x_{n-1}]$. Computing the derivative at $x_n$ requires the function value $y$ at the exterior point $x_{n+1}$ as discussed previously.

<br>

**Backward difference** (order $h$ accuracy) :

The result is obtained by expanding taking $x=x_k$,  $a=(x_k+h)$ in the Taylor expansion. It is called backward differences because it uses a backward step (from $x_k$ to $x_{k-1}$) to estimate the derivative.

```{math}
:label: backwards1
\begin{align}
y_{k-1}&=y_k - h y^{\prime}_k + \frac{h^2}{2!}y^{\prime\prime}_k -\frac{h^3}{3!}y^{\prime\prime\prime}_k + \frac{h^4}{4!}y^{(4)}_k - \frac{h^5}{5!}y^{(5)}_k + \dots \qquad \text{where }  y_{k-1}=y(x_k-h)\\
&\Rightarrow \quad y^{\prime}_k = \frac{y_{k}-y_{k-1}}{h} + h\left[\frac{1}{2}y^{\prime\prime}_k - \frac{h}{3!}y^{\prime\prime\prime}_k + \frac{h^2}{4!}y^{(4)}_k - \frac{h^3}{5!}y^{(5)}_k + \dots\right]
\end{align}
```

The backward difference formula gives an estimate of the derivative at the interior points $[x_2,x_3,\dots,x_{n}]$. Computing the derivative at the $x_1$ requires the function value $y$ at the exterior point $x_{0}$.

<br>

**Central difference** (order $h^2$ accuracy):

The result is obtained by subtracting the backward difference expression {eq}`backwards1` from the forward difference expression {eq}`forwards1`. It is called central differences because it uses both a backward step and a forward step to estimate the derivative.

```{math}
:label: central1a
y^{\prime}_{k} = \frac{y_{k+1} - y_{k-1}}{2h}+h^2\left[-\frac{2}{3}y^{\prime\prime}_k+\dots\right]
```

The central difference formula gives an estimate of the derivative at the interior points $[x_2,x_3,\dots,x_{n-1}]$. Computing the derivative at the two end points $x_1$ and $x_n$ requires the function value $y$ at the exterior points $x_{0}$ and $x_{n+1}$

```{exercise} (hand derivation)

The given results for the first derivative all require only two points to calculate. Can you derive a result from the Taylor series that uses three points $[y_k,y_{k+1},y_{k+2}]$ to calculate an estimate of the first derivative $y^{\prime}(x_k)$ that gives quadratic order accuracy?

**Hint:** Start by expanding $y(x_k+2h)$ and $y(x_{k+h})$ and use a weighted combination of these two expansions that eliminates the second derivative terms.
```

```{toggle}

  $y(x_k+2h)=y(x_k)+2h y^{\prime}(x_k)+\frac{4h^2}{2}y^{\prime\prime}(x_k)+\frac{8h^3}{6}y^{\prime\prime\prime}(x_k)+\dots$

  $y(x_k+h)=y(x_k)+h y^{\prime}(x_k)+h y^{\prime}(x_k)+\frac{h^2}{2}y^{\prime\prime}(x_k)+\frac{h^3}{6}y^{\prime\prime\prime}(x_k)+\dots$

  Subtracting four lots of the second equation from the first gives:

  $y(x_k+2h)-4y(x_k+h)=3y(x_k)-2hy^{\prime}(x_k)+\frac{4h^3}{6}y^{\prime\prime\prime}(x_k)+\dots$

  which rearranges to:

  \begin{equation*}y^{\prime}(x_k)=\frac{-3y(x_k)+4y(x_k+h)-y(x_k+2h)}{2h}+\frac{h^3}{3}y^{\prime\prime\prime}(x_k)\end{equation*}

```


## Second derivative formulae

<br>

**Forward difference** :

The result is obtained by expanding taking $x=(x_k+2h)$,  $a=x_k$ in the Taylor expansion, and then substituting in the result for the first forward difference

```{math}
:label: forward2
\begin{align}
 y_{k+2}&=y_{k}+2hy^{\prime}_{k}+\frac{(2h)^2}{2!}y^{\prime\prime}_{k}+\frac{(2h)^3}{3!}y^{\prime\prime\prime}_k\dots \qquad \text{where } y_{k+2}=y(x_k+2h)\\
 &=y_k+2h\left[\frac{y_{k+1}-y_k}{h}-\frac{h}{2}y_k^{\prime\prime}-\frac{h^2}{6}y_k^{\prime\prime\prime}+\dots\right]+2h^2y_k^{\prime\prime}+\frac{4}{3}h^3 y_k^{\prime\prime\prime}+\dots\\
 &\Rightarrow y^{\prime\prime}_k = \frac{y_{k}-2y_{k+1}+y_{k+2}}{h^2}-hy^{\prime\prime\prime}_k+\dots
 \end{align}
 ```

The formula requires two exterior points on the forward side.

<br>

**Central difference** :

By adding the forward difference expression {eq}`forwards1` to the backward difference expression {eq}`backwards1` we obtain

```{math}
:label: central2
y^{\prime\prime}_k = \frac{y_{k-1}-2y_k+y_{k+1}}{h^2} + h^2\left[-\frac{1}{12}y^{(4)}_k +\dots\right]
```
