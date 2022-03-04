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

# Dirichlet - PDE


**Alternatively, using a simultaneous equations technique**

The function `cdiffM` below was written to solve the Poisson problem

\begin{equation*}
\nabla^2\phi=f
\end{equation*}

using a matrix method for any possible combination of boundary conditions.
The function sets up the coefficient matrix for the problem $A\Phi=F$, where $F,Phi$ are the function value and unknown solution at each grid node $(x,y)$, arranged as vectors.
Array $A$  is the (sparse) coefficient matrix that relates each node to its neighbours using the five-point formula. The boundary equations are modified to use any specified boundary conditions.


### Neumann

Examples

\begin{equation*}
y^{\prime\prime}+y=0, \qquad y^{\prime}(a)=\alpha, \quad y^{\prime}(b)=\beta
\end{equation*}

\begin{equation*}
\nabla^2 y + y = 0 \qquad \nabla y(x).\hat{\underline{n}}=f(\underline{x}), \ \forall \underline{x} \in \partial\underline{\Omega}
\end{equation*}

Uses fictitious nodes at each end (give single example)

### new section

**Schr\"{o}dinger equation**

\begin{equation*}
y^{\prime\prime}(x)+y(x)=\lambda q(x)y(x), \qquad q(x)=\frac{1}{1+e^{(x-r)/\epsilon}}
\end{equation*}
\begin{equation*}
y(0)=0, \qquad \lim_{x\rightarrow\infty}y(x)=0
\end{equation*}
