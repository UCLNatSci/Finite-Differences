# Higher order problems (examples)

## Problem 1 : A clamped ODE

We will solve the following second order ODE, which has "clamped" boundary conditions

```{math}
:label: 2order
\frac{\mathrm{d}^2 y}{\mathrm{d}x^2} + \frac{\mathrm{d}y}{\mathrm{d}x}-6 y = 0, \qquad y(0)=1, \quad y^{\prime}(0)=0.
```

The problem could be re-written in array form by definining $z=y^{\prime}$ to obtain:

\begin{equation}\frac{\mathrm{d}}{\mathrm{d}x}\left[\begin{array}{cc}z\\y\end{array}\right]=\left[\begin{array}{cc}6y-z\\z\end{array}\right]\end{equation}

Therefore it could be solved using any of the methods for first order IVPs previously illustrated. However, to demonstrate the used of techniques we will need for solving partial differential equations, we will instead proceed with use of finite difference formulae in equation {eq}`2order`.

### Forward difference solution

By using the finite difference formulae {eq}`forward2` and {eq}`forwards1`, we obtain

\begin{equation}
y_{k+2}=(2-h)y_{k+1}+(6h^2+h-1)y_k
\end{equation}

Notice that we need two starting points $y_0$ and $y_1$ for this scheme, since the problem is second order. This is equivalent to defining $y(0)$ and $z(0)$ in the array version of the problem. We can obtain the required values from the given initial conditions $y(0)=1$ and $y^{\prime}(0)=0$.

By using the the first order forward finite difference formula we obtain

\begin{equation}\frac{y_1-y_0}{h^2}=y^{\prime}_0 = 0 \quad \Rightarrow \quad y_1 = y_0=1.\end{equation}

An implementation of the forward stepping method is given below. By comparing to the analytic result it is seen that the solution has the correct shape, but relatively poor accuracy, as expected.

SOLUTION GOES HERE!

### Central difference solution

We can do better using a quadratic order scheme based on central differences. Note that it is not possible to use a central difference scheme for first order IVPs because the first derivative result requires the use of two starting grid points. We can do it for second order IVPs because two starting conditions are given.

By applying the central formulae for the first and second derivatives {eq}`central2` and {eq}`central1a`, we obtain

\begin{equation}y_{k+1} = \frac{y_k(2+6h^2)-y_{k-1}(1-h/2)}{1+h/2}\end{equation}

Using a central differences formula for the boundary condition gives:

\begin{equation}\frac{y_1-y_{-1}}{h} = 0 \quad \Rightarrow \quad y_1= y_{-1}.\end{equation}

However, this relationship involves the solution at the "fictitious" point $y_{-1}$ where $x=-h$. We do not know the result at this fictitious point, but we can proceed by solving the three equations at the left-hand end simultaneously to obtain our starting points. We have

\begin{equation}y_1=-y_{-1}, \quad y_0=1, \quad y_1=\frac{(h-2)y_{-1}+4(3h^2+1)y_0}{h+2}.\end{equation}

Together, these equations give $y_{-1}=1+3h^2$ and so we can start our forward propagation scheme.

SHOW SOLUTION


As an alternative to solving the left-hand points by hand, we can write the full system of equations for all nodes including the fictitious node as a matrix and solve simultaneously using Gaussian elimination. We have

\begin{equation}\left[\begin{array}{c|ccccc}-1 &0 &1 & 0 & \dots & 0&\\\hline0 &1 &0 & 0 & \ddots &\vdots\\1-h/2 & -(2+6h^2) & 1+h/2 & 0 & \ddots &\vdots\\0 & 1-h/2 & -(2+6h^2) & 1+h/2 & \ddots&\vdots\\\vdots& \vdots & \ddots &\ddots &\ddots & 0\\0 & \dots & 0& 1-h/2 & -(2+6h^2) & 1+h/2\end{array}\right]\left[\begin{array}{cc}y_{-1}&\\\hline y_0\phantom{\vdots}&\\y_1&\phantom{\vdots}\\y_2\phantom{\vdots}&\\ \vdots&\\y_N\end{array}\right]=\left[\begin{array}{cc}0&\\\hline1&\phantom{\vdots}\\0&\phantom{\vdots}\\0&\phantom{\vdots}\\\vdots&\\0\end{array}\right]\end{equation}


This system is of the form $AX=B$. If you are struggling to understand where the result comes from, write down the some of the simultaneous equations that you get by computing the product $AX$. You should see that each row in the system gives the equation relating a node to its neighbours using the finite difference formula. The boundary conditions are implemented in the first and second rows.

You can solve the simultaneous system in Python by using the lstsq function from numpy's linear algebra module (linalg).

```{exercise}
Set up matrices $A$ and $B$ for the problem in Python for step $h=10^{-3}$ and solve to obtain the solution for $y$.
```


### Problem 2 : 1D heat equation with Dirichlet boundary conditions
Use the central differences formula, to set up a system of algebraic equations for the value of the nodes $[u_1,u_2,...,u_n]$ in the following problem, ensuring that you enforce the boundary conditions at the endpoints

\begin{equation}u^{\prime\prime}(x)=\sin(2\pi x), \quad u(x_1) = 0, \quad u(x_n)=0.\end{equation}

Solve and plot your solution.


### Problem 3: Simple harmonic motion
We consider the equation of motion for a particle of unit mass moving in one dimension under the influence of a restoring force $F(x)=-x$. The particle is initially at rest at a displacement of $x=1$, and we will solve for motion during the first 100 seconds using discretisation with a time step $\Delta t =0.1$
The equations are given by:

\begin{equation}\frac{\mathrm{d}v}{\mathrm{d}t} = -x, \qquad \frac{\mathrm{d}x}{\mathrm{d}t} = v, \qquad t\in[0,100], \qquad x(0)=1, \qquad v(0)=0.\end{equation}

By using central difference formulas with a half-step $\Delta t/2$, we can obtain:

\begin{equation}v(t+\Delta t/2) = v(t-\Delta t/2)- x(t)\Delta t, \qquad x(t+\Delta t)=x(t)+v(t+\Delta t/2)\Delta t.\end{equation}

This is called a "leapfrog'" method, because the successively computed solutions for $x$ and $v$ are obtained at staggered positions, as illustrated in the schematic diagram below. Use this approach to obtain solutions for $x(t)$ and $v(t+\Delta t/2)$. To get your algorithm started, you can take $v_{-1/2}=0$.

```{image} images/leapfrog.png
:alt: leapfrog algorithm
:height: 200px
:align: center
```

Plot $(v^2+x^2-1)/2$, which provides the difference between the estimated energy and exact (analytic) energy. To calculate $v(t)$ you can use $v(t) = \frac{v(t-\Delta t/2)+v(t+\Delta t/2)}{2}.$

**Answers coming soon!**
