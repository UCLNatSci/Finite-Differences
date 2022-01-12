# Higher order problems (examples)

### Problem 1 : A clamped ODE
We will use a quadratic order scheme based on central differences to solve the problem

\begin{equation}\frac{\mathrm{d}^2 y}{\mathrm{d}x^2} + \frac{\mathrm{d}y}{\mathrm{d}x}-6 y = 0, \qquad y(0)=1, \quad y^{\prime}(0)=0.\end{equation}

By applying the central formulae for the first and second derivatives, and rearranging, we obtain

\begin{equation}y_{k+1} = \frac{y_k(2+6h^2)-y_{k-1}(1-h/2)}{1+h/2}\end{equation}

Using a central differences formula for the boundary condition gives:

\begin{equation}\frac{y_1-y_{-1}}{h} = 0 \quad \Rightarrow \quad y_1= y_{-1}.\end{equation}

However, this relationship involves the solution at the "fictitious" point where $x=-h$. We do not know the result at this fictitious point, but we can proceed by writing the full system of equations for all nodes including the fictitious node as a matrix and solving simultaneously using Gaussian elimination. We have

\begin{equation}\left[\begin{array}{c|ccccc}-1 &0 &1 & 0 & \dots & 0&\\\hline0 &1 &0 & 0 & \ddots &\vdots\\1-h/2 & -(2+6h^2) & 1+h/2 & 0 & \ddots &\vdots\\0 & 1-h/2 & -(2+6h^2) & 1+h/2 & \ddots&\vdots\\\vdots& \vdots & \ddots &\ddots &\ddots & 0\\0 & \dots & 0& 1-h/2 & -(2+6h^2) & 1+h/2\end{array}\right]\left[\begin{array}{cc}y_{-1}&\\\hline y_0\phantom{\vdots}&\\y_1&\phantom{\vdots}\\y_2\phantom{\vdots}&\\ \vdots&\\y_N\end{array}\right]=\left[\begin{array}{cc}0&\\\hline1&\phantom{\vdots}\\0&\phantom{\vdots}\\0&\phantom{\vdots}\\\vdots&\\0\end{array}\right]\end{equation}

```{admonition} Hint
:class: tip
This system is of the form $AX=B$ if you are struggling to understand where the result comes from, write down the some of the simultaneous equations that you get by computing the product $AX$. You should see that each row in the system gives the equation relating a node to its neighbours using the finite difference formula. The boundary conditions are implemented in the first and second rows.

You can solve the simultaneous system in Python by using the lstsq function from numpy's linear algebra module (linalg).

Set up this problem in Python (construct the matrix), for step $h=10^{-3}$ and solve to obtain the solution for $y$.
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
