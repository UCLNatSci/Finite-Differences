# Boundary conditions


## Dirichlet: Solution specified on each boundary

Examples:

\begin{equation*}
y^{\prime\prime}+y=0, \qquad y(a)=\alpha, \quad y(b)=\beta
\end{equation*}

\begin{equation*}
\nabla^2 y + y = 0 \qquad y(x)=f(x), \ \forall x \in \partial\Omega
\end{equation*}

Eg:
fluid dynamics - no slip: zero velocity relative to the boundary
thermodynamics - surface held at constant temperature

**Worked example: 1D heat equation**

\begin{equation*}
u^{\prime\prime}(x)=\sin(2\pi x), \qquad u(x_1)=0, \quad u(x_n)=0
\end{equation*}

**Worked example: 2D heat equation**

\begin{equation*}
\nabla^2\psi=0\end
{equation*}

Plate:
* $\phi=300$ (Left,Right)
* $\phi=400$ (Top,Bottom)

Radial version:
\begin{equation*}
\frac{\partial^2\phi}{\partial r^2}+\frac{1}{r}\frac{\partial\phi}{\partial r}=0, \qquad \phi(0.05)=400, \quad \phi(0.10)=300
\end{equation*}

Enceladus:
\begin{equation*}
2r \frac{\partial c}{\partial r}+r^2\frac{\partial^2 c}{\partial r^2}=r^2 x \beta_{loss}, \qquad c(1)-1, \quad c(50)=0.1, \quad r\in[1,50], \quad \beta_{loss=0.01}
\end{equation*}

## Neumann

Examples

\begin{equation*}
y^{\prime\prime}+y=0, \qquad y^{\prime}(a)=\alpha, \quad y^{\prime}(b)=\beta
\end{equation*}

\begin{equation*}
\nabla^2 y + y = 0 \qquad \nabla y(x).\hat{\underline{n}}=f(\underline{x}), \ \forall \underline{x} \in \partial\underline{\Omega}
\end{equation*}

Uses fictitious nodes at each end (give single example)


## new section

**Schr\"{o}dinger equation**

\begin{equation*}
y^{\prime\prime}(x)+y(x)=\lambda q(x)y(x), \qquad q(x)=\frac{1}{1+e^{(x-r)/\epsilon}}
\end{equation*}
\begin{equation*}
y(0)=0, \qquad \lim_{x\rightarrow\infty}y(x)=0
\end{equation*}

**Sine-Gordon**

\begin{equation*}
u_{tt}=u_{xx}-\sin(u)
\end{equation*}


**D'Alembert wave**

\begin{equation*}
u_{tt}=c^2 u_{xx}, \qquad y(x,0)=\phi(x), \quad u_t(x,0)=\psi(x)
\end{equation*}

Hyperbolic in $(x,t)$ plane

## Cauchy: Function and normal derivative on each boundary

Examples:

\begin{equation*}
y^{\prime\prime}=f(y(s),y^{\prime}(s),s), \qquad y(a)=\alpha, \quad y^{\prime}(b)=\beta
\end{equation*}

\begn{equation*}
A(x,y)\psi_{xx}+B(x,y)\psy_{xy}+C(x,y)\psi_{yy}=f(x,y,\psi,\psi_x,psi_y), \qquad \psi(x,y)=\alpha(x,y), \quad \underline{n}.\psi = \beta(x,y)
\end{equation*}

**Worked example: ODE**

\begin{equation*}
y^{\prime\prime}+y^{\prime}-6y=0, \qquad y(0)=1, \quad y^{\prime}(0)=0
\end{equation*}

**Worked example**

Cauchy in time, Dirichlet in space

\begin{multline*}u_{tt}=c^2 u_{xx}, \qquad t\in(0,T),x\in(0,L)\\
u(x,0)=2e^{-(x-L/2)^2}, \quad u_t(0,x)=0, \quad u(t,0)=0, \quad u(t,L)=0
\end{multline*}

Generally use forward stepping (do first!)

$x=0: \quad u(0,0)=2e^{-L^2/4}$
$x=L: \quad u(0,L)=2e^{-L^2/4}$

(both very small)

## new section

**Damped 1D wave**
Hard: leave as "extra"

\begin{equation*}
\phi_{tt}+\gamma \phi_t =c^2 \phi_{xx}, \qquad \phi(t=0)=\sin(2\pi x), \quad \phi(t=\Delta t)=\sin(2\pi(x-c\Delta t))
\end{equation*}
