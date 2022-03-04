# Junkyard

Let us first import the relevant packages that will be required for the below coding problems:

```{code-cell} ipython3
from math import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
```


## Multi-stage methods - <span style="color: red;">Coding challenge</span>

You can find hidden the various functions described in the previous chapter that are being used to solve the challenge.

```{code-cell} ipython3
---
render:
    image:
        align: center
---

# Heun's method
def my_heun(f,t,x0):

    nstep = len(t)
    h = t[1] - t[0]
    x = np.zeros((len(x0),nstep))
    x[:,0] = x0

    for k in range(nstep-1):
        s1 = f(t[k], x[:,k])
        s2 = f(t[k+1], x[:,k]+[h*a for a in s1])
        x[:,k+1] = x[:,k]+[h/2*a for a in np.add(s1,s2)]

    return x

# Euler's implicit method
def my_euli(f,t,X0):

    n = len(t)
    h = t[1]-t[0]
    X = np.zeros((len(X0),n))
    X[:,0] = X0

    for k in range(n-1):
        Z0 = X[:,k]
        X[:,k+1] = fsolve(lambda Z1: Z1-Z0-[h*a for a in f(t[k], Z1)], Z0)

    return X

# Euler's explicit method
def my_eulr(f,x,y0):

    nstep = len(x)
    h = x[1]-x[0]
    y = np.zeros((len(y0),nstep))
    y[:,0]=y0

    for k in range(nstep-1):
        y[:,k+1] = y[:,k]+[h*a for a in f(x[k], y[:,k])]
    return y
```

```{code-cell} ipython3
---
render:
    image:
        align: center
---
f = lambda t,X: (998*X[0]+1998*X[1],-999*X[0]-1999*X[1])
u0 = 2
v0 = -1
X0 = [u0,v0]

i=0    
fig, ax = plt.subplots(2,2)

for _,h in enumerate([0.0025, 0.001]):
    t = np.arange(0,1,h)
    u = 2*np.exp(-t)

    Xe = my_eulr(f,t,X0)
    Xh = my_heun(f,t,X0)

    ax[0,i].plot(t,Xe[0,:]-u, 'b')
    ax[0,i].set_title('Euler, h=' + str(h))

    ax[1,i].plot(t,Xh[0,:]-u, 'r')
    ax[1,i].set_title('Heun, h=' + str(h))

    i+=1

fig.tight_layout()
plt.show()

# Using implicit Euler
h = 0.1
t = np.arange(0,1,h)
n = len(t)
u = 2*np.exp(-t)
Xi = my_euli(f,t,X0)

plt.plot(t,Xi[0,:]-u,'r')
plt.title('Implicit, h=' + str(h))
plt.show()
```



## Problem 2 : 1D heat equation with Dirichlet boundary conditions
Use the central differences formula, to set up a system of algebraic equations for the value of the nodes $[u_1,u_2,...,u_n]$ in the following problem, ensuring that you enforce the boundary conditions at the endpoints

\begin{equation}u^{\prime\prime}(x)=\sin(2\pi x), \quad u(x_1) = 0, \quad u(x_n)=0.\end{equation}

Solve and plot your solution.


## LEAPFROG: Simple harmonic motion
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

```{code-cell} ipython3
---
render:
    image:
        align: center
---

# Set up grids
ht = 0.1
t = np.arange(0,100,ht)
N = len(t)

x = np.zeros((1,N))[0]
x[0] = 1
v = np.zeros((1,N+1))[0]
v[0] = 0

#Leap frog method
for i in range(N-1):
    v[i+1] = v[i]-x[i]*ht
    x[i+1] = x[i]+v[i+1]*ht

v[N] = v[N-1]-x[N-1]*ht
v = (v[:-1]+v[1:])/2

# Compute differenced from extact energy result
plt.plot(t,abs(v**2+x**2-1)/2,'r')
plt.show()
```

## 1D diffusion
Enceladus is a small, icy moon of Saturn, which has been observed to emit molecules of water from "cracks" in its surface. The following differential equation represents the competition between outward diffusion of water molecules from the surface and destruction due to ionization:
```{math}
	2r\frac{\partial c}{\partial r} + r^2\frac{\partial^2 c}{\partial r^2}=r^2 c \beta_{LOSS}.
```

In this equation, $r$ is the radial distance from the centre of the moon, $c$ is the steady state concentration of water molecules, and $\beta_{LOSS}$ represents the relative importance of ionization compared to diffusion. We will take
\begin{equation*}
	\beta_{LOSS}=0.01, \qquad c(1)=1, \qquad c(50)=0.1,\qquad r\in[1,50].
\end{equation*}

```{exercise}
* Write down  a discretised form of these equations, using a central difference scheme.
*	Solve the discretised system with 50 grid points simultaneously using a matrix method.
*	Solve the discretised system with 50 grid points using a Gauss-Seidel technique. Determine the number of iterations required for agreement with your part (2) solution to within $10^{-4}$.
```

**Solution**

**Part 1: Discretisation**
\begin{equation*}
2r_i\frac{c_{i+1}-c_{i-1}}{2\Delta r}+r_i^2\frac{c_{i+1}-2c_i+c_{i-1}}{\Delta r^2}=\beta_{LOSS} r_i^2 c_i
\end{equation*}

We can also eliminate a factor of $\frac{r_i}{\Delta r^2}$ to give:

\begin{equation*}
(c_{i+1}-c_{i-1})\Delta r +r_i(c_{i+1}-2c_i+c_{i-1})=\beta_{LOSS}\Delta r^2 r_i c_i
\end{equation*}

**Part 2: Matrix method**

Grouping together terms gives (for example):

\begin{equation*}
(r_i-\Delta r)c_{i-1}-(2r_i+\beta_{LOSS}\Delta r^2 r_i)c_i +(r_i+\Delta r)c_{i+1}=0
\end{equation*}

```{code}
n=50; bloss=0.01;   % Values defined in the question

r=linspace(1,50,n); % Set up the radial coordinate
hr=r(2)-r(1);       % Determine the grid spacing

M = zeros(n,n);     % Construct the grid


% CONSTRUCTION OF COEFFICIENTS FOR INTERIOR NODES
for k=2:n-1
    M(k,k-1)=(r(k)-hr);            
    M(k,k)=-(2*r(k)+bloss*hr^2*r(k));
    M(k,k+1)=(r(k)+hr);
end

% alternative approach using diag:
% rint = r(2:n-1);
% diag(rint-hr,-1)+diag([1,-2+bloss*hr^2*rint,1]) + diag(rint+hr,1)

% CONSTRUCTION OF COEFFICIENTS FOR BOUNDARY NODES
M(1,1)=1; M(end,end)=1;

% CONSTRUCTION OF RHS
b=zeros(n,1);b(1)=1;b(n)=0.1; % interior + boundary

sol1=(M\b).';
```

**Part 3: Gauss-Seidel method**

```{code}
% Set up an initial grid for the solution
c=zeros(1,n); c(1) = 1; c(end)=0.1;

for j=1:1000 %bound on maximum number of iterations
    for k=2:n-1
        c(k) = ((r(k)+hr)*c(k+1) +(r(k)-hr)*c(k-1))/(2*r(k)+bloss*hr^2*r(k));
    end
    if norm(c-sol1)<1e-4
         fprintf('Converged after %d iterations,',j)
        break
    end
end
```

```{code}
% Plots
figure
subplot(2,1,1)
plot(r,sol1,'b')
title('Solution by Matrix method')
subplot(2,1,2)
plot(r,c,'r')
title('Solution by Gauss-Seidel method')
```


**Solution**

```{code}
clear;
c = 0.2; g=0.5; L = 2; T = 4; % given parameters

Nt=81; Nx=200;
x = linspace(0,L,Nx);  % discretisation of spatial dimension
t = linspace(0,T,Nt);  % discretisation of spatial parameter
Dt=T/(Nt-1); Dx=L/(Nx-1);

r=c*Dt/Dx              % ensure it's less than 1

F=zeros(Nt,Nx);
F(1,:)=sin(2*pi*x);            %phi(0,x)
F(2,:) = sin(2*pi*(x-c*Dt));   %phi(Dt,x) - see comment at end

%w=sqrt(c^2*(2*pi)^2-g^2/4);
%cp=w/2/pi;
%F(2,:)=exp(-g/2*Dt)*sin(2*pi*(x-cp*Dt));

K0=(1-g*Dt/2); K2=(1+g*Dt/2);  %constants appearing in update rule

for i=2:Nt-1
   f=F(i,:);
    neigh=circshift(f,1)-2*f+circshift(f,-1);
    F(i+1,:)=(2*f+r^2*neigh-K0*F(i-1,:))/K2;
end

surf(x,t(1:5:end),F(1:5:end,:),'EdgeColor','none')
xlabel('x');ylabel('t');zlabel('\phi')

plot(t,max(F,[],2)/max(F(1,:)))
xlabel('t');ylabel('\phi_{max}');

```






## More hyperbolic problems

**Sine-Gordon**

\begin{equation*}
u_{tt}=u_{xx}-\sin(u)
\end{equation*}



**Alternatively, using a simultaneous equations technique**

The function `cdiffM` below was written to solve the Poisson problem

\begin{equation*}
\nabla^2\phi=f
\end{equation*}

using a matrix method for any possible combination of boundary conditions.
The function sets up the coefficient matrix for the problem $A\Phi=F$, where $F,\Phi$ are the function value and unknown solution at each grid node $(x,y)$, arranged as vectors.
Array $A$  is the (sparse) coefficient matrix that relates each node to its neighbours using the five-point formula. The boundary equations are modified to use any specified boundary conditions.


### new section

**Schr\"{o}dinger equation**

\begin{equation*}
y^{\prime\prime}(x)+y(x)=\lambda q(x)y(x), \qquad q(x)=\frac{1}{1+e^{(x-r)/\epsilon}}
\end{equation*}
\begin{equation*}
y(0)=0, \qquad \lim_{x\rightarrow\infty}y(x)=0
\end{equation*}

### General form
Or, to a general second order PDE

\begin{align*}
& A(x,y)\psi_{xx}+B(x,y)\psi_{xy}+C(x,y)\psi_{yy}=f(x,y,\psi,\psi_x,\psi_y),\\
& \psi(x,y)=\alpha(x,y), \quad \underline{n}.\psi = \beta(x,y).
\end{align*}

The solution can only be found in the region between the characteristic drawn through the initial domain $D$. The solution at all points in space is determined by the points they are projected from.

## D'Alembert problem,
Generally use forward stepping (do first!)

$x=0: \quad u(0,0)=2e^{-L^2/4}$
$x=L: \quad u(0,L)=2e^{-L^2/4}$

(both very small)
