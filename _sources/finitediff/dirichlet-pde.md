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

**Examples:**

\begin{equation*}
\nabla^2 y + y = 0 \qquad y(x)=f(x), \ \forall x \in \partial\Omega
\end{equation*}

**Worked example: 2D heat equation**

\begin{equation*}
\nabla^2\psi=0
\end{equation*}

Plate:
* $\phi=300$ (Left,Right)
* $\phi=400$ (Top,Bottom)

### Problem 2 : 1D heat equation with Dirichlet boundary conditions

```{code-cell} ipython3
---
render:
    image:
        align: center
---

n=100
x = np.linspace(0,1,n).T
h = x[1]-x[0]

f = lambda x: sin(2*pi*x)

A = np.diag([-2]*n,0)+np.diag([1]*(n-1),1)+np.diag([1]*(n-1),-1)

A[0,[0,1]]=[1,0]
A[n-1,[n-2,n-1]]=[0,1]
F =[f(a) for a in x]
F[0]=0
F[len(F)-1] = 0

U = [h**2*a for a in np.linalg.lstsq(A,F,rcond=None)[0]]
sol = [-sin(2*pi*a)/(4*pi**2) for a in x]

fig,ax = plt.subplots(1,2)
ax[0].plot(x,U)

U = np.array(U)
sol = np.array(sol)
ax[1].plot(x,abs(U-sol))
plt.show()
```

### Illustration of approach using simultaneous equations

Consider a rectangular plate, with temperature $\phi(x,y)$ described by the steady heat equation $\nabla^2\phi = 0$. The plate is maintained at boundary temperatures $\phi = 300, 400$ on pairs of opposite edges as shown. We will solve this problem using a central finite difference scheme on a (3 × 3) grid of interior points.

<br>

```{image} images/3_3_grid.png
:alt: 3x3 grid
:scale: 100%
:align: center
```
<br>

The governing equation can be discretised using a five-point stencil:

$$ \phi_{i+1,j}+\phi_{i-1,j}+\phi_{i,j+1}+\phi_{i,j-1}-4\phi_{i,j}=0$$

There is one equation for each node (equation variable), so we number the nodes sequentially $\phi_{1,...9)$ as illustrated, resulting in the following system of equations :

$$
\begin{bmatrix}
-4 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & -4 & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & -4 & 0 & 0 & 1 & 0 & 0 & 0 \\
1 & 0 & 0 & -4 & 1 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 1 & -4 & 1 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 1 & -4 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & -4 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 1 & -4 & 1 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & -4
\end{bmatrix}
\begin{bmatrix}
\phi_1\\\phi_2\\\phi_3\\\phi_4\\\phi_5\\\phi_6\\\phi_7\\\phi_8\\\phi_9
\end{bmatrix}
=
\begin{bmatrix}
-700\\-400\\-700\\-300\\0\\-300\\-700\\-400\\-700
\end{bmatrix}
$$

Notice that as a consequence of reshaping the array of node values into a column it has become slightly harder to keep track of the nodes that are neighbours. Additionally, inverting the sparse coefficient matrices can be computationally impractical for problems with many nodes. We therefore introduce an alternative, below.

### Relaxation Method

We begin by rearranging the governing finite difference formula into the form $$u=F(u).$$ For instance, for the five-point formula given in the previous example, we may write

```{math}
:label: discr
u_{i,j} = \frac{1}{4}\left(u_{i,j-1}+u_{i-1,j}+u_{i+1,j}+u_{i,j+1}\right)
```


## 2D heat equation with Dirichlet conditions
We will consider the problem
\begin{equation}
\Delta u=f(x,y), \qquad 0\leq x,y\leq 1, \qquad \Delta = \frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}.
\end{equation}

By applying the second derivative central difference formula at the interior points:

\begin{equation*}
%\begin{rcases}
u_{xx}(x_i,y_j) = \frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{h^2},\quad
u_{yy}(x_i,y_j) = \frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{h^2}.
%\end{rcases}2\leq i,j\leq N-1
\end{equation*}

we obtain
\begin{equation}\label{fivept}
u_{i,j-1}+u_{i-1,j}-4u_{i,j}+u_{i+1,j}+u_{i,j+1}=h^2 f(x_i,y_j)
\end{equation}
This is known as the five-point formula. It can be represented using a stencil:

<br>

```{image} images/stencil.png
:alt: secant
:align: center
:scale: 50%
```
<br>

The equation rearranges to the following iterative formula:
\begin{equation}\label{discr}
u_{i,j} = \frac{1}{4}\left(u_{i,j-1}+u_{i-1,j}+u_{i+1,j}+u_{i,j+1}-h^2 f(x_i,y_j)\right)
\end{equation}

**Step 1:**
Solve the problem using an iterative solution technique, for the case where:
\begin{equation*}
f(x,y) = 2\cos(\pi y)-\pi^2(1+x)^2\cos(\pi y),
\end{equation*}
subject to the conditions
\begin{align*}
u(x,0)=(1+x)^2, \quad u(x,1)=-(1+x)^2,\\
u(0,y)=\cos(\pi y), \quad u(1,y)=4\cos(\pi y).
\end{align*}

**Step 2:**
Compare your solution to the analytic result, which is given by
\begin{equation*}
u(x,y)=(1+x)^2\cos(\pi y).
\end{equation*}

**Step 3:**
Solve the problem using an iterative solution technique, for the case where:

\begin{equation}
f(x,y)=-5 \sin(3\pi x)\cos(2\pi y),
\end{equation}
subject to the conditions
\begin{equation}
u_x(0,y)=0, \quad u_x(1,y)=0, \quad u_y(x,0)=0, \quad u_y(x,1)=0.
\end{equation}

*Hint: to enforce the Neumann boundary conditions, you can use the first order central differences approximation*

\begin{equation*}
u_x(x,y) = \frac{u(x+h,y)-u(x-h,y)}{2h}, \quad u_y(x,y) = \frac{u(x,y+k)-u(x,y-k)}{2h}
\end{equation*}

*The expressions to enforce these conditions at the boundary are given in terms of the result at nodes outside the boundaries. We can construct the required ``fictitious'' nodes and remove them from out solution at the end.*

**Part 1:**

```{code}
clear

% This is the RHS of the equation
f=@(x,y)(2-pi^2*(1+x).^2).*cos(pi*y);

% set up the grid
n=30;
x = linspace(0,1,n); y = linspace(0,1,n);

% Useful to store f(x,y) in an array and use F(i,j) instead of f(x(i),y(k))
[X,Y]=meshgrid(x.',y.'); F=f(X,Y);

% determine grid size
h =x(2)-x(1);

% to hold the solutions
U = zeros(n,n);

% enforce boundary conditions
U(1,:) = (1+x).^2;    U(end,:)=-(1+x).^2;
U(:,1) = cos(pi*y);   U(:,end)=4*cos(pi*y);

%relaxation parameter is r=1 for Gauss-Seidel.
% You could try some higher values for relaxation.
r=1;

[n,m] = size(U);
for k = 1:300 %for each sweep
    for i = 2:n-1
        for j = 2:m-1
            Unew = (U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1))/4-h^2/4*F(i,j);
            U(i,j) = U(i,j)+r*(Unew-U(i,j));
        end
    end
end

% compare to analytic results
T = (1+X).^2.*cos(pi*Y);

figure
subplot(1,2,1)
surf(x,y,T);title('Exact')
subplot(1,2,2)
surf(x,y,abs(U-T));title('Gauss-Seidel error')
```

**Part 2:**
```{code}
% This is the RHS of the equation
f=@(x,y)(-5*sin(3*pi*x).*cos(2*pi*y));

% set up the grid
n=30;
x = linspace(0,1,n); y = linspace(0,1,n);

[X,Y]=meshgrid(x,y);
F=f(X,Y);

% determine grid size
h =x(2)-x(1);

% the solution grid is padded with fictitious nodes to remove at the end
U = zeros(n+2,n+2);

% we need to pad F as well, so that the two grids are not mismatched
F=padarray(F,[1,1]);

r=1;%relaxation parameter is 1 for Gauss-Seidel

[n,m] = size(U);
for k = 1:500 %for each sweep
    %enforce boundary conditions on fictitious nodes
    U(1,:)=U(3,:); U(n,:)=U(n-2,:);   
    U(:,1)=U(:,3); U(:,m)=U(:,m-2);
    for i = 2:n-1
        for j = 2:m-1
            Unew = (U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1))/4-h^2/4*F(i,j);
            U(i,j) = U(i,j)+r*(Unew-U(i,j));
        end
    end
end

figure
surf(x,y,U(2:end-1,2:end-1));
```

Note: If $\phi_{sol}$ is a solution of this problem, which satisfies the boundary conditions, then $\phi_{sol}+C$ is also a solution. Hence, it is possible for the surface to appear shifted up or down in your answers.

**Alternatively, using a simultaneous equations technique**

The function `cdiffM` below was written to solve the Poisson problem

\begin{equation*}
\nabla^2\phi=f
\end{equation*}

using a matrix method for any possible combination of boundary conditions.
The function sets up the coefficient matrix for the problem $A\Phi=F$, where $F,Phi$ are the function value and unknown solution at each grid node $(x,y)$, arranged as vectors.
Array $A$  is the (sparse) coefficient matrix that relates each node to its neighbours using the five-point formula. The boundary equations are modified to use any specified boundary conditions.

```{code}
%% Question 1
% set up the grid
n=30;
x = linspace(0,1,n); y = linspace(0,1,n); [X,Y]=meshgrid(x,y);

%determine grid size
h =x(2)-x(1);

% RHS with enforced boundary conditions
F = h^2*(2-pi^2*(1+X).^2).*cos(pi*Y);
F(1,:)   = (1+x).^2;   F(end,:) =-(1+x).^2;
F(:,1)   = cos(pi*y);  F(:,end) =4*cos(pi*y);
F = F(:); % convert to a list

% LHS with enforced boundary conditions
U = cdiffM(n,{'dirichlet','dirichlet','dirichlet','dirichlet'},F);

% compare to analytic results
U_exact = (1+X).^2.*cos(pi*Y);

figure; surf(x,y,U);

%% Question 2
n=30;
x = linspace(0,1,n).'; y = linspace(0,1,n).'; [X,Y]=meshgrid(x,y);

%determine grid size
h =x(2)-x(1);

% RHS with enforced boundary conditions
F = h^2*(-5*sin(3*pi*X).*cos(2*pi*Y));
F = padarray(F,[1,1],0);
F = F(:); % convert to a list

% LHS with enforced boundary conditions
U = cdiffM(n,{'neumann','neumann','neumann','neumann'},F);

figure; surf(x,y,U);
```

```{code}
function U = cdiffM(n,LRUD,F)
% Construct the coefficient matrix for (nxn) implementation of central
% difference formula, with dirichlet or neumann boundary conditions, and
% solve for given forcing
%
% INPUTS
% n    : dimensions of grid
% LRUD : cell array of left, right, up, down boundary types
%        (e.g. {'neumann','neumann','neumann','neumann'}
% F    : RHS of equation
%
% OUTPUTS
% U    : solutions

m = n;

% Introduce fictitious nodes at any neumann boundaries
if strcmpi(LRUD{1},'neumann'), m=m+1; end
if strcmpi(LRUD{2},'neumann'), m=m+1; end
if strcmpi(LRUD{3},'neumann'), n=n+1; end
if strcmpi(LRUD{4},'neumann'), n=n+1; end

nodes = reshape(1:n*m,n,m); % number the nodes in our (n x m) domain

i = 2:n-1;  j = 2:m-1;      % location of interior nodes

% nodes and neighbours in five-point formula
C = nodes(i,j);
L = nodes(i,j-1); R = nodes(i,j+1);
U = nodes(i-1,j); D = nodes(i+1,j);

% order the five-point  indices as a list
idx = [D(:), L(:), C(:), R(:), U(:)];

% construct the array by looping over the rows
M = zeros(n*m); for k=1:numel(C), M(C(k),idx(k,:)) = [1 1 -4 1 1]; end

%% apply boundary conditions

% left boundary
if strcmpi(LRUD{1},'neumann')
    M = neumann(M,nodes(:,1:3),[-1,1]);
else
    M = dirichlet(M,nodes(:,1));
end

% right boundary
if strcmpi(LRUD{2},'neumann')
    M = neumann(M,fliplr(nodes(:,end-2:end)),[1,-1]);
else
    M = dirichlet(M,fliplr(nodes(:,end)));
end

% upper boundary
if strcmpi(LRUD{3},'neumann')
    M = neumann(M,nodes(1:3,:).',[-1,1]);
else
    M = dirichlet(M,nodes(1,:).');
end

% lower boundary
if strcmpi(LRUD{4},'neumann')
    M = neumann(M,fliplr(nodes(end-2:end,:).'),[1,-1]);
else
    M = dirichlet(M,nodes(end,:).');
end

%% solve the system
U = reshape(M\F,n,m);

% cut off any fictitious nodes that were introduced
if strcmpi(LRUD{1},'neumann'),U(:,1)=[];end
if strcmpi(LRUD{2},'neumann'),U(:,end)=[];end
if strcmpi(LRUD{3},'neumann'),U(1,:)=[];end
if strcmpi(LRUD{4},'neumann'),U(end,:)=[];end

end

function M = dirichlet(M,C)
% Apply dirirchlet boundary conditions at nodes numbered in vector C
    for k = 1:numel(C), M(C(k),C(k)) = 1; end
end

function M = neumann(M,idx,coeffs)
% Apply neumann boundary conditions at nodes idx = [L C R] or [U C D].
    for k = 1:size(idx,1), M(idx(k,1),[idx(k,1),idx(k,3)]) = coeffs; end
end
```


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
