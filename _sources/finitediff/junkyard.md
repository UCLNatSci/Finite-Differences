# Junkyard

```{code-cell} ipython3
def xlist(x0,xend=None,**kwargs):

    xendTrue = (xend is not None)
    nTrue,hTrue = [(k in kwargs.keys()) for k in ['n','h']]

    if xendTrue:
        if nTrue:
            x=np.linspace(x0,xend,kwargs["n"])        #start,stop,length
        elif hTrue:
            x=np.arange(x0,xend,kwargs["h"])          #start,stop,step
        else:
            x=np.linspace(x0,xend,1000)               #start,stop (default length)
    elif nTrue and hTrue:
            h,n=kwargs["h"],kwargs["n"]
            x=[x0+h*n for n in range(n)]              #start,step,length
    else:
            print("Did you forget to define xend?")   #missing xend
            x=None
    return x
```

For example,

* `xlist(1,h=0.01,n=20)`
* `xlist(1,2,h=0.01)`
* `xlist(1,2,n=20)`
* `xlist(1,2)`

```{code-cell} ipython3

def my_eulr2(f,xrange,y0,**kwargs):

  x=xlist(*xrange,**kwargs)    # form output x array
  n=len(x)
  h=x[1]-x[0]

  y=np.zeros(len(x))           # form output y array
  y[0]=y0

  for k in range(n-1):
      x1,y1=x[k],y[k]          # introduced for convenience
      y2=y1+h*f(x1,y1)         # Euler forward difference
      y[k+1]=y2
  return x,y
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
