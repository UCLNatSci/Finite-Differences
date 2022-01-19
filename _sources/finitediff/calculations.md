# Derivative calculations

```{admonition} Big idea : Discretisation
:class: tip
We can use the finite difference formulae to estimate the derivative of a function described by a discrete set of points $(x_k,y_k)$.

Obtaining the derivative at the end-points may require us to know the function value at phantom grid-points outside the grid domain.
```

## Using forward differences
Suppose that instead of being given an analytic function $y(x)$, we have a set of (e.g. experimental) points

$$\begin{equation}(x_1,y_1),(x_2,y_2),\dots,(x_n,y_n).\end{equation}$$

We can approximate the derivative of the hypothetical function described by these points, using the forward difference estimate

$$\begin{equation}y_k^{\prime}\simeq\frac{y_{k+1}-y_k}{x_{k+1}-x_k}.\end{equation}$$

For simplicity we will assume that the grid spacing $h_k=(x_{k+1}-x_k)$ is the same value $h$ for all neighbours (though this isn't necessary), which gives:

$$\begin{align}y^{\prime}&=\biggr[\frac{y(x_2)-y(x_1)}{h},\frac{y(x_3)-y(x_2)}{h},\dots,\frac{y(x_n)-y(x_{n-1})}{h}\biggr]\\&=\frac{[y_2,y_3,\dots,y_n]-[y_1,y_2,\dots,y_{n-1}]}{h}\end{align}$$

The idea, which is not very sophisticated, is illustrated in the schematic diagram below. Essentially, we are joining up the points with straight lines and using the slope of those line segments to estimate the gradient at the left-hand point of each interval.

Notice that with $n$ experimental points we can only find the derivative at the first $(n-1)$ points, because the forward projection of the function at the final point is not known. If we wish to obtain the derivative at $x_n$, we require the phantom exterior value $(x_{n+1},y_{n+1})$.

```{image} images/griddly3.png
:alt: computing the derivative from secant lines
:height: 300px
:align: center
```

The code below provides an implementation of the formula for a given set of points $(x,y)$.

```{code-cell} ipython3
def forward_diff(x,y):

    # Assuming equal grid spacing h
    h = x[1]-x[0]
    fd = (y[1:]-y[0:-1])/h

    # OR
    # No assumption of equal grid spacing
    fd = (y[1:]-y[0:-1])/(x[1:]-x[0:-1])

    return fd
```

**Example**:

We will demonstrate for the function

$$\begin{equation}y=\sin(x)+x, \quad y(-\pi)=-\pi\end{equation}$$

To obtain a result for the derivative at a set of points including the endpoint, we will extend the grid by adding a phantom exterior point at the right hand side.

```{code-cell} ipython3
---
render:
    image:
        align: center
tags: [hide-input]
---

import numpy as np
from math import pi, sin
import matplotlib.pyplot as plt

x = np.linspace(-pi,pi,100)
h = x[1]-x[0]
xx = np.insert(x,len(x), pi+h)
yy = list(map(lambda a: sin(a), xx)) + xx

fd = (yy[1:]-yy[:-1])/h

plt.plot(x, fd)
plt.xlabel('x')
plt.title('Estimated derivative of sin(x)+x')
plt.ylim(0,2)
plt.show()
```
```{note}
The relevant packages were imported here (numpy, math, matplotlib), therefore no longer need importing in the next sections. But make sure you do import them if you are running some of the sections independently.
```
```{admonition} Discretisation tip
:class: tip
In the above example we allowed python to compute the grid points, by using linspace to ensure that the right-hand end point was included. However, for some applications it may not be convenient to use linspace.

If you want to compute the step size $h$ that will discretise a domain with the right-hand end point included, it is useful to recognise that the $x$ values in the grid are in arithmetic progression. If the first term is $a$, then the last term is $a+(n-1)h$, where $n$ is the number of points. This gives the following result for an interval $[a,b]$

$$\begin{equation}a+(n-1)h=b \quad \Rightarrow\quad h=\frac{b-a}{n-1}\end{equation}$$
```

## Using central differences
Assuming again an equal grid spacing $h$, we can use either of the following two results

$$\displaystyle y_k^{\prime}\simeq\frac{y_{k+1/2}-y_{k-1/2}}{h}$$

This result requires the function values at the midpoint of each pair of grid values. It is convenient to relabel the index, so that the subscript on the right had side of the formula matches the numbering of the gridpoints, giving $y_{k+1/2}^{\prime}\simeq\frac{y_{k+1}-y_{k}}{h}$. This result is identical to the forward derivative estimate, but it should be interpreted as the derivative of the function at the mid-point of each interval, rather than at the left-hand point.

$$\displaystyle y_k^{\prime}=\frac{y_{k+1}-y_{k-1}}{2h}$$

This result gives an estimate of the derivative at the original grid locations for the derivative at the interior points $[x_2,x_3,\dots,x_{n-1}]$. Computing the derivative at the two end-points requires the function value $y$ at phantom exterior points $x_{0}$ and $x_{n+1}$, as shown below, which is given to match the example in the previous section.

```{code-cell} ipython3
---
render:
    image:
        align: center
tags: [hide-input]
---

x = np.linspace(-pi,pi,100)
h = x[1]-x[0]
xx = np.insert(x,[0,len(x)],[-pi-h, pi+h])
yy = list(map(lambda a: sin(a), xx)) + xx

fd = (yy[2:]-yy[:-2])/(2*h)

plt.plot(x, fd)
plt.xlabel('x')
plt.title('Estimated derivative of sin(x)+x')
plt.ylim(0,2)
plt.show()
```

## Demonstrating the truncation error
To illustrate the accuracy of the formulae, we can compute the maximum error in the approximation for a range of step sizes and plot the result. Since the resulting errors $E$ are proportional to $h^n$, a plot of $\ln(E)$ against $\ln(h)$ should show a linear relationship. The gradient of the line gives the estimated order of the truncation error, $n$. We find (as expected) that the formula based on forward differences has a first order error relationship and the formula based on central differences has a quadratic order error relationship.

```{code-cell} ipython3
---
render:
    image:
        align: center
tags: [hide-input]
---
from math import cos
from matplotlib.pyplot import subplots, show

def fdiff(x, fun, type):

    h=x[1]-x[0]
    if type =='forward':
        xx = np.insert(x,len(x), x[-1]+h)
        yy = fun(xx)
        fd = (yy[1:]-yy[:-1])/h
    elif type == 'central':
        xx = np.insert(x,[0,len(x)],[x[0]-h, x[-1]+h])
        yy = fun(xx)
        fd = (yy[2:]-yy[:-2])/(2*h)

    return fd


num_h=10
hvals = 2*pi/(np.logspace(1,5,num_h)-1)

fig, ax = subplots()

for _,method in enumerate(['forward', 'central']):
    Evals = np.zeros(hvals.shape)
    fun = lambda a: list(map(lambda x: sin(x), a)) + a
    for i in range(num_h):
        # Set up grid
        h=hvals[i]
        x= np.arange(-pi,pi,h)

        fd = fdiff(x,fun,method)
        dydx = np.asarray(list(map(lambda b: cos(b), x))) + 1

        err = abs(dydx-fd)
        Evals[i] = max(abs(err))

    ax.loglog(hvals,Evals,'-o', label=method)

    # Fitting degree 1 polynomial
    p = np.polyfit(np.log(hvals), np.log(Evals), 1)
    # Truncation error
    print(method, 'estimated n= {:.5f}\n'.format(p[0]))

ax.set_xlabel('h')
ax.set_ylabel('error')
ax.legend()
show()
```

## <span style="color: red;">Coding challenge</span>
See if you can compute the second derivative of the function

$$\begin{equation}\sin(x)+x, \quad x\in[-\pi,\pi],\end{equation}$$
using either the forward or central difference formula. You will need to extend the function estimate at *two* exterior points to obtain a result for the derivative at each point in the interval.


**Appendix: The difference between truncation errors and roundoff errors**

Truncation errors occur due to working with finite approximations to infinitesimal quantities, whilst the roundoff errors are a computational artefact due to working with finite precision arithmetic. Numeric computation packages ordinarily use floating point representation of numbers which are very efficient and represent numbers faithfully to around 16 digits.

It is possible to increase the working precision of a calculation to a specified degree by using variable-precision arithmetic. However, this results in a significant slowdown in computational speed. For most practical purposes, floating point precision is sufficient.
