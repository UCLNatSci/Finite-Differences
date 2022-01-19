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

# Multi-stage methods

```{admonition} Please note!
:class: danger
This chapter material is optional, and will not be required to complete the project
```

The methods seen in the previous chapter used a single calculation step to define the relationship between each neighbouring pair of solution point.

In this chapter we will look at a general class of multi-stage methods that use weighted averages of slope estimates obtained at different points within a single step (e.g. at a half-step) to reduce the truncation error. These algorithms can be highly accurate, so larger step sizes can be taken, but they may also exhibit less stability than lower order algorithms. The stability can be assessed by comparison of more than one technique or comparison of the results for different step sizes. Some algorithms use these comparisons to improve efficiency by using an adaptive step size.

Knowledge of multi-stage methods is not crucial for this module, as the techniques that we will develop for solving higher order ODEs and partial differential equations are based on single stage methods. However, the material in this chapter gives some insight about methods employed by `scipy.integrate.solve_ivp` to improve accuracy and manage stability. Seeing how the techniques are implemented may also provide good Python experience.


## Modified Euler method
If we approximate the derivative using the central differences formula, then we obtain

\begin{equation}y(x_k+h)\simeq y(x_k)+f\biggr(x_k+\frac{h}{2},y\left(x_k+\frac{h}{2}\right)\biggr)\end{equation}

This result requires the solutions for both the previous value $y_k$ and $y_{k+1/2}$. However, we can use Euler's explicit formula with a half-step to estimate this value. Due to the nesting of the approximation this does not affect the accuracy of the method, which remains quadratic order. The resulting explicit algorithm, which is known as the modified Euler scheme can be calculated via a two-stage process:

\begin{equation}s_1=f(x_k,y_k), \quad s_2=f\left(x_k+\frac{h}{2},y_k+\frac{h}{2}s_1\right), \qquad y_{k+1}=y_k+h s_2\end{equation}

Here, $s_1$ is the first estimate for the slope, using the left-hand derivative rule $s_2$ is the modified estimate for the slope at the midpoint.

An example implementation is shown below:

```{code-cell} ipython3

def my_predcorr(f,x,y0):

  n,h=len(x),x[1]-x[0]    #get number of values and step size
  y=np.zeros(len(x))      #form output y array
  y[0]=y0

  for k in range(n-1):
      s1 = f(x[k],y[k])                 #first stage
      s2 = f(x[k]+h/2, y[k]+h/2*s1)     #second stage
      y[k+1]=y[k]+h*s2

  return y
```

## Improved Euler method (Heun's method)
Recall that the trapezoidal rule {eq}`trapz` also gave quadratic order accuracy, but using it involved solving an implicit relationship involving $y_{k+1}$.

\begin{equation}y_{k+1}=y_k+\frac{h}{2}\biggr[f(x_k,y_k)+f(x_{k+1},y_{k+1})\biggr]\end{equation}

Again, we can replace this value on the right-hand side of the equation by the explicit Euler method, giving the following quadratic order multi-stage method:

\begin{equation}s_1=f(x_k,y_k), \quad s_2=f(x_{k+1},y_k+h s_1), \qquad y_{k+1}=y_k+\frac{h}{2}(s_1+s_2)\end{equation}

Here, $s_1$ is the first estimate for the slope, using the left-hand derivative rule $s_2$ is the modified estimate for the slope at the right-hand point.

An example implementation is shown below

```{code-cell} ipython3

def my_heun(f,x,y0):

  n,h=len(x),x[1]-x[0]    #get number of values and step size
  y=np.zeros(len(x))      #form output y array
  y[0]=y0

  for k in range(n-1):
    s1 = f(x[k], y[k])            #first stage
    s2 = f(x[k+1], y[k]+h*s1)     #second stage
    y[k+1] = y[k]+h*(s1+s2)/2

  return y
```

## Runge Kutta methods
By using more locations (stages) we can improve the local accuracy of the method at the expense of performing additional calculations. The computation time for each step increases linearly with the number of stages, but this is usually offset by the ability to use fewer steps to obtain the required accuracy. Runge-kutta methods are general one-step, multi-stage methods of the form

\begin{equation}y_{k+1}=y_k+h\sum_{i=1}^n b_i s_i, \qquad s_i = f\left(x_k+c_i h,y_k+h\sum_{j=1}^{n}a_{i,j}s_j\right)\end{equation}

where $k$ is the number of stages, $A =(a_{i,j})$ is a ($n\times n$) constant matrix, and $B=(b_i)$, $C=(c_i)$ are constant vectors. The method takes a weighted average of several adjusted slopes and is explicit only if $A$ is lower diagonal.

The choice of coefficients follows from expanding expression the expression for $y_{k+1}$ fully in terms of $f(x_k,y_k)$ and its derivatives. The result can then be compared to the Taylor expansion for $y_{k+1}$ which satisfies the initial value problem to determine the coefficients. Results for the coefficients are often represented as a "Butcher Tableau", which you can look up:

\begin{equation}\begin{array}{c|ccccc}c_1=0 & \\c_2 & a_{2,1}\\c_3& a_{3,1} & a_{3,2}\\\vdots&\vdots & & \ddots\\c_n&a_{n,1}& a_{n,2} &\dots &a_{n,n-1}\\\hline &b_1 &b_2& &\dots &b_n\end{array} \quad \begin{array}{cl}\phantom{c_0}&s_1 = (x_k,y_k)\\\phantom{c_1} &s_2 = f(x_k+c_2 h,y_k+ha_{2,1}s_1)\\\phantom{c_2}&s_3 = f(x_k+c_3 ,y_k+h(a_{3,1}s_1+a_{3,2}s_2)\\\vdots{}\\\phantom{c_n} &s_n = f(x_k+c_n h,y_k+h(a_{n,1}s_1+a_{n,2}s_2+\dots))\\ &y_{k+1}=y_k+h(b_1 s_1 + b_2 s_2 +\dots) +b_k s_n\end{array}\end{equation}

For example, the modified Euler and Heun's improved Euler methods are both examples of two stage RK methods, which demonstrate quadratic order accuracy.

**Modified Euler**:

$$\displaystyle \begin{array}{c|cc}0\\ 1/2 & 1/2\\\hline &0&1\end{array}\quad\begin{array}{cl}&s_1=f(x_k,y_k)\\\phantom{1/2}&s_2=f\left(x_k+\frac{h}{2},y_k+\frac{h}{2}s_1\right)\\\phantom{1/2}&y_{k+1}=y_k+h s_2\end{array}$$

**Heun**:

$$\displaystyle\begin{array}{c|cc}0\\ 1 & 1\\\hline &1/2&1/2\end{array}\quad\begin{array}{cl}&s_1=f(x_k,y_k)\\\phantom{1/2}&s_2=f(x_{k+1},y_k+h s_1)\\\phantom{1/2}&y_{k+1}=y_k+\frac{h}{2}(s_1+s_2)\end{array}$$

It is not possible to construct an explicit $n$-stage RK algorithm with $\mathcal{O}(h^n)$ accuracy for $n> 4$ (more stages are required), and so explicit RK4 methods are extremely popular. The classic four stage algorithm that has $\mathcal{O}(h^4)$ accuracy is given by

$$\displaystyle\begin{array}{c|cccc}0\\1/2&1/2\\1/2&0&1/2\\1&0&0&1\\\hline &1/6&1/3&1/3&1/6\end{array}\quad\begin{array}{cl}&s_1=f(x_k,y_k)\\\phantom{1/2}&s_2=f\left(x_k+\frac{h}{2},y_k+\frac{h}{2}s_1\right)\\\phantom{1/2}&s_3=f\left(x_k+\frac{h}{2},y_k+\frac{h}{2}s_2\right)\\\phantom{1}&s_4=f(x_k+h,y_k+h s_3)\\ & y_{k+1}=y_k+\frac{h}{6}(s_1+2s_2+2s_3+s_4) \end{array}$$

By comparing the solutions of a given algorithm for two different step sizes we can estimate how small the step needs to be to keep the error within specified bounds over the integration domain.
By comparing results from different algorithms after one (or a few) steps it is also possible to implement a variable step method that uses larger steps in regions where errors are estimated to be small. This is possible with the Python `solve_ivp` within the `scipy.integrate` module, using comparison of a $\mathcal{O}(h^4)$ and $\mathcal{O}(h^5)$ method.


````{exercise}
Set up a scheme to apply Heun's algorithm to the following system for $t\in[0,1]$, using time steps of 0.1, 0.01 and 0.001.

\begin{equation*}\begin{aligned}\frac{\mathrm{d}u}{\mathrm{d}t}&= 998 u +1998 v, \qquad &&u(0)=2,\\\frac{\mathrm{d}v}{\mathrm{d}t}&= -999 u -1999 v, &&v(0)=-1.\end{aligned}\end{equation*}

Compare your results with the forward Euler method.

**Hint**
You could start by constructing a function handle for the RHS of the form:

```python
f = lambda t,X: np.array([998*X[0]+1998*X[1],-999*X[0]-1999*X[1]])
```

The function takes the input vector $X=(u,v)$ and calculates the resulting slopes $\mathrm{d}X=(\mathrm{d}u/\mathrm{d}t,\mathrm{d}v/\mathrm{d}t)$, which are used in the finite difference formulas to step forwards.
In this manner, you can build up a matrix of values

\begin{equation*}\left[\begin{array}{ccc}u_0 & u_1&\dots\\v_0 & v_1&\dots\end{array}\right]\end{equation*}

As the result is a single step method, the columns in this resulting matrix are computed one-by-one. The multiple stages are estimates of the slopes $\mathrm{d}X$ at different locations.

````
