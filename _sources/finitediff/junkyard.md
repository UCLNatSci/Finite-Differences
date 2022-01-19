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
