# Solutions

**Exercise 1.1.**

The series expansion is found to be:

\begin{equation}p_2(x;\pi/3)=\frac{\sqrt{3}}{2}+\frac{1}{2}\biggr(x-\frac{\pi}{3}\biggr)-\frac{\sqrt{3}}{4}\biggr(x-\frac{\pi}{3}\biggr)^2\end{equation}

And from Lagrange's formula, the error in the expansion at $x=\frac{\pi}{2}$ is given by:

\begin{equation}R(\xi)= \biggr|\frac{\cos(\xi)(\frac{\pi}{2}-\frac{\pi}{3})^3}{6}\biggr| \end{equation}

Since $|\cos(\xi)|$ is bounded above by $1$ on the given domain, Lagrange's remainder theorem gives an upper bound of $\frac{\pi^3}{6^4}=0.0239246$

The exact error is

\begin{equation}\biggr|\sin\left(\frac{\pi}{2}\right)-p_2\left(\frac{\pi}{2};\frac{\pi}{3}\right)\biggr| = 0.0091119\end{equation}

---
