# SBP.jl

Contains code to implement the summation by parts finite difference methods for some problems.

## Advection-diffusion equation

Consider the one-dimensional model problem [(Mattsson, K. and Nordström, J., 2004)](https://www.sciencedirect.com/science/article/pii/S0021999104000932?via%3Dihub)

$$
\begin{align*}
  u_t + au_x &= \epsilon u_{xx}, \quad 0\le x \le 1, \; t \ge 0\\
  u(x,0) &= f(x),\\
  \alpha u(0,t) + u_x(0,t) &= g_0(t),\\
  \beta u(1,t) + u_x(1,t) &= g_1(t).\\
\end{align*}
$$

Assuming an exact solution to the problem

$$
u(x,t) = \sin ({w(x-ct)})e^{-bx}
$$

with the parameters

$$
\alpha = 1, \quad \beta = 0, \quad c = 2, \quad a = 1\\
b = \frac{c-a}{2\epsilon}, \quad w = \frac{\sqrt{c^2 - a^2}}{2\epsilon}
$$

Solving the problem using the SBP finite difference method and using Simultaneous Approximation Term (SAT) for applying the boundary condition:

Solution at `T=1.0` s | | Rate |
--- | --- | --- |
![](./MY_SBP/Images/sol.png) | ![](./MY_SBP/Images/rate.png) | 4.206668554230709, 4.184137602101776, 4.12890189205842, 4.083294430882183
Solution at `T=4.0` s | | |
![](./MY_SBP/Images/sol4.0.png) | ![](./MY_SBP/Images/rate4.0.png) | 4.177707021531643, 4.171771585661089, 4.128172006795482, 4.08626801731685

The spatial axis is discretized using $N = 30,60,100,200,300$ points (similar to the paper) and using the SBP method whose order of accuracy is 4 in the interior. The temporal direction was discretized using the fourth order Runge-Kutta scheme with $\Delta t = 5\times 10^{-5}$. The observed rate of convergence in the spatial direction and is in agreement with the theory.

## References

- Mattsson, K. and Nordström, J., 2004. Summation by parts operators for finite difference approximations of second derivatives. Journal of Computational Physics, 199(2), pp.503-540.