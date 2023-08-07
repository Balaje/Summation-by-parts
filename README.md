# SBP.jl

Contains code to implement the summation by parts finite difference methods for some problems. To use this package, type the following in the Julia prompt:

```julia
julia> ]
pkg> activate /path/to/this/project
julia> using SBP
```

I have added only the fourth-order SBP operators in this code. To get the SBP operators corresponding to the constant coefficients

```julia
sbp = SBP_1_2_CONSTANT_0_1(n+1) # Get the SBP operators
```


## Advection-diffusion equation

The code can be found in the `examples/sbp_sat_advection_eq.jl` folder. Consider the one-dimensional model problem [(Mattsson, K. and Nordström, J., 2004)](https://www.sciencedirect.com/science/article/pii/S0021999104000932?via%3Dihub)

$$
\begin{align*}
  u_t + au_x &= \epsilon u_{xx}, \quad 0\le x \le 1, \quad t \ge 0\\
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
\begin{align*}
  &\alpha = 1, \quad \beta = 0, \quad c = 2, \quad a = 1,\\
  &b = \frac{c-a}{2\epsilon}, \quad w = \frac{\sqrt{c^2 - a^2}}{2\epsilon}
\end{align*}
$$

Solving the problem using the SBP finite difference method and using Simultaneous Approximation Term (SAT) for applying the boundary condition:

Solution at `T=1.0` s | |
--- | --- |
![](./Images/sol.png) | ![](./Images/rate.png)
**Solution at `T=4.0` s** | |
![](./Images/sol4.0.png) | ![](./Images/rate4.0.png) |

The numerical values of the convergence rates at `T=1.0` s and `T=4.0` s are $[4.2067, 4.1841, 4.1289, 4.0833]$ and $[ 4.1777, 4.1718, 4.1282, 4.0863]$, respectively. The spatial axis is discretized using $N = 30,60,100,200,300$ points (similar to the paper) and using the SBP method whose order of accuracy is 4 in the interior. The temporal direction was discretized using the fourth order Runge-Kutta scheme with $\Delta t = 5\times 10^{-5}$. The observed rate of convergence in the spatial direction and is in agreement with the theory. On invalidating the penalty parameter choice by taking $\tau_0 = -\epsilon/2$ instead of $\tau_0 = -\epsilon$:

https://github.com/Balaje/Summation-by-parts/blob/master/examples/sbp_sat_advection_eq.jl#L64

we observe that the rate of convergence is close to $3$ instead of $4$. This can be seen in the figures below

![](./Images/sol_non_opt.png) | ![](./Images/rate_non_opt.png) |
--- | --- |

The numerical values of the convergence rates is $[3.0323, 3.0295, 3.0204, 3.0130]$.
	
## Incomplete parabolic problem

Now I solve the incomplete parabolic problem

$$
\begin{align*}
  u_t + Cu_x &= D u_{xx} + F(x,t), \quad 0 \le x \le 1, \quad t \ge 0\\
  u(x, 0) &= f(x), \quad t \ge 0\\
  L_0 u &= g_0(t), \quad x = 0, \quad t \ge 0\\
  L_1 u &= g_1(t), \quad x = 1, \quad t \ge 0 
\end{align*}
$$

from [(Mattsson, K. and Nordström, J., 2004)](https://www.sciencedirect.com/science/article/pii/S0021999104000932?via%3Dihub) using the fourth-order SBP method with the diagonal norm. We observe a suboptimal convergence rate ($\approx 3$) which was also observed in the paper. The code can be found in `examples/sbp_sat_incomplete_parabolic.jl`.

![](./Images/sol_incomplete_parabolic.png) | ![](./Images/sol_incomplete_parabolic_rate.png) |
--- | --- |


## Arbitrary domain

Arbitrary domains can be handled using [Transfinite Interpolation](https://en.wikipedia.org/wiki/Transfinite_interpolation). Any point in the physical domain can be mapped to the reference domain using the transfinite interpolation. We can then solve the PDE in the reference domain.

![](./Images/interpolation.png)

## Single-Layer Linear Elasticity

We consider the following PDE

$$
\begin{align*}
  \mathbf{u}_{tt} = \nabla \cdot  \sigma + \mathbf{f}, &\quad \mathbf{x} \in \Omega = [0,1]^2, \quad t>0\\
  \mathbf{u}(x,0) = \mathbf{{u}_0(x)}, &\quad \mathbf{x} \in \Omega\\
  \sigma \cdot \mathbf{n} = \bf{g}(t), &\quad \mathbf{x} \in \partial\Omega, \quad t>0
\end{align*}
$$

where

$$
\mathbf{u}(\mathbf{x},t) = 
    \begin{bmatrix}
      u(\mathbf{x},t)\\
      v(\mathbf{x},t)
    \end{bmatrix}, \qquad 
\sigma(\mathbf{u}) = 
    \begin{bmatrix}
      A(\mathbf{x})\frac{\partial \mathbf{u}}{\partial x} + C(\mathbf{x})\frac{\partial \mathbf{u}}{\partial y}&
      C^T(\mathbf{x})\frac{\partial \mathbf{u}}{\partial x} + B(\mathbf{x})\frac{\partial \mathbf{u}}{\partial y}\\
    \end{bmatrix}, \qquad \mathbf{f} = \mathbf{f}(\mathbf{x},t),
$$

are the displacement field and the Cauchy Stress tensor, respectively. The quantity $\mathbf{n}$ denotes the outward unit normal on the boundary. The material properties 

$$
  A(\mathbf{x}), B(\mathbf{x}) \quad \text{and} \quad C(\mathbf{x})
$$

are symmetric matrices which are generally functions of the spatial coordinates. We then solve the PDE in the unit square using the 4th order SBP method. The script `examples/LinearElasticity/1_layer_linear_elasticity.jl` contains the code to solve the PDE in an arbitrary domain. We assume an exact solution

$$
\mathbf{u}(\mathbf{x},t) = 
\begin{bmatrix}
  \sin(\pi x)\sin(\pi y)\sin(\pi t)\\
  \sin(2\pi x)\sin(2\pi y)\sin(\pi t)  
\end{bmatrix}\\\\
$$

and compute the right-hand side $\mathbf{f}$ and the boundary data $\mathbf{g}$. We consider a uniform two-dimensional discretization with $N = [11,21,31,41,51]$ points. To discretize the temporal direction, we use the Crank Nicolson scheme with $\Delta t = 10^{-3}$ and solve till final time $T = 1.25$ s. Following are the approximate and exact solutions with $N = 51$ points. 

![](./Images/le-x-disp.png) | ![](./Images/le-y-disp.png) 
-- | -- |

The $L^2$-error and the convergence rates are as follows

``` julia 
julia> L²Error
5-element Vector{Float64}:
 0.01675066748858688
 0.0010581963168786465
 0.00018482019369399396
 5.387918811243126e-5
 2.111002788292322e-5

julia> rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
4-element Vector{Float64}:
 3.9845393793309083
 4.303545948944458
 4.2847270008177825
 4.199073174065968
```

Convergence Rates |
--- |
![](./Images/le-rate.png) |

The code now works for problems in arbitrary domain:

![](./Images/solu-arbitrary.png) | ![](./Images/solv-arbitrary.png) | ![](./Images/domain-rate.png) | 
--- | --- | --- |

``` julia
julia> rate
6-element Vector{Float64}:
 3.2746487909958084
 3.7040378918078023
 3.9304181708948893
 4.0365715322662075
 4.0870114543739
 4.111380600861835

julia> L²Error
7-element Vector{Float64}:
 0.006591573645793299
 0.0017472380550776494
 0.0006019691925094348
 0.00025042483107953257
 0.00011996557109679588
 6.389168322477752e-5
 3.689923513170043e-5
```

## Two-Layer Linear Elasticity

The code to solve the two-layer elasticity problem is given in `examples/LinearElasticity/2_layer_linear_elasticity.jl`. The problem contains two domains, each of which is transformed to the reference grid. At the interface between the two domains, continuity of displacements and the traction is enforced. The method is discussed in [Duru and Virta, 2014](https://doi.org/10.1016/j.jcp.2014.08.046).

$$
\sigma_1(\mathbf{u}_1) \cdot \mathbf{n} = \sigma_2(\mathbf{u}_2) \cdot \mathbf{n}, \quad \mathbf{u}_1 = \mathbf{u}_2
$$

In all the experiments, we assume the following exact solution for the numerical tests on both domains

$$
\begin{align*}
	u(x,y,t) &= \sin(\pi x)\sin(\pi y)\sin(\pi t)\\
	v(x,y,t) &= \sin(2\pi x)\sin(2\pi y)\sin(\pi t)\\		
\end{align*}
$$

The material properties, i.e., the Young's modulus and the Poisson's ratio, are $E = 1.0$ units and $\nu = 0.33$, respectively. The density of the material $\rho = 1.0$ units. The right-hand side and the initial conditions are computed using the exact solution. The same material properties are considered on both layers. In addition, we apply homogeneous Neumann boundary conditions on all the boundaries other than the interface.

## Examples:

### Example 1:

In this example, we consider a vanilla straight-line interface at $y=1$ which separates the two domains $[0,1] \times [0,1]$ and $[0,1] \times [1,2]$. The boundary of the domain is parametrized by the following curves
- Layer 1: 
  - Left: $c_0(r) = [0, r+1]$
  - Bottom: $c_1(q) = [q, 1]$ (interface)
  - Right: $c_2(r) = [1, r+1]$
  - Top: $c_3(q) = [q, 2]$
- Layer 2:
  - Left: $c_0(r) = [0, r]$
  - Bottom: $c_1(q) = [q, 0]$
  - Right: $c_2(r) = [1, r]$
  - Top: $c_3(q) = [q, 1]$ (interface)

We have the following results:

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg1/domain.png) | ![](./Images/2-layer/Eg1/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg1/horizontal-disp.png) | ![](./Images/2-layer/Eg1/vertical-disp.png) | 
--- | --- | 

```julia
julia> L²Error
5-element Vector{Float64}:
 0.0014048765373793973
 0.0002489052338364618
 7.220295847822627e-5
 2.773681028865246e-5
 1.2746113394244e-5

julia> rate
4-element Vector{Float64}:
 4.2682648457948105
 4.301940698846879
 4.287466939069201
 4.264630264509913
```

We generally observe optimal rate of convergence for this problem.

### Example 2:

For Example 2, we assume the following boundaries for the two layers. 
- Layer 1: 
  - Left: $c_0(r) = [0, r+1]$
  - Bottom: $c_1(q) = [q, 1]$ (interface)
  - Right: $c_2(r) = [1, r+1]$
  - Top: $c_3(q) = [q, 2 + 0.1\sin(2\pi q)]$
- Layer 2:
  - Left: $c_0(r) = [0, r]$
  - Bottom: $c_1(q) = [q, 0.1\sin(2\pi q)]$
  - Right: $c_2(r) = [1, r]$
  - Top: $c_3(q) = [q, 1]$ (interface)

Here we add a curved boundary on the top and bottom, keeping the interface a straight line. This is intened to check if the interface implementation of the traction is correct. Following are the sparsity pattern of the surface Jacobian on the interface.

![](./Images/2-layer/Eg2/sparsity_layer_1.png) | ![](./Images/2-layer/Eg2/sparsity_layer_2.png) |
--- | --- |

The surface Jacobian arises in the traction term due to the transfinite interpolation. Following are the results and the convergence rates for Example 2.

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg2/domain.png) | ![](./Images/2-layer/Eg2/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg2/horizontal-disp.png) | ![](./Images/2-layer/Eg2/vertical-disp.png) | 
--- | --- | 

```julia
julia> L²Error
5-element Vector{Float64}:
 0.0027907285183550058
 0.0005527940423015698
 0.00017091759284248921
 6.854049976017636e-5
 3.252490179608264e-5

julia> rate
4-element Vector{Float64}:
 3.9931240221672377
 4.080212430116334
 4.094927744183956
 4.0884841923321895
```

Again, we observe optimal convergence rates for this problem. This shows that there seems to be no issue having the surface Jacobian on the traction term on the interface.

### Example 3:

We consider the following computational domain. This example problem can be found in [Duru and Virta, 2014](https://doi.org/10.1016/j.jcp.2014.08.046). The boundary of the domain is parametrized by the following curves
- Layer 1: 
  - Left: $c_0(r) = [0, r+1]$
  - Bottom: $c_1(q) = [q, 1 + 0.2\sin(2\pi q)]$ (interface)
  - Right: $c_2(r) = [1, r+1]$
  - Top: $c_3(q) = [q, 2]$
- Layer 2:
  - Left: $c_0(r) = [0, r]$
  - Bottom: $c_1(q) = [q, 0]$
  - Right: $c_2(r) = [1, r]$
  - Top: $c_3(q) = [q, 1 + 0.2\sin(2\pi q)]$ (interface)

We have the following results:

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg3/domain.png) | ![](./Images/2-layer/Eg3/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg3/horizontal-disp.png) | ![](./Images/2-layer/Eg3/vertical-disp.png) | 
--- | --- | 

The following rate of convergence is observed with the current parameters in the code.

```julia
julia> L²Error
5-element Vector{Float64}:
 0.005192096994131349
 0.001075534324039911
 0.0003477263472045686
 0.0001499647187842862
 7.856626145623064e-5

julia> rate
4-element Vector{Float64}:
 3.8827510567289
 3.925017090032561
 3.7689449170540046
 3.545700826435687
```

The convergence rates seem to be optimal but it appears to reduce to 3.5. Not sure why this happens, maybe due to the choice in the penalty term for the interface conditions?

### Example 4:

In this example, the boundary of the domain is parametrized by the following curves
- Layer 1: 
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r+1]$
  - Bottom: $c_1(q) = [q, 1]$ (interface)
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r+1]$
  - Top: $c_3(q) = [q, 2]$
- Layer 2:
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r]$
  - Bottom: $c_1(q) = [q, 0]$
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r]$
  - Top: $c_3(q) = [q, 1]$ (interface)

We add a curved boundary on the left and right-hand sides of the domain but keep the interface a straight line. This does not do anything to the interface condition, i.e., the surface Jacobian is still equal to 1 on the interface. But regardless, we still perform a convergence test.

We have the following results:

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg4/domain.png) | ![](./Images/2-layer/Eg4/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg4/horizontal-disp.png) | ![](./Images/2-layer/Eg4/vertical-disp.png) | 
--- | --- | 

The following rate of convergence is observed with the current parameters in the code.

```julia
julia> L²Error
5-element Vector{Float64}:
 0.007682909032643207
 0.00154046099064925
 0.00047469531532226
 0.00018966438103023314
 8.97056800505284e-5

julia> rate
4-element Vector{Float64}:
 3.9631438251784785
 4.091891539718783
 4.111331278966825
 4.106601643470132
```

This time, we observe optimal convergence rates.

### Example 5:

In this example, the boundary of the domain is parametrized by the following curves
- Layer 1: 
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r+1]$
  - Bottom: $c_1(q) = [q, 1]$ (interface)
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r+1]$
  - Top: $c_3(q) = [q, 2 + 0.1\sin(2\pi q)]$
- Layer 2:
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r]$
  - Bottom: $c_1(q) = [q, 0 + 0.1\sin(2\pi q)]$
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r]$
  - Top: $c_3(q) = [q, 1]$ (interface)

Now we have a straight-line interface, but we add curved boundaries on the rest of the domain. This should change the surface Jacobian for the interface condition. We have the following results:

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg5/domain.png) | ![](./Images/2-layer/Eg5/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg5/horizontal-disp.png) | ![](./Images/2-layer/Eg5/vertical-disp.png) | 
--- | --- | 

The following rate of convergence is observed with the current parameters in the code.

```julia
julia> L²Error
5-element Vector{Float64}:
 0.014592823041234606
 0.005485928773323668
 0.0025281592366808623
 0.0012984937719568185
 0.0007230180349547938

julia> rate
4-element Vector{Float64}:
 2.4128917795706855
 2.692885722704873
 2.9859097598727833
 3.2115021316773915
```

~~We observe that the convergence rates drop to $\approx 3$.~~ The convergence rates eventually reach 4. Here are updated the rates along with the $L^2-$ errors:

```julia
julia> N
8-element Vector{Int64}:
  41                                                                               
  51                                                                                     
  61                                                                                     
  71                                                                                     
  81 
  91                                                                                  
 101                                                                                     
 111                                                                                                                                                                                                       
julia> L²Error                                                                          
8-element Vector{Float64}:
 0.0025281592366808623                                                                                  
 0.0012984937719568185                                                                                  
 0.0007230180349547938                                                                                  
 0.0004292941192332254 
 0.0026860800040655016                                                                                 
 0.00017553818765438645                                                                                   
 0.00011899345104733748                                                                                   
 8.321818450187672e-5                                                                                   
                                                              
julia> rate  
7-element Vector{Float64}:
 3.2115021316773915 
 3.381703476165346 
 3.511452986406 
 3.611695471642677 
 3.6900745207942363
 3.7519871586182476 
```

### Example 6:

In this example, the boundary of the domain is parametrized by the following curves
- Layer 1: 
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r+1]$
  - Bottom: $c_1(q) = [q, 1 + 0.1\sin(2\pi q)]$ (interface)
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r+1]$
  - Top: $c_3(q) = [q, 2 + 0.1\sin(2\pi q)]$
- Layer 2:
  - Left: $c_0(r) = [0 + 0.1\sin(2\pi r), r]$
  - Bottom: $c_1(q) = [q, 0 + 0.1\sin(2\pi q)]$
  - Right: $c_2(r) = [1 + 0.1\sin(2\pi r), r]$
  - Top: $c_3(q) = [q, 1 + 0.1\sin(2\pi q)]$ (interface)

Lastly, we add curved boundaries on both domains. We have the following results:

Computational domain | Convergence Rates |
--- | --- |
![](./Images/2-layer/Eg6/domain.png) | ![](./Images/2-layer/Eg6/rate.png) |

The solution obtained from the code is 

![](./Images/2-layer/Eg6/horizontal-disp.png) | ![](./Images/2-layer/Eg6/vertical-disp.png) | 
--- | --- | 

The following rate of convergence is observed with the current parameters in the code.

```julia
julia> L²Error
5-element Vector{Float64}:
 0.014592823041234606
 0.005485928773323668
 0.0025281592366808623
 0.0012984937719568185
 0.0007230180349547938

julia> rate
4-element Vector{Float64}:
 2.4128917795706855
 2.692885722704873
 2.9859097598727833
 3.2115021316773915
```

The convergence rates are similar to Example 5. It drops to $\approx 3$. This behaviour needs to be investigated further. 

# References

- Mattsson, K. and Nordström, J., 2004. Summation by parts operators for finite difference approximations of second derivatives. Journal of Computational Physics, 199(2), pp.503-540.
- Duru, K., Virta, K., 2014. Stable and high order accurate difference methods for the elastic wave equation in discontinuous media. Journal of Computational Physics 279, 37–62. https://doi.org/10.1016/j.jcp.2014.08.046
