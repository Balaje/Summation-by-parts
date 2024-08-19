# Summation By Parts

This repository contains the Julia code to solve the elastic wave equation with Perfectly Matched Layers (PML) in layered media. The code can be used to generate the results presented in the accompanying manuscript found on Arxiv: [https://arxiv.org/abs/2210.00229](https://arxiv.org/abs/2210.00229). This code is in v1.0 and will be constantly updated to make it more organised and performant. To use the package, download the latest release and run 

``` julia
julia> ]
pkg> activate /path/to/this/project
julia> using SummationByPartsPML
```

The only method currently available in this repository is the fourth-order Summation By Parts Finite Difference technique (Mattsson and Nordstrom). To check, run this test snippet:

``` julia
sbp_1d = SBP4_1D(20)
φ(x) = x^2
Dₓ = sbp_1d.D1; Dₓₓ = sbp_1d.D2[1];
using Test
x = LinRange(0,1,20)
@test Dₓ*φ.(x) ≈ 2x.^1
@test Dₓₓ*φ.(x) ≈ 2x.^0
```

An example to solve a 1D Hyperbolic problem is given in the `test/` folder: To run the package testing routine, in the julia terminal, type

```julia
julia> ]
(SummationByPartsPML) pkg>  test SummationByPartsPML
```

Four examples are considered in the paper. 


## 2-layer examples

`t = 1.0` | `t = 2.0` | `t = 40.0` |
-- | -- | -- |
![Two-layer Gaussian 1.0](./Images/2-layer-1.0-uniform.png) | ![Two-layer Gaussian 1.0](./Images/2-layer-2.0-uniform.png) | ![Two-layer Gaussian 1.0](./Images/2-layer-40.0-uniform.png) 


`t = 1.0` | `t = 2.0` | `t = 40.0` |
-- | -- | -- |
![Two-layer Gaussian 1.0](./Images/2-layer-1.0.png) | ![Two-layer Gaussian 1.0](./Images/2-layer-2.0.png) | ![Two-layer Gaussian 1.0](./Images/2-layer-40.0.png) 

## Convergence of PML

![Two-layer PML solution](./Images/pml-solution.png) | ![Two-layer Full solution](./Images/abc-solution.png) |
-- | -- |
![Two-layer PML solution](./Images/full-solution.png) | ![Two-layer Full solution](./Images/PML-vs-ABC.png) |

## 4-layer example

![Four-layer PML solution](./Images/4-layer-3.0-uniform.png) | ![Four-layer PML solution](./Images/4-layer-5.0-uniform.png) | ![Four-layer PML solution](./Images/4-layer-10.0-uniform.png)
-- | -- | -- |

## Marmousi Model

![p-wave speed Marmousi](./Images/Marmousi-p-wave.png) |
-- |
![Two-layer Marmousi 0.5](./Images/marmousi-t-0.5.png) | 
![Two-layer Marmousi 0.5](./Images/marmousi-t-1.0.png) | 

# References