using StaticArrays
using ForwardDiff
using NLsolve
using LinearAlgebra
using Plots

include("geometry.jl");
include("material_props.jl")
include("SBP_2d.jl")
include("time-stepping.jl")

"""
Exact solution
"""
U(x,t) = (@SVector [sin(π*x[1])*sin(π*x[2])*sin(π*t), sin(2π*x[1])*sin(2π*x[2])*sin(π*t)]);

"""
First time derivative Uₜ(x,t)
"""
Uₜ(x,t) = ForwardDiff.derivative(τ->U(x,τ), t)

"""
Second time derivative Uₜₜ(x,t)
"""
Uₜₜ(x,t) = ForwardDiff.derivative(τ->Uₜ(x,τ), t)


# Compute the initial data from the exact solution
"""
Initial Displacement
"""
U₀(x) = U(x,0);

"""
Initial Velocity 
"""
Uₜ₀(x) = Uₜ(x,0);

"""
Right hand side function 
  f(x,t) = Uₜₜ(x,t) - ∇⋅(σ(U))(x,t)
"""
function F(x,t) 
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  ρ(x)*Uₜₜ(x,t) - div(𝛔, x);
end

"""
Non-zero traction at the boundary
  c: The curve function
  u: Parameter in the curve function
    [x,y] = c(u)
  o: Orientation of the normal
  = σ(c₀(u),t) ⋅ n(c₀)
"""
function g(t,c,u,o)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  x = c(u)
  τ = 𝛔(x)  
  _n = 𝐧(c,u; o=o)
  @SVector [τ[1]*_n[1] + τ[2]*_n[2]; τ[3]*_n[1] + τ[4]*_n[2]]
end