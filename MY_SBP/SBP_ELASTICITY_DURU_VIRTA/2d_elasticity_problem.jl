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
  Uₜₜ(x,t) - div(𝛔, x);
end

"""
Non-zero traction at the left boundary (x=0)
  σ ⋅ ([-1, 0])
"""
function g₀(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(-1) + τ[2]*(0); τ[3]*(-1) + τ[4]*(0)]
end

"""
Non-zero traction at the bottom boundary (y=0)
  σ ⋅ ([0,-1])
"""
function g₁(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(0) + τ[2]*(-1); τ[3]*(0) + τ[4]*(-1)]
end

"""
Non-zero traction at the rigth boundary (x=1)
  σ ⋅ ([1,0])
"""
function g₂(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(1) + τ[2]*(0); τ[3]*(1) + τ[4]*(0)]
end

"""
Non-zero traction at the top boundary (y=1)
  σ ⋅ ([0,1])
"""
function g₃(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(0) + τ[2]*(1); τ[3]*(0) + τ[4]*(1)]
end