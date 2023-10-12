using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Plots

include("geometry.jl");
include("time-stepping.jl")

"""
Gradient (Jacobian) of the displacement field
"""
@inline function ∇(u,x)
 vec(ForwardDiff.jacobian(u, x))
end

"""
Divergence of a tensor field
  v is a 2×2 matrix here, where each entries are scalar functions
"""
function div(v,x)
  v₁₁(x) = v(x)[1]; 
  v₁₂(x) = v(x)[3]; 
  v₂₁(x) = v(x)[2];
  v₂₂(x) = v(x)[4];   
  ∂xv₁₁ = ForwardDiff.gradient(v₁₁,x)[1];
  ∂xv₁₂ = ForwardDiff.gradient(v₁₂,x)[1];
  ∂yv₂₁ = ForwardDiff.gradient(v₂₁,x)[2];
  ∂yv₂₂ = ForwardDiff.gradient(v₂₂,x)[2];
  @SVector [∂xv₁₁ + ∂yv₂₁; ∂xv₁₂ + ∂yv₂₂]
end

"""
Exact solution
"""
U(x,t) = (@SVector [sin(2π*x[1])*sin(2π*x[2])*cos(2π*t), -sin(2π*x[1])*sin(2π*x[2])*cos(2π*t)]);

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
function F(x,t,σ,ρ) 
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
function g(t,c,u,σ,o)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  x = c(u)
  τ = 𝛔(x)  
  _n = 𝐧(c,u; o=o)
  @SVector [τ[1]*_n[1] + τ[2]*_n[2]; τ[3]*_n[1] + τ[4]*_n[2]]
end
