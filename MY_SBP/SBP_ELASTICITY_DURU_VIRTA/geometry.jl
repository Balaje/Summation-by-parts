# Module to define the computational domain
using NLsolve
using ForwardDiff
using LinearAlgebra
using StaticArrays

"""
Function to compute the intersection point of the two curves c₁,c₂
P₁₂ = P(c₁,c₂)
"""
function P(c₁, c₂; guess=[0.0,0.0])   
  function f!(F, x)
    F[1] = c₁(x[1])[1] - c₂(x[2])[1]
    F[2] = c₁(x[1])[2] - c₂(x[2])[2]
  end  
  x0 = guess
  nlsolve(f!, x0, autodiff=:forward).zero
end

"""
Parametric Representation of the boundary
Define c₁, c₂, c₃, c₄
"""
c₁(u) = @SVector [0.1*sin(2π*u), u]
c₃(u) = @SVector [1.0 + 0.1*sin(2π*u), u]
c₂(v) = @SVector [v, 0.1*sin(2π*v)]
c₄(v) = @SVector [v, 1.0 + 0.1*sin(2π*v)]

# Get the intersection points
const P₁₂ = SVector{2}(P(c₁,c₂));
const P₃₄ = SVector{2}(P(c₃,c₄));
const P₄₁ = SVector{2}(P(c₄,c₁));
const P₂₃ = SVector{2}(P(c₂,c₃));

"""
The transfinite interpolation formula
"""
𝒮(x) = (1-x[2])*c₁(x[1]) + x[2]*c₃(x[1]) + (1-x[1])*c₂(x[2]) + x[1]*c₄(x[2]) - 
((1-x[1])*(1-x[2])*P₁₂ + x[1]*x[2]*P₃₄ + x[1]*(1-x[2])*P₄₁ + (1-x[1])*x[2]*P₂₃);

"""
Function to return the Jacobian of the transformation
"""
function J(S,r)
  SMatrix{2,2,Float64}(ForwardDiff.jacobian(S,r))
end

"""
Function to return the inverse of the Jacobian
"""
function J⁻¹(S, r)
  inv(J(S,r))
end

"""
Fancy defintion of the Kronecker product
"""
⊗(A,B) = kron(A,B)

"""
Function to return the material tensors in the transformed coordinates
"""
function t(S, r)  
  invJ = J⁻¹(S, r)    
  Pr = 𝒫(r)
  S = invJ ⊗ I(2)
  S*Pr*S'
end
Aₜ(r) = t(𝒮,r)[1:2, 1:2];
Bₜ(r) = t(𝒮,r)[3:4, 3:4];
Cₜ(r) = t(𝒮,r)[1:2, 3:4];

"""
Evaluate the material tensor on the grid and sort accordingly
"""
function ToGrid(A::Function, R::AbstractVecOrMat{SVector{2,T}}) where T<:Number
  AX = A.(R)
  (getindex.(AX, 1, 1), getindex.(AX, 1, 2), getindex.(AX, 2, 1), getindex.(AX, 2, 2))
end