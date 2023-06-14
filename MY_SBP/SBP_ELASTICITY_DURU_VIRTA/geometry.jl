# Module to define the computational domain

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
Define c₀, c₁, c₂, c₃
"""
c₀(u) = @SVector [0.0, u] # Left boundary 
c₁(v) = @SVector [v, 0.0] # Bottom boundary
c₂(u) = @SVector [1.0, u] # Right boundary
c₃(v) = @SVector [v, 1.0] # Top boundary

# Get the intersection points
P₀₁ = SVector{2}(P(c₀,c₁));
P₁₂ = SVector{2}(P(c₁,c₂));
P₂₃ = SVector{2}(P(c₂,c₃));
P₃₀ = SVector{2}(P(c₃,c₀));

"""
The transfinite interpolation formula
"""
𝒮(x) = (1-x[1])*c₀(x[2]) + x[1]*c₂(x[2]) + (1-x[2])*c₁(x[1]) + x[2]*c₃(x[1]) - 
((1-x[2])*(1-x[1])*P₀₁ + x[2]*x[1]*P₂₃ + x[2]*(1-x[1])*P₃₀ + (1-x[2])*x[1]*P₁₂);

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
Function to return the material tensor in the reference coordinates (0,1)×(0,1). Returns 
  𝒫' = S*𝒫*S'
where S is the transformation matrix
"""
function t(S, r)  
  invJ = J⁻¹(S, r)      
  S = invJ ⊗ I(2)
  S*𝒫(r)*S'
end

"""
The material coefficient matrices in the reference coordinates (0,1)×(0,1).
  A(x) -> Aₜ(r)
  B(x) -> Bₜ(r)
  C(x) -> Cₜ(r) 
"""
Aₜ(r) = t(𝒮,r)[1:2, 1:2];
Bₜ(r) = t(𝒮,r)[3:4, 3:4];
Cₜ(r) = t(𝒮,r)[1:2, 3:4];

"""
Flatten the 2d function as a single vector for the time iterations
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Unit normals on the boundary
"""
function 𝐧(c,u; o=1.0) 
  res = ForwardDiff.derivative(t->c(t), u)
  r = @SMatrix [0 -1; 1 0]
  o*r*res/norm(res)  
end