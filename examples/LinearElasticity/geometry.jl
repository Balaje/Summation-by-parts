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
  x1 = nlsolve(f!, x0, autodiff=:forward).zero
  c₁(x1[1])
end

"""
Parametric Representation of the boundary
Define c₀ (Left), c₁ (Bottom), c₂ (Right), c₃ (Top)
"""
c₀(r) = @SVector [0.0 + 0.1*sin(π*r), r] # Left boundary 
c₁(q) = @SVector [q, 0.0 + 0.1*sin(2π*q)] # Bottom boundary
c₂(r) = @SVector [1.0 + 0.1*sin(π*r), r] # Right boundary
c₃(q) = @SVector [q, 1.0 + 0.1*sin(2π*q)] # Top boundary

# Get the intersection points
# Solved using a non-linear (function P(c₁,c₂)) solver to obtain (x,y) s.t
#   (x,y) = cᵢ(r) = cⱼ(q), 
P₀₁ = SVector{2}(P(c₀,c₁));
P₁₂ = SVector{2}(P(c₁,c₂));
P₂₃ = SVector{2}(P(c₂,c₃));
P₃₀ = SVector{2}(P(c₃,c₀));

"""
The transfinite interpolation formula which takes points (q,r) in the reference grid and returns the points in the physical grid.
"""
𝒮(qr) = (1-qr[1])*c₀(qr[2]) + qr[1]*c₂(qr[2]) + (1-qr[2])*c₁(qr[1]) + qr[2]*c₃(qr[1]) - 
((1-qr[2])*(1-qr[1])*P₀₁ + qr[2]*qr[1]*P₂₃ + qr[2]*(1-qr[1])*P₃₀ + (1-qr[2])*qr[1]*P₁₂);

"""
Function to return the Jacobian of the transformation. 
The entries of the matrices are 
  J(f)[j,k] = ∂f(x)[j]/∂x[k]
i.e.,
  J = [∂f₁/∂x₁ ∂f₁/∂x₂
       ∂f₂/∂x₁ ∂f₂/∂x₂]
We require the transpose in our computations.
Here the parameters
- S is the transfinite interpolation operator
- qr is the (q,r) pair in the reference grid
"""
function J(S, qr)
  SMatrix{2,2,Float64}(ForwardDiff.jacobian(S, qr))'
end

"""
Function to return the inverse of the Jacobian matrix
Here the parameters
- S is the transfinite interpolation operator
- qr is the (q,r) pair in the reference grid
"""
function J⁻¹(S, qr)
  inv(J(S, qr))
end

"""
Function to compute the surface jacobian. 
Here the parameters
- n is the normal vector in the reference domain.
- S is the transfinite interpolation operator
- qr is the (q,r) pair in the reference grid
"""
function J⁻¹s(S, qr, n)  
  norm(J⁻¹(S, qr)*n)
end


"""
Fancy defintion of the Kronecker product
"""
⊗(A,B) = kron(A,B)

"""
Function to return the material tensor in the reference coordinates (0,1)×(0,1). Returns 
  𝒫ₜ = S'*𝒫*S
where S = (J⁻¹ ⊗ I(2)) is the transformation matrix.
"""
function t𝒫(𝒮, qr)
  x = 𝒮(qr)  
  invJ = J⁻¹(𝒮, qr)
  S = invJ ⊗ I(2)  
  S'*𝒫(x)*S
end

"""
The transformed material coefficient matrices in the reference coordinates (0,1)×(0,1). 
Extracted from the bigger matrix:
    t𝒫(𝒮, (q,r)) = (J⁻¹ ⊗ I(2))ᵀ * 𝒫(x(q,r), y(q,r))* (J⁻¹ ⊗ I(2))
  A(x) -> Aₜ(r)
  B(x) -> Bₜ(r)
  C(x) -> Cₜ(r) 
"""
Aₜ(qr) = t𝒫(𝒮,qr)[1:2, 1:2];
Bₜ(qr) = t𝒫(𝒮,qr)[3:4, 3:4];
Cₜ(qr) = t𝒫(𝒮,qr)[1:2, 3:4];

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Unit normals on the boundary parameterized as c(u).
  𝐧(c, u; o=1.0)
The parameter o=1 or -1 depends on how the surface is oriented. Can be skipped, but defaults to 1.

EXPLANATION:
  I traverse left to right (u = 0..1) always, so the normal direction obeys the right-hand rule.
  Thus -1 needs to be multiplied to obtain the correct normal on the bottom and right boundaries.
"""
function 𝐧(c,u; o=1.0) 
  res = ForwardDiff.derivative(t->c(t), u) # Tangent vector in the physical domain
  r = @SMatrix [0 -1; 1 0] # Rotation matrix
  o*r*res/norm(res) # Normal vector
end
