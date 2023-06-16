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
c₀(r) = @SVector [0.1*r*(1-r), r] # Left boundary 
c₁(q) = @SVector [q, 0.0*q*(1-q)] # Bottom boundary
c₂(r) = @SVector [1.0 - 0.0*r*(1-r), r] # Right boundary
c₃(q) = @SVector [q, 1.0 - 0.0*q*(1-q)] # Top boundary

# Get the intersection points
P₀₁ = SVector{2}(P(c₀,c₁));
P₁₂ = SVector{2}(P(c₁,c₂));
P₂₃ = SVector{2}(P(c₂,c₃));
P₃₀ = SVector{2}(P(c₃,c₀));

"""
The transfinite interpolation formula
"""
𝒮(qr) = (1-qr[1])*c₀(qr[2]) + qr[1]*c₂(qr[2]) + (1-qr[2])*c₁(qr[1]) + qr[2]*c₃(qr[1]) - 
((1-qr[2])*(1-qr[1])*P₀₁ + qr[2]*qr[1]*P₂₃ + qr[2]*(1-qr[1])*P₃₀ + (1-qr[2])*qr[1]*P₁₂);

"""
Function to return the Jacobian of the transformation
"""
function J(S, qr)
  SMatrix{2,2,Float64}(ForwardDiff.jacobian(S, qr))'
end

"""
Function to return the inverse of the Jacobian
"""
function J⁻¹(S, qr)
  inv(J(S, qr))
end

"""
Function to compute the surface jacobian
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
  𝒫' = S*𝒫*S'
where S is the transformation matrix
"""
function t(𝒮, qr)
  x = 𝒮(qr)  
  invJ = J⁻¹(𝒮, qr)      
  S = invJ ⊗ I(2)  
  S'*𝒫(x)*S
end

"""
The material coefficient matrices in the reference coordinates (0,1)×(0,1).
  A(x) -> Aₜ(r)
  B(x) -> Bₜ(r)
  C(x) -> Cₜ(r) 
"""
Aₜ(qr) = t(𝒮,qr)[1:2, 1:2];
Bₜ(qr) = t(𝒮,qr)[3:4, 3:4];
Cₜ(qr) = t(𝒮,qr)[1:2, 3:4];

"""
Flatten the 2d function as a single vector for the time iterations
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Unit normals on the boundary
"""
function 𝐧(c,u; o=1.0) 
  res = ForwardDiff.derivative(t->c(t), u) # Tangent vector in the physical domain
  r = @SMatrix [0 -1; 1 0] # Rotation matrix
  o*r*res/norm(res) # Normal vector
end