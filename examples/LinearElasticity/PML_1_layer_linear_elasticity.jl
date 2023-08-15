###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine

const α = 1.0; # The frequency shift parameter

"""
The PML damping function
"""
function σₚ(x)
  1.0
end

# Define the domain
c₀(r) = @SVector [0.0, r]
c₁(q) = @SVector [q, 0.0]
c₂(r) = @SVector [1.0, r]
c₃(q) = @SVector [q, 1.0]
domain = domain_2d(c₀, c₁, c₂, c₃)
Ω(qr) = S(qr, domain)

"""
The Lamé parameters μ, λ
"""
λ(x) = 2.0
μ(x) = 1.0

"""
The density of the material
"""
ρ(x) = 1.0

"""
Material properties coefficients of an anisotropic material
"""
c₁₁(x) = 2*μ(x)+λ(x)
c₂₂(x) = 2*μ(x)+λ(x)
c₃₃(x) = μ(x)
c₁₂(x) = λ(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫(x) = @SMatrix [c₁₁(x) 0 0 c₁₂(x); 0 c₃₃(x) c₃₃(x) 0; 0 c₃₃(x) c₃₃(x) 0; c₁₂(x) 0 0 c₂₂(x)];

"""
The material property tensor with the PML is given as follows:
  𝒫ᴾᴹᴸ(x) = [-σₚ(x)*A(x)      0; 
              0         σₚ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫ᴾᴹᴸ(x) = @SMatrix [-σₚ(x)*c₁₁(x) 0 0 0; 0 -σₚ(x)*c₃₃(x) 0 0; 0 0 σₚ(x)*c₃₃(x) 0; 0 0 0 σₚ(x)*c₂₂(x)];

"""
Transform the material property matrix to the reference grid
"""
function t𝒫(Ω, qr)
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫(x)*S)
end 

"""
Transform the PML properties to the material grid
"""
function t𝒫ᴾᴹᴸ(Ω, qr)
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫ᴾᴹᴸ(x))
end 

"""
Function to get the property tensors on the grid
Input a Matrix or Vector of Tensors (in turn a matrix) evaluated on the grid points.
  Pqr::Matrix{SMatrix{m,n,Float64}}
    = [P(x₁₁) P(x₁₂) ... P(x₁ₙ)
       P(x₂₁) P(x₂₂) ... P(x₂ₙ)
       ...
       P(xₙ₁) P(xₙ₂) ... P(xₙₙ)]
  where P(x) = [P₁₁(x) P₁₂(x)
                P₂₁(x) P₂₂(x)]
Returns a matrix of matrix with the following form
   result = [ [P₁₁(x₁₁) ... P₁₁(x₁ₙ)        [P₁₂(x₁₁) ... P₁₂(x₁ₙ)
               ...                          ...
               P₁₁(xₙ₁) ... P₁₁(xₙₙ)],         P₁₂(xₙ₁) ... P₁₂(x₁ₙ)];              
               [P₂₁(x₁₁) ... P₂₁(x₁ₙ)        [P₂₂(x₁₁) ... P₂₂(x₁ₙ)
                ...                          ...
                P₂₁(xₙ₁) ... P₂₁(xₙₙ)],         P₂₂(xₙ₁) ... P₂₂(x₁ₙ)] 
            ]
"""
function get_property_matrix_on_grid(Pqr)
  m,n = size(Pqr[1])
  Ptuple = Tuple.(Pqr)
  P_page = reinterpret(reshape, Float64, Ptuple)
  dim = length(size(P_page))
  reshape(splitdimsview(P_page, dim-2), (m,n))
end

"""
SBP operator to approximate the PML part. 
Contains a matrix of sparse matrices representing the individual derivatives of the PML part
"""
function Dᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
  P_vec = get_property_matrix_on_grid(Pqr)
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  m, n = size(Pqr)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1
  I1 = [1 1 1 1; 1 1 1 1]
  D₁ = vcat(I1⊗[Dq], I1⊗[Dr])
  res = [D₁[i,j]*P_vec_diag[i,j] for i=1:4, j=1:4]
  res
end

"""
Assemble the PML contribution in the stiffness matrix
"""
function Pᴾᴹᴸ(D::Matrix{SparseMatrixCSC{Float64, Int64}})
  [D[1,1] D[1,2] D[1,3] D[1,4]; 
  D[2,1] D[2,2] D[2,3] D[2,4]] + 
  [D[3,1] D[3,2] D[3,3] D[3,4]; 
  D[4,1] D[4,2] D[4,3] D[4,4]]
end

function 𝐊ᴾᴹᴸ(𝐪𝐫, Ω)
  detJ(x) = (det∘J)(x,Ω)
  detJ𝒫(x) = detJ(x)*t𝒫(Ω, x)
  detJ𝒫ᴾᴹᴸ(x) = detJ(x)*t𝒫ᴾᴹᴸ(Ω, x)

  P = t𝒫.(Ω,𝐪𝐫) # Elasticity Bulk (For traction)
  JP = detJ𝒫.(𝐪𝐫) # Elasticity Bulk with det(J) multiplied
  PML =  t𝒫ᴾᴹᴸ.(Ω, 𝐪𝐫) # PML Bulk (For traction??)
  JPML =  detJ𝒫ᴾᴹᴸ.(𝐪𝐫) # PML Bulk with det(J) multiplied

  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1

  # Bulk stiffness matrix
  𝐏 = Pᴱ(Dᴱ(JP))  
  𝐏ᴾᴹᴸ = Pᴾᴹᴸ(Dᴾᴹᴸ(JPML))
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Z = zero(Id)
  σ = I(2) ⊗ spdiagm(vec(σₚ.(𝐪𝐫)))
  σpα = I(2) ⊗ spdiagm(α .+ vec(σₚ.(𝐪𝐫)))  
  ρσ = I(2) ⊗ spdiagm(vec(ρ.(𝐪𝐫).*σₚ.(𝐪𝐫)))
  ρσα = α*ρσ

  # Get the derivate matrix transformed to the reference grid
  Jinv_vec = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω))
  Jinv_vec_diag = [spdiagm(vec(p)) for p in Jinv_vec]
  JD₁ = [(I(2)⊗Jinv_vec_diag[1,1]) (I(2)⊗Jinv_vec_diag[1,2])]*vcat((I(2)⊗Dq), (I(2)⊗Dr))
  JD₂ = [(I(2)⊗Jinv_vec_diag[2,1]) (I(2)⊗Jinv_vec_diag[2,2])]*vcat((I(2)⊗Dq), (I(2)⊗Dr))

  # Assemble the bulk stiffness matrix
  Σ = [Z      Z       Z       Z       Id;
       JD₁    -σpα    Z       Z       Z;
       JD₂    Z      -α*Id    Z       Z;
       α*Id   Z       Z     -α*Id     Z;
       (𝐏+ρσα) (𝐏ᴾᴹᴸ)        -ρσα    -ρσ]

  # Get the traction operator of the elasticity part
  𝐓 = Tᴱ(P)
  𝐓q, 𝐓r = 𝐓.A, 𝐓.B

  # TODO: The SAT Terms
end 

function 𝐌ᴾᴹᴸ(𝐪𝐫, Ω)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  ρᵥ = I(2)⊗spdiagm(vec(ρ.(Ω.(𝐪𝐫))))
  blockdiag(Id, Id, Id, Id, ρᵥ)
end 

𝐪𝐫 = generate_2d_grid((21,21));
stima = 𝐊ᴾᴹᴸ(𝐪𝐫, Ω);
massma = 𝐌ᴾᴹᴸ(𝐪𝐫, Ω);