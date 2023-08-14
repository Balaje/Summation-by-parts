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
PML properties matrix
"""
function Π(𝒫, Ω, qr) 
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)   
  S = ([σₚ(x) 0; 0 σₚ(x)].*invJ) ⊗ I(2)  
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫(x))
end


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
Transform the material properties to the reference grid
"""
function t𝒫(𝒮, qr)
  x = 𝒮(qr)
  invJ = J⁻¹(qr, 𝒮)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫(x)*S)
end

"""
Structure to define the PML part of the elasticity equation
"""
struct Dᴾᴹᴸ <: SBP.SBP_TYPE
  A::Matrix{SparseMatrixCSC{Float64, Int64}}
end
function Dᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
  m,n = size(Pqr)
  Ptuple = Tuple.(Pqr)
  P_page = reinterpret(reshape, Float64, Ptuple)
  dim = length(size(P_page))
  P_vec = reshape(splitdimsview(P_page, dim-2), (4,4))
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1      
  Dᴱ = [[Dq] [Dq] [Dr] [Dr]; [Dq] [Dq] [Dr] [Dr]; [Dq] [Dq] [Dr] [Dr]; [Dq] [Dq] [Dr] [Dr]]
  res = [Dᴱ[i,j]*P_vec_diag[i,j] for i=1:4, j=1:4]
  Dᴾᴹᴸ(res)
end

function Pᴾᴹᴸ(D1::Dᴾᴹᴸ)
  D = D1.A
  [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[1,3] D[1,4]; D[2,3] D[2,4]], [D[3,1] D[4,1]; D[3,2] D[4,2]] + [D[3,3] D[3,4]; D[4,3] D[4,4]]
end 


function Kᴾᴹᴸ(𝐪𝐫)  
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  
  Jinv = Tuple.(J⁻¹.(𝐪𝐫, Ω))
  Jinv_page = reinterpret(reshape, Float64, Jinv)
  dim = length(size(Jinv_page))
  Jinv_vec = reshape(splitdimsview(Jinv_page, dim-2), (2,2))
  Jinv_vec_diag  = [spdiagm(vec(p)) for p in Jinv_vec]

  Dq, Dr = sbp_2d.D1 
  Z = spzeros(Float64, 2m^2, 2n^2)  
  Bulk_u = Pᴱ(Dᴱ(t𝒫.(Ω,𝐪𝐫)));
  PML_v, PML_w = Pᴾᴹᴸ(Dᴾᴹᴸ(Π.(𝒫, Ω, 𝐪𝐫)));
  JD₁ = [I(2)⊗Jinv_vec_diag[1,1] I(2)⊗Jinv_vec_diag[1,2]]*vcat(I(2)⊗Dq, I(2)⊗Dr)
  JD₂ = [I(2)⊗Jinv_vec_diag[2,1] I(2)⊗Jinv_vec_diag[2,2]]*vcat(I(2)⊗Dq, I(2)⊗Dr)
  σα = I(2) ⊗ spdiagm(vec(σₚ.(𝐪𝐫)) .+ α)
  𝛂 = α*sparse(I(2)⊗I(m)⊗I(n))
  Σ = [Bulk_u   -PML_v     PML_w     Z;
       JD₁      -σα       Z         Z;
       JD₂      Z         -𝛂        Z;
       𝛂        Z         Z         -𝛂]
end
 
𝐪𝐫 = generate_2d_grid((21,21));
stima = Kᴾᴹᴸ(𝐪𝐫);