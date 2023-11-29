"""
Transform the PML properties to the material grid
"""
function P2Rᴾᴹᴸ(𝒫ᴾᴹᴸ, Ω, qr)
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)  
  detJ = (det∘J)(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(detJ*S'*𝒫ᴾᴹᴸ(x))
end 

"""
SBP operator to approximate the PML part:
1) Pᴾᴹᴸ(Pqr) ≈ 𝛛/𝛛𝐪(𝐀 ) +  𝛛/𝛛𝐫(𝐁 )
    (-) Asssemble bulk PML difference operator
"""
struct Pᴾᴹᴸ
  A::Tuple{SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{Float64, Int64}}
end
function Pᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
  P_vec = get_property_matrix_on_grid(Pqr, 2)
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  n,m = size(Pqr)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1
  I1 = [1 1 1 1; 1 1 1 1]
  D₁ = vcat(I1⊗[Dq], I1⊗[Dr])
  D = [D₁[i,j]*P_vec_diag[i,j] for i=1:4, j=1:4]  
  X = [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[3,1] D[3,2];  D[4,1] D[4,2]]
  Y = [D[1,3] D[1,4]; D[2,3] D[2,4]] + [D[3,3] D[3,4];  D[4,3] D[4,4]]
  Pᴾᴹᴸ((X,Y))
end

"""
Function to obtain the Impedance matrix
"""
function 𝐙(𝒫, Ω, qr)
  𝒫₁, 𝒫₂ = 𝒫
  𝐉⁻¹(qr) = J⁻¹(qr, Ω) ⊗ I(size(𝒫₁(Ω(qr)),1))
  𝐏(qr) = (E1(1,1,(2,2)) ⊗ 𝒫₁(Ω(qr))) + (E1(2,2,(2,2)) ⊗ 𝒫₂(Ω(qr)))  
  get_property_matrix_on_grid(𝐏.(qr).*𝐉⁻¹.(qr), 2)  
end

"""
Function to obtain the Traction with PML
"""
struct Tᴾᴹᴸ
  A::Tuple{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}}
end
function Tᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}}, 𝛀::DiscreteDomain, 𝐧::AbstractVecOrMat{Int64}; X=[1]) 
  P_vec = spdiagm.(vec.(get_property_matrix_on_grid(Pqr,2)))
  # Compute the traction
  𝐧 = reshape(𝐧, (1,2))
  JJ = Js(𝛀, 𝐧; X=I(2)) 
  JJ⁻¹ = JJ\I(size(JJ,1)) 
  Pn = ([P_vec[1,1]  P_vec[1,2]; P_vec[2,1]  P_vec[2,2]]*abs(𝐧[1]) + [P_vec[3,1]  P_vec[3,2]; P_vec[4,1]  P_vec[4,2]]*abs(𝐧[2]), 
        [P_vec[1,3]   P_vec[1,4]; P_vec[2,3]  P_vec[2,4]]*abs(𝐧[1]) + [P_vec[3,3]   P_vec[3,4]; P_vec[4,3]  P_vec[4,4]]*abs(𝐧[2]))
  Tr₁, Tr₂ = JJ⁻¹*Pn[1], JJ⁻¹*Pn[2]
  Tᴾᴹᴸ((X⊗Tr₁, X⊗Tr₂))
end

"""
Function to obtain the characteristic boundary condition
"""
struct χᴾᴹᴸ
  A::Vector{SparseMatrixCSC{Float64, Int64}}
end
function χᴾᴹᴸ(PQR, 𝛀::DiscreteDomain, 𝐧::AbstractVecOrMat{Int64}; X=[1]) 
  Pqrᴱ, Pqrᴾᴹᴸ, Z₁₂, σ₁₂¹, σ₁₂², J = PQR  
  # [Zx, Zy](∂u/∂t)
  impedance_normal = Z₁₂*(vec(abs.(𝐧))⊗[1;1])  
  impedance_normal_vec = [spdiagm(vec(p)) for p in impedance_normal]  
  Z₁ = blockdiag(impedance_normal_vec[1], impedance_normal_vec[2])
  Z₂ = blockdiag(impedance_normal_vec[3], impedance_normal_vec[4])
  # [Zx*σy - Zx*σx*σy, Zy*σx - Zy*σx*σy] (u - q)
  mass_p = abs(𝐧[1])*J*Z₁ + abs(𝐧[2])*J*Z₂
  T_elas_u = Tᴱ(Pqrᴱ, 𝛀, 𝐧).A
  T_pml_v, T_pml_w = Tᴾᴹᴸ(Pqrᴾᴹᴸ, 𝛀, 𝐧).A
  impedance_u_normal = σ₁₂¹*(vec(abs.(𝐧))⊗[1;1])
  impedance_u_normal_vec = [spdiagm(vec(p)) for p in impedance_u_normal]  
  σᵥqr = blockdiag(impedance_u_normal_vec[1], impedance_u_normal_vec[2])
  σₕqr = blockdiag(impedance_u_normal_vec[3], impedance_u_normal_vec[4])
  impedance_u = abs(𝐧[1])*J*σᵥqr + abs(𝐧[2])*J*σₕqr  
  impedance_q = impedance_u
  # [Zx*σx*σy, Zy*σx*σy](u - q - r)
  impedance_r_normal = σ₁₂²*(vec(abs.(𝐧))⊗[1;1])
  impedance_r_normal_vec = [spdiagm(vec(p)) for p in impedance_r_normal]    
  σₕσᵥqr = blockdiag(impedance_r_normal_vec[1], impedance_r_normal_vec[2])
  impedance_r = abs(𝐧[1])*J*σₕσᵥqr + abs(𝐧[2])*J*σₕσᵥqr
  𝐧 = reshape(𝐧, (1,2))
  JJ = Js(𝛀, 𝐧; X=I(2))  
  JJ⁻¹ = sparse(JJ\I(size(JJ,1)))
  χᴾᴹᴸ([sum(𝐧)*T_elas_u + (JJ⁻¹*(impedance_u + impedance_r)), JJ⁻¹*mass_p, 𝐧[1]*T_pml_v, 𝐧[2]*T_pml_w, -JJ⁻¹*(impedance_q + impedance_r), -JJ⁻¹*impedance_r])
end