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
  m, n = size(Pqr)
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
  𝐉⁻¹(qr) = J⁻¹(qr, Ω) ⊗ I(size(𝒫₁(qr),1))
  𝐏(qr) = (E1(1,1,(2,2)) ⊗ 𝒫₁(qr)) + (E1(2,2,(2,2)) ⊗ 𝒫₂(qr))
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
  P = [[[P_vec[1,1]  P_vec[1,2]; P_vec[2,1]  P_vec[2,2]]] [[P_vec[1,3]   P_vec[1,4]; P_vec[2,3]  P_vec[2,4]]]; 
       [[P_vec[3,1]  P_vec[3,2]; P_vec[4,1]  P_vec[4,2]]] [[P_vec[3,3]   P_vec[3,4]; P_vec[4,3]  P_vec[4,4]]]]  
  # Compute the traction
  𝐧 = reshape(𝐧, (1,2))
  JJ = Js(𝛀, 𝐧; X=I(2))  
  Pn = (𝐧*P)  
  Tr₁, Tr₂ = JJ\Pn[1], JJ\Pn[2]
  Tᴾᴹᴸ((X⊗Tr₁, X⊗Tr₂))
end

"""
Function to obtain the characteristic boundary condition
"""
struct χᴾᴹᴸ
  A::Vector{SparseMatrixCSC{Float64, Int64}}
end
function χᴾᴹᴸ(PQR, 𝛀::DiscreteDomain, 𝐧::AbstractVecOrMat{Int64}; X=[1]) 
  Pqrᴱ, Pqrᴾᴹᴸ, Z₁₂, σᵥqr, σₕqr, J = PQR  
  impedance_normal = Z₁₂*(vec(abs.(𝐧))⊗[1;1])  
  impedance_normal_vec = [spdiagm(vec(p)) for p in impedance_normal]  
  Z₁ = blockdiag(impedance_normal_vec[1], impedance_normal_vec[2])
  Z₂ = blockdiag(impedance_normal_vec[3], impedance_normal_vec[4])
  mass_p = abs(𝐧[1])*J*Z₁ + abs(𝐧[2])*J*Z₂
  T_elas_u = Tᴱ(Pqrᴱ, 𝛀, 𝐧).A
  T_pml_v, T_pml_w = Tᴾᴹᴸ(Pqrᴾᴹᴸ, 𝛀, 𝐧).A
  impedance_u = 𝐧[1]*Z₁*σᵥqr + 𝐧[2]*Z₂*σₕqr  
  impedance_q = impedance_u
  impedance_r = 𝐧[1]*Z₁*σₕqr*σᵥqr + 𝐧[2]*Z₂*σₕqr*σᵥqr
  𝐧 = reshape(𝐧, (1,2))
  JJ = Js(𝛀, 𝐧; X=I(2)) 
  χᴾᴹᴸ([sum(𝐧)*T_elas_u + 0*(JJ\(impedance_u + impedance_r)), JJ\mass_p, T_pml_v, T_pml_w, -JJ\(impedance_q + impedance_r), -JJ\impedance_r])
end