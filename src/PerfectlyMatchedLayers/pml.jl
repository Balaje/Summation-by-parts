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
  A::SparseMatrixCSC{Float64, Int64}
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
  Pᴾᴹᴸ([D[1,1] D[1,2] D[1,3] D[1,4]; D[2,1] D[2,2] D[2,3] D[2,4]] + [D[3,1] D[3,2] D[3,3] D[3,4]; D[4,1] D[4,2] D[4,3] D[4,4]])
end

"""
Function to obtain the Impedance matrix
"""
function 𝐙(𝒫, Ω, qr)
  Pqr = 𝒫.(Ω.(qr))
  Jqr = J⁻¹.(qr, Ω)
  P_vec = get_property_matrix_on_grid(Pqr, 2)
  J_vec = get_property_matrix_on_grid(Jqr, 2)
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  J_vec_diag = [spdiagm(vec(p)) for p in J_vec]
  [P_vec_diag[1,1] P_vec_diag[1,2]; P_vec_diag[2,1] P_vec_diag[2,2]]*[J_vec_diag[1,1] J_vec_diag[1,2]; J_vec_diag[2,1]  J_vec_diag[2,2]]
end

"""
Function to obtain the Characteristic boundary condition
"""
function χᴾᴹᴸ()