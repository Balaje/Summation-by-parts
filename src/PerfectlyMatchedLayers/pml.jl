"""
Transform the PML properties to the material grid. 
Similar signature to TransfiniteInterpolation.transform_material_properties()
"""
function transform_material_properties_pml(P::Function, Ω::Function, qr::SVector{2,Float64})
  x = Ω(qr)
  invJ = inverse_transfinite_interpolation_jacobian(qr, Ω)  
  detJ = (det∘transfinite_interpolation_jacobian)(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(detJ*S'*P(x))
end 

"""
SBP operator to approximate the PML part:
1) elasticity_pml_operator(Pqr) ≈ 𝛛/𝛛𝐪(𝐀 ) +  𝛛/𝛛𝐫(𝐁 )
    (-) Asssemble bulk PML difference operator
"""
struct elasticity_pml_operator <: SBP_TYPE
  A::NTuple{2,SparseMatrixCSC{Float64, Int64}}
end
function elasticity_pml_operator(Pqr::Matrix{SMatrix{4,4,Float64,16}})
  P_vec = get_property_matrix_on_grid(Pqr, 2)
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  n,m = size(Pqr)
  sbp_q = SBP4_1D(m)
  sbp_r = SBP4_1D(n)
  sbp_2d = SBP4_2D(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1
  I1 = [1 1 1 1; 1 1 1 1]
  D₁ = vcat(I1⊗[Dq], I1⊗[Dr])
  D = [D₁[i,j]*P_vec_diag[i,j] for i=1:4, j=1:4]  
  X = [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[3,1] D[3,2];  D[4,1] D[4,2]]
  Y = [D[1,3] D[1,4]; D[2,3] D[2,4]] + [D[3,3] D[3,4];  D[4,3] D[4,4]]
  elasticity_pml_operator((X,Y))
end

"""
Function to obtain the Impedance function on the grid
"""
function compute_impedance_function(P::NTuple{2,Function}, Ω::Function, qr::AbstractMatrix{SVector{2,Float64}})
  P₁, P₂ = P
  𝐉⁻¹(qr) = inverse_transfinite_interpolation_jacobian(qr, Ω) ⊗ I(size(P₁(Ω(qr)),1))
  𝐏(qr) = (δᵢⱼ(1,1,(2,2)) ⊗ P₁(Ω(qr))) + (δᵢⱼ(2,2,(2,2)) ⊗ P₂(Ω(qr)))  
  get_property_matrix_on_grid(𝐏.(qr).*𝐉⁻¹.(qr), 2)  
end

"""
Function to obtain the Traction with PML
"""
struct elasticity_traction_pml_operator <: SBP_TYPE
  A::NTuple{2,SparseMatrixCSC{Float64, Int64}}
end
function elasticity_traction_pml_operator(Pqr::Matrix{SMatrix{4,4,Float64,16}}, Ω::Function, qr::AbstractMatrix{SVector{2,Float64}}, 𝐧::AbstractVecOrMat{Int64}; X=[1]) 
  P_vec = spdiagm.(vec.(get_property_matrix_on_grid(Pqr,2)))
  # Compute the traction
  𝐧 = reshape(𝐧, (1,2))
  J = surface_jacobian(Ω, qr, 𝐧; X=I(2)) 
  J⁻¹ = J\I(size(JJ,1)) 
  Pn = ([P_vec[1,1]  P_vec[1,2]; P_vec[2,1]  P_vec[2,2]]*abs(𝐧[1]) + [P_vec[3,1]  P_vec[3,2]; P_vec[4,1]  P_vec[4,2]]*abs(𝐧[2]), 
        [P_vec[1,3]  P_vec[1,4]; P_vec[2,3]  P_vec[2,4]]*abs(𝐧[1]) + [P_vec[3,3]  P_vec[3,4]; P_vec[4,3]  P_vec[4,4]]*abs(𝐧[2]))
  Tr₁, Tr₂ = J⁻¹*Pn[1], J⁻¹*Pn[2]
  elasticity_traction_pml_operator((X⊗Tr₁, X⊗Tr₂))
end

"""
Function to obtain the characteristic boundary condition
"""
struct elasticity_absorbing_boundary_pml_operator <: SBP_TYPE
  A::NTuple{5,SparseMatrixCSC{Float64, Int64}}
end
function elasticity_absorbing_boundary_pml_operator(coeffs, Ω::Function, qr::AbstractMatrix{SVector{2,Float64}}, 𝐧::AbstractVecOrMat{Int64}; X=[1]) 
  # Pqrᴱ, Pqrᴾᴹᴸ, Z₁₂, σ₁₂¹, σ₁₂², J = coeffs
  elastic_properties, pml_elastic_properties, impedances, horz_pml, vert_pml, jacobian = coeffs
  # [Zx, Zy](∂u/∂t)
  impedance_normal = impedances*(vec(abs.(𝐧))⊗[1;1])  
  impedance_normal_vec = [spdiagm(vec(p)) for p in impedance_normal]  
  Z1 = blockdiag(impedance_normal_vec[1], impedance_normal_vec[2])
  Z2 = blockdiag(impedance_normal_vec[3], impedance_normal_vec[4])
  # [Zx*σy - Zx*σx*σy, Zy*σx - Zy*σx*σy] (u - q)
  mass_p = abs(𝐧[1])*jacobian*Z1 + abs(𝐧[2])*jacobian*Z2
  T_elas_u = elasticity_traction_operator(elastic_properties, Ω, qr, 𝐧).A
  T_pml_v, T_pml_w = elasticity_traction_pml_operator(pml_elastic_properties, Ω, qr, 𝐧).A
  impedance_u_normal = horz_pml*(vec(abs.(𝐧))⊗[1;1])
  impedance_u_normal_vec = [spdiagm(vec(p)) for p in impedance_u_normal]  
  σᵥqr = blockdiag(impedance_u_normal_vec[1], impedance_u_normal_vec[2])
  σₕqr = blockdiag(impedance_u_normal_vec[3], impedance_u_normal_vec[4])
  impedance_u = abs(𝐧[1])*jacobian*σᵥqr + abs(𝐧[2])*jacobian*σₕqr  
  impedance_q = impedance_u
  # [Zx*σx*σy, Zy*σx*σy](u - q - r)
  impedance_r_normal = vert_pml*(vec(abs.(𝐧))⊗[1;1])
  impedance_r_normal_vec = [spdiagm(vec(p)) for p in impedance_r_normal]    
  σₕσᵥqr = blockdiag(impedance_r_normal_vec[1], impedance_r_normal_vec[2])
  impedance_r = abs(𝐧[1])*J*σₕσᵥqr + abs(𝐧[2])*J*σₕσᵥqr
  𝐧 = reshape(𝐧, (1,2))
  J = surface_jacobian(Ω, qr, 𝐧; X=I(2))  
  J⁻¹ = sparse(J\I(size(J,1)))
  elasticity_absorbing_boundary_operator_pml((sum(𝐧)*T_elas_u + (J⁻¹*(impedance_u + impedance_r)), J⁻¹*mass_p, 
                                                       𝐧[1]*T_pml_v, 𝐧[2]*T_pml_w, 
                                                       -J⁻¹*(impedance_q + impedance_r), -J⁻¹*impedance_r))
end