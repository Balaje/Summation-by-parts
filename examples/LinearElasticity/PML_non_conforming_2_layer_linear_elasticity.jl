##################################################
# Program to solve the 2 layer linear elasticity #
# Incluing the Perfectly Matched Layer Boundary  #
##################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine
using LoopVectorization

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [1,2]
# Define the parametrization for interface
# pf = 8
f(q) = 0.1*sin(π*q)
# f(q) = 0.1*exp(-16π*(1.1*(q-0.5))^2)
cᵢ(q) = [4.4π*q, 4π*f(q)];
# Define the rest of the boundary
c₀¹(r) = [0.0, 4π*r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [4.4π, 4π*r]; # Right boundary
c₃¹(q) = [4.4π*q, 4π]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, 4π*r - 4π]; # Left boundary
c₁²(q) = [4.4π*q, -4π]; # Bottom boundary. 
c₂²(r) = [4.4π, 4π*r - 4π]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₁(qr) = S(qr, domain₁)
Ω₂(qr) = S(qr, domain₂)

###############################################
# We use different properties for both layers #
###############################################
"""
The Lamé parameters μ, λ
"""
λ¹(x) = 4.8629
μ¹(x) = 4.86
λ²(x) = 26.9952
μ²(x) = 27.0

"""
Material properties coefficients of an anisotropic material
"""
c₁₁¹(x) = 2*μ¹(x)+λ¹(x)
c₂₂¹(x) = 2*μ¹(x)+λ¹(x)
c₃₃¹(x) = μ¹(x)
c₁₂¹(x) = λ¹(x)
c₁₁²(x) = 2*μ²(x)+λ²(x)
c₂₂²(x) = 2*μ²(x)+λ²(x)
c₃₃²(x) = μ²(x)
c₁₂²(x) = λ²(x)

"""
Density of the material
"""
ρ¹(x) = 1.5
ρ²(x) = 3.0

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫¹(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫²(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ¹(∇u,x) = 𝒫¹(x)*∇u
σ²(∇u,x) = 𝒫²(x)*∇u

"""
The PML damping
"""
const δ = 0.4π
const Lₓ = 4π
const σ₀ = 4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀*0.05; # The frequency shift parameter
function σₚ(x)
  if((x[1] ≈ Lₓ) || (x[1] > Lₓ))
    return σ₀*((x[1] - Lₓ)/δ)^3  
  else
    return 0.0
  end
end

"""
The material property tensor with the PML is given as follows:
  𝒫ᴾᴹᴸ(x) = [-σₚ(x)*A(x)      0; 
                 0         σₚ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫ᴾᴹᴸ₁(x) = @SMatrix [-σₚ(x)*c₁₁¹(x) 0 0 0; 0 -σₚ(x)*c₃₃¹(x) 0 0; 0 0 σₚ(x)*c₃₃¹(x) 0; 0 0 0 σₚ(x)*c₂₂¹(x)];
𝒫ᴾᴹᴸ₂(x) = @SMatrix [-σₚ(x)*c₁₁²(x) 0 0 0; 0 -σₚ(x)*c₃₃²(x) 0 0; 0 0 σₚ(x)*c₃₃²(x) 0; 0 0 0 σₚ(x)*c₂₂²(x)];

"""
Transform the PML properties to the material grid
"""
function P2Rᴾᴹᴸ(𝒫ᴾᴹᴸ, Ω, qr)
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫ᴾᴹᴸ(x))
end 

"""
SBP operator to approximate the PML part: Contains two parts
1) Contains a 4×4 matrix of sparse matrices representing the individual derivatives of the PML part
    (-) 𝛛/𝛛𝐪(𝐀 ) : 4 sparse matrices
    (-) 𝛛/𝛛𝐪(𝟎 ) : 4 sparse matrices
    (-) 𝛛/𝛛𝐫(𝟎 ) : 4 sparse matrices 
    (-) 𝛛/𝛛𝐫(𝐁 ) : 4 sparse matrices
2) Pᴾᴹᴸ(Dᴾᴹᴸ(Pqr)) ≈ 𝛛/𝛛𝐪(𝐀 ) +  𝛛/𝛛𝐫(𝐁 )
    (-) Asssemble the PML matrices to obtain the bulk PML difference operator
"""
struct Dᴾᴹᴸ
  A::Matrix{SparseMatrixCSC{Float64, Int64}}
end
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
function Pᴾᴹᴸ(D::Matrix{SparseMatrixCSC{Float64, Int64}})
  [D[1,1] D[1,2] D[1,3] D[1,4]; 
  D[2,1] D[2,2] D[2,3] D[2,4]] + 
  [D[3,1] D[3,2] D[3,3] D[3,4]; 
  D[4,1] D[4,2] D[4,3] D[4,4]]
end

"""
Function to obtain the PML contribution to the traction on the boundary:
Tᴾᴹᴸ(Pqr, Zxy, σₚ, Ω, 𝐪𝐫)
1) Pqr: PML Material tensor evaluated at the grid points
2) Zxy: Impedance matrices evaluated at the grid points
3) σₚ: PML damping function
4) Ω: Physical to Reference map
5) 𝐪𝐫: Reference coordinates
"""
function Tᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}}, Zxy::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}},
              σₚ::Function, Ω::Function, 𝐪𝐫::Matrix{SVector{2, Float64}})
  P_vec = get_property_matrix_on_grid(Pqr)
  P_vec_diag = [spdiagm(vec(p)) for p in P_vec]
  m, n = size(Pqr)
  Z = spzeros(Float64, 2m^2, 2n^2)  
  # Get the trace norms
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm
  # Get the physical coordinates
  Zx, Zy = Zxy
  𝐱𝐲 = Ω.(𝐪𝐫)    
  SJr₀ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω))*J⁻¹s([q,0.0], Ω, [0,-1])) for q in LinRange(0,1,m)].^-1)⊗E1(1,1,m), m)
  SJq₀ = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω))*J⁻¹s([0.0,q], Ω, [-1,0])) for q in LinRange(0,1,m)].^-1), m)
  SJrₙ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω))*J⁻¹s([q,1.0], Ω, [0,1])) for q in LinRange(0,1,m)].^-1)⊗E1(m,m,m), m)
  SJqₙ = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω))*J⁻¹s([1.0,q], Ω, [1,0])) for q in LinRange(0,1,m)].^-1), m)
  σ = I(2) ⊗ (spdiagm(vec(σₚ.(𝐱𝐲))))
  # PML part of the Traction operator
  A = [P_vec_diag[1,1] P_vec_diag[1,2]; P_vec_diag[2,1] P_vec_diag[2,2]]
  B = [P_vec_diag[3,3] P_vec_diag[3,4]; P_vec_diag[4,3] P_vec_diag[4,4]]  
  Tq₀ = SJq₀*[Z    (I(2)⊗𝐇q₀)*Zx     -(I(2)⊗𝐇q₀)*A     Z     Z]
  Tqₙ = SJqₙ*[Z     (I(2)⊗𝐇qₙ)*Zx     (I(2)⊗𝐇qₙ)*A     Z     Z]
  Tr₀ = SJr₀*[(I(2)⊗𝐇r₀)*σ*Zy    (I(2)⊗𝐇r₀)*Zy     Z     -(I(2)⊗𝐇r₀)*B     -(I(2)⊗𝐇r₀)*σ*Zy] 
  Trₙ = SJrₙ*[(I(2)⊗𝐇rₙ)*σ*Zy     (I(2)⊗𝐇rₙ)*Zy     Z     (I(2)⊗𝐇rₙ)*B     -(I(2)⊗𝐇rₙ)*σ*Zy] 
  Tq₀, Tqₙ, Tr₀, Trₙ
end

"""
Function to get the marker matrix for implementing the jump conditions on the interface.
The resulting matrix uses an interpolation operator used in SBP techniques.
"""
function get_marker_matrix(N_C)  
  C2F, F2C = INTERPOLATION_4(N_C)
  N_F = 2*N_C-1  
  I_N_C = spzeros(Float64, N_C, N_F)  
  I_N_C[1, N_F] = 1.0
  I_N_F = spzeros(Float64, N_F, N_C)  
  I_N_F[N_F, 1] = 1.0
  J_N_C = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for  q in LinRange(0,1,N_C)].^(0.5))
  J_N_F = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for  q in LinRange(0,1,N_F)].^(0.5))
  W₁ = I(2) ⊗ I(N_C) ⊗ E1(1, 1, N_C)
  W₂ = I(2) ⊗ I(N_F) ⊗ E1(N_F, N_F, N_F)
  Z₁ = I(2) ⊗ (J_N_C\(F2C*J_N_F)) ⊗ I_N_C
  Z₂ = I(2) ⊗ (J_N_F\(C2F*J_N_C)) ⊗ I_N_F 

  # Bulk zero matrices
  Z_2_10_N_C_N_C = spzeros(2*N_C^2, 10*N_C^2);
  Z_2_10_N_C_N_F = spzeros(2*N_C^2, 10*N_F^2);
  Z_2_10_N_F_N_F = spzeros(2*N_F^2, 10*N_F^2);
  Z_2_10_N_F_N_C = spzeros(2*N_F^2, 10*N_C^2);

  Z_2_8_N_C_N_C = spzeros(2*N_C^2, 8*N_C^2);
  Z_2_8_N_C_N_F = spzeros(2*N_C^2, 8*N_F^2);
  Z_2_8_N_F_N_C = spzeros(2*N_F^2, 8*N_C^2);
  Z_2_8_N_F_N_F = spzeros(2*N_F^2, 8*N_F^2);

  Z_6_10_N_C_N_C = spzeros(6*N_C^2, 10*N_C^2);
  Z_6_10_N_C_N_F = spzeros(6*N_C^2, 10*N_F^2);
  Z_6_10_N_F_N_C = spzeros(6*N_F^2, 10*N_C^2);
  Z_6_10_N_F_N_F = spzeros(6*N_F^2, 10*N_F^2);

  mk1 = [Z_2_10_N_C_N_C   Z_2_10_N_C_N_F; 
         [(-W₁) Z_2_8_N_C_N_C (Z₁) Z_2_8_N_C_N_F]; 
         Z_6_10_N_C_N_C   Z_6_10_N_C_N_F; 
         Z_2_10_N_F_N_C   Z_2_10_N_F_N_F; 
         [(-Z₂) Z_2_8_N_F_N_C (W₂) Z_2_8_N_F_N_F]; 
         Z_6_10_N_F_N_C   Z_6_10_N_F_N_F]
  
  mk2 = [Z_2_10_N_C_N_C   Z_2_10_N_C_N_F; 
         [(-W₁) Z_2_8_N_C_N_C (Z₁) Z_2_8_N_C_N_F]; 
         Z_6_10_N_C_N_C   Z_6_10_N_C_N_F; 
         Z_2_10_N_F_N_C   Z_2_10_N_F_N_F; 
         [(Z₂) Z_2_8_N_F_N_C (-W₂) Z_2_8_N_F_N_F]; 
         Z_6_10_N_F_N_C   Z_6_10_N_F_N_F]
  
  Z_8_10_N_C_N_C = spzeros(8*N_C^2, 10*N_C^2)
  Z_8_10_N_C_N_F = spzeros(8*N_C^2, 10*N_F^2)
  Z_8_10_N_F_N_C = spzeros(8*N_F^2, 10*N_C^2)
  Z_8_10_N_F_N_F = spzeros(8*N_F^2, 10*N_F^2)  

  mk3 = [[(-W₁)  Z_2_8_N_C_N_C   (Z₁)  Z_2_8_N_C_N_F];
         Z_8_10_N_C_N_C  Z_8_10_N_C_N_F;
         [(-Z₂)  Z_2_8_N_F_N_C   (W₂)  Z_2_8_N_F_N_F];
         Z_8_10_N_F_N_C  Z_8_10_N_F_N_F]

  mk1, mk2, mk3
end

function 𝐊2ᴾᴹᴸ_NC(𝐪𝐫₁, 𝐪𝐫₂)
  # Obtain the properties of the first layer
  detJ₁(x) = (det∘J)(x,Ω₁)  
  P₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫₁) # Elasticity Bulk (For traction)
  PML₁ =  P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ₁, Ω₁, 𝐪𝐫₁) # PML Bulk  
  # Obtain the properties of the second layer
  detJ₂(x) = (det∘J)(x,Ω₂)  
  P₂ = P2R.(𝒫², Ω₂, 𝐪𝐫₂) # Elasticity Bulk (For traction)
  PML₂ =  P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ₂, Ω₂, 𝐪𝐫₂) # PML Bulk  
  # Get the 2d operators
  m₁,n₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  Dq₁, Dr₁ = sbp_2d₁.D1
  m₂,n₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  Dq₂, Dr₂ = sbp_2d₂.D1
  # Jacobian
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫₁))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫₂))    
  # Bulk stiffness matrix components on Layer 1
  𝐏₁ = Pᴱ(Dᴱ(P₁))  
  𝐏ᴾᴹᴸ₁ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₁))  
  xy₁ = Ω₁.(𝐪𝐫₁)  
  σ₁ = I(2) ⊗ spdiagm(vec(σₚ.(xy₁)))  
  ρσ₁ = I(2) ⊗ spdiagm(vec(ρ¹.(xy₁).*σₚ.(xy₁)))
  ρσα₁ = α*ρσ₁
  Jinv_vec₁ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫₁, Ω₁))
  Jinv_vec_diag₁ = [spdiagm(vec(p)) for p in Jinv_vec₁] #[qx rx; qy ry]
  JD₁¹ = (I(2)⊗Jinv_vec_diag₁[1,1])*(I(2)⊗Dq₁) + (I(2)⊗Jinv_vec_diag₁[1,2])*(I(2)⊗Dr₁)
  JD₂¹ = (I(2)⊗Jinv_vec_diag₁[2,1])*(I(2)⊗Dq₁) + (I(2)⊗Jinv_vec_diag₁[2,2])*(I(2)⊗Dr₁)
  SJr₀¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m₁)])⊗E1(1,1,m₁), m₁)
  SJq₀¹ = get_surf_J(I(2)⊗E1(1,1,m₁)⊗spdiagm([(det(J([0.0,q], Ω₁))*J⁻¹s([0.0,q], Ω₁, [-1,0])) for q in LinRange(0,1,m₁)]), m₁)
  SJrₙ¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₁))*J⁻¹s([q,1.0], Ω₁, [0,1])) for q in LinRange(0,1,m₁)])⊗E1(m₁,m₁,m₁), m₁)
  SJqₙ¹ = get_surf_J(I(2)⊗E1(m₁,m₁,m₁)⊗spdiagm([(det(J([1.0,q], Ω₁))*J⁻¹s([1.0,q], Ω₁, [1,0])) for q in LinRange(0,1,m₁)]), m₁)
  # Bulk stiffness matrix components on Layer 2
  𝐏₂ = Pᴱ(Dᴱ(P₂))  
  𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₂))
  xy₂ = Ω₂.(𝐪𝐫₂)
  σ₂ = I(2) ⊗ spdiagm(vec(σₚ.(xy₂)))  
  ρσ₂ = I(2) ⊗ spdiagm(vec(ρ².(xy₂).*σₚ.(xy₂)))
  ρσα₂ = α*ρσ₂
  Jinv_vec₂ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫₂, Ω₂))
  Jinv_vec_diag₂ = [spdiagm(vec(p)) for p in Jinv_vec₂] #[qx rx; qy ry]
  JD₁² = (I(2)⊗Jinv_vec_diag₂[1,1])*(I(2)⊗Dq₂) + (I(2)⊗Jinv_vec_diag₂[1,2])*(I(2)⊗Dr₂) # x-Derivative operator in physical domain
  JD₂² = (I(2)⊗Jinv_vec_diag₂[2,1])*(I(2)⊗Dq₂) + (I(2)⊗Jinv_vec_diag₂[2,2])*(I(2)⊗Dr₂) # y-Derivative operator in physical domain
  SJr₀² = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₂))*J⁻¹s([q,0.0], Ω₂, [0,-1])) for q in LinRange(0,1,m₂)])⊗E1(1,1,m₂), m₂)
  SJq₀² = get_surf_J(I(2)⊗E1(1,1,m₂)⊗spdiagm([(det(J([0.0,q], Ω₂))*J⁻¹s([0.0,q], Ω₂, [-1,0])) for q in LinRange(0,1,m₂)]), m₂)
  SJrₙ² = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m₂)])⊗E1(m₂,m₂,m₂), m₂)
  SJqₙ² = get_surf_J(I(2)⊗E1(m₂,m₂,m₂)⊗spdiagm([(det(J([1.0,q], Ω₂))*J⁻¹s([1.0,q], Ω₂, [1,0])) for q in LinRange(0,1,m₂)]), m₂)
  # Identity matrices on the two grids
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Z₁ = zero(Id₁)  
  Z₂ = zero(Id₂)  
  # Assemble the bulk stiffness matrix
  Σ₁ = [   Z₁      Id₁       Z₁       Z₁       Z₁;
      (spdiagm(detJ1₁.^-1)*𝐏₁+ρσα₁)  -ρσ₁     (spdiagm(detJ1₁.^-1)*𝐏ᴾᴹᴸ₁)        -ρσα₁;
      JD₁¹     Z₁    -(α*Id₁+σ₁)   Z₁       Z₁;
      JD₂¹     Z₁       Z₁      -α*Id₁      Z₁;
      α*Id₁    Z₁       Z₁       Z₁       -α*Id₁]
  Σ₂ = [   Z₂      Id₂       Z₂       Z₂       Z₂;
      (spdiagm(detJ1₂.^-1)*𝐏₂+ρσα₂)  -ρσ₂     (spdiagm(detJ1₂.^-1)*𝐏ᴾᴹᴸ₂)        -ρσα₂;
      JD₁²    Z₂    -(α*Id₂+σ₂)   Z₂       Z₂;
      JD₂²    Z₂       Z₂      -α*Id₂      Z₂;
      α*Id₂    Z₂       Z₂         Z₂     -α*Id₂ ]
  Σ = blockdiag(Σ₁, Σ₂)  
  # Get the traction operator of the elasticity and PML parts on Layer 1
  𝐓q₀¹ = Tᴱ(P₁, Ω₁, [-1,0]).A
  𝐓r₀¹ = Tᴱ(P₁, Ω₁, [0,-1]).A
  𝐓qₙ¹ = Tᴱ(P₁, Ω₁, [1,0]).A 
  𝐓rₙ¹ = Tᴱ(P₁, Ω₁, [0,1]).A 
  Zx₁ = blockdiag(spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₁₁¹.(xy₁)))), spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₃₃¹.(xy₁)))))
  Zy₁ = blockdiag(spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₃₃¹.(xy₁)))), spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₂₂¹.(xy₁)))))  
  𝐓ᴾᴹᴸq₀¹, 𝐓ᴾᴹᴸqₙ¹, _, 𝐓ᴾᴹᴸrₙ¹  = Tᴾᴹᴸ(PML₁, (Zx₁, Zy₁), σₚ, Ω₁, 𝐪𝐫₁)
  # Get the traction operator of the elasticity and PML parts on Layer 2
  𝐓q₀² = Tᴱ(P₂, Ω₂, [-1,0]).A
  𝐓r₀² = Tᴱ(P₂, Ω₂, [0,-1]).A
  𝐓qₙ² = Tᴱ(P₂, Ω₂, [1,0]).A 
  𝐓rₙ² = Tᴱ(P₂, Ω₂, [0,1]).A 
  Zx₂ = blockdiag(spdiagm(vec(sqrt.(ρ².(xy₂).*c₁₁².(xy₂)))), spdiagm(vec(sqrt.(ρ².(xy₂).*c₃₃².(xy₂)))))
  Zy₂ = blockdiag(spdiagm(vec(sqrt.(ρ².(xy₂).*c₃₃².(xy₂)))), spdiagm(vec(sqrt.(ρ².(xy₂).*c₂₂².(xy₂)))))  
  𝐓ᴾᴹᴸq₀², 𝐓ᴾᴹᴸqₙ², 𝐓ᴾᴹᴸr₀², _  = Tᴾᴹᴸ(PML₂, (Zx₂, Zy₂), σₚ, Ω₂, 𝐪𝐫₂)
  # Norm matrices
  𝐇q₀¹, 𝐇qₙ¹, _, 𝐇rₙ¹ = sbp_2d₁.norm  
  𝐇q₀², 𝐇qₙ², 𝐇r₀², _ = sbp_2d₂.norm  
  # Get the overall traction operator on the outer boundaries of both Layer 1 and Layer 2
   # Get the overall traction operator on the outer boundaries
  # Layer 1
  𝐓𝐪₀¹ = spdiagm(detJ1₁.^-1)*([-(I(2)⊗𝐇q₀¹)*SJq₀¹*𝐓q₀¹   Z₁    Z₁   Z₁   Z₁] + SJq₀¹*𝐓ᴾᴹᴸq₀¹)
  𝐓𝐪ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇qₙ¹)*SJqₙ¹*𝐓qₙ¹  Z₁   Z₁    Z₁   Z₁] + SJqₙ¹*𝐓ᴾᴹᴸqₙ¹)
  𝐓𝐫ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇rₙ¹)*SJrₙ¹*𝐓rₙ¹  Z₁   Z₁   Z₁   Z₁] + SJrₙ¹*𝐓ᴾᴹᴸrₙ¹)
  # Layer 2
  𝐓𝐪₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇q₀²)*SJq₀²*𝐓q₀²   Z₂    Z₂   Z₂   Z₂] + SJq₀²*𝐓ᴾᴹᴸq₀²)
  𝐓𝐪ₙ² = spdiagm(detJ1₂.^-1)*([(I(2)⊗𝐇qₙ²)*SJqₙ²*𝐓qₙ²  Z₂   Z₂   Z₂   Z₂] + SJqₙ²*𝐓ᴾᴹᴸqₙ²)
  𝐓𝐫₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇r₀²)*SJr₀²*𝐓r₀²  Z₂  Z₂   Z₂   Z₂] + SJr₀²*𝐓ᴾᴹᴸr₀²)
  # Interface conditions: 
  zbT₁ = spzeros(Float64, 2m₁^2, 10n₁^2)
  zbB₁ = spzeros(Float64, 6m₁^2, 10n₁^2)
  zbT₂ = spzeros(Float64, 2m₂^2, 10n₂^2)
  zbB₂ = spzeros(Float64, 6m₂^2, 10n₂^2)
  P_vec₁ = get_property_matrix_on_grid(PML₁)
  P_vec₂ = get_property_matrix_on_grid(PML₂)
  P_vec_diag₁ = [spdiagm(vec(p)) for p in P_vec₁]  
  P_vec_diag₂ = [spdiagm(vec(p)) for p in P_vec₂]
  B₁ = SJr₀¹\([P_vec_diag₁[3,3] P_vec_diag₁[3,4]; P_vec_diag₁[4,3] P_vec_diag₁[4,4]])
  B₂ = SJrₙ²\([P_vec_diag₂[3,3] P_vec_diag₂[3,4]; P_vec_diag₂[4,3] P_vec_diag₂[4,4]])
  𝐓𝐫₁ = spdiagm(detJ1₁.^-1)*[(𝐓r₀¹)   Z₁     Z₁    (B₁)     Z₁]  
  𝐓𝐫₂ = spdiagm(detJ1₂.^-1)*[(𝐓rₙ²)    Z₂     Z₂    (B₂)     Z₂]   
  𝐓𝐫 = blockdiag([𝐓𝐫₁; zbT₁; zbB₁], [𝐓𝐫₂; zbT₂; zbB₂])
  # Transpose matrix
  𝐓𝐫₁ᵀ = spdiagm(detJ1₁.^-1)*[(𝐓r₀¹)'   Z₁     Z₁    (B₁)'   Z₁]  
  𝐓𝐫₂ᵀ = spdiagm(detJ1₂.^-1)*[(𝐓rₙ²)'    Z₂     Z₂    (B₂)'   Z₂]  
  𝐓𝐫ᵀ = blockdiag([zbT₁;  𝐓𝐫₁ᵀ; zbB₁], [zbT₂;  𝐓𝐫₂ᵀ; zbB₂])
  ##### Get the Jump matrices #####
  BH, BT, BHᵀ = get_marker_matrix(m₁);
  #################################
  Hq₁⁻¹ = (sbp_q₁.norm\I(m₁)) |> sparse
  Hr₁⁻¹ = (sbp_r₁.norm\I(n₁)) |> sparse
  Hq₂⁻¹ = (sbp_q₂.norm\I(m₂)) |> sparse
  Hr₂⁻¹ = (sbp_r₂.norm\I(n₂)) |> sparse
  # Hq = sbp_q.norm
  Hr₁ = sbp_q₁.norm
  Hr₂ = sbp_q₂.norm
  SJ₁ = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m₁)])
  SJ₂ = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m₂)])
  𝐃₁⁻¹ = blockdiag((I(10)⊗Hq₁⁻¹⊗Hr₁⁻¹), 
                   (I(10)⊗Hq₂⁻¹⊗Hr₂⁻¹))
  𝐃 = blockdiag((I(10)⊗(SJ₁*Hr₁)⊗I(m₁))*(I(10)⊗I(m₁)⊗ E1(1,1,m₁)), 
                (I(10)⊗(SJ₂*Hr₂)⊗I(m₂))*(I(10)⊗I(m₂)⊗ E1(m₂,m₂,m₂)))
  𝐃₂ = blockdiag((I(2)⊗(SJ₁*Hr₁)⊗I(m₁))*(I(2)⊗I(m₁)⊗ E1(1,1,m₁)), Z₁, Z₁, (I(2)⊗(SJ₁*Hr₁)⊗I(m₁))*(I(2)⊗I(m₁)⊗ E1(1,1,m₁)), Z₁, 
                 (I(2)⊗(SJ₂*Hr₂)⊗I(m₂))*(I(2)⊗I(m₂)⊗ E1(m₂,m₂,m₂)), Z₂, Z₂, (I(2)⊗(SJ₂*Hr₂)⊗I(m₂))*(I(2)⊗I(m₂)⊗ E1(m₂,m₂,m₂)), Z₂)
  ζ₀ = 200/h₂
  𝚯 = 𝐃₁⁻¹*𝐃*BH*𝐓𝐫
  𝚯ᵀ = -𝐃₁⁻¹*𝐓𝐫ᵀ*𝐃₂*BHᵀ
  Ju = -𝐃₁⁻¹*𝐃*BT
  𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju
  𝐓ₙ = blockdiag([zbT₁;   𝐓𝐪₀¹ + 𝐓𝐪ₙ¹ + 𝐓𝐫ₙ¹;   zbB₁], [zbT₂;   𝐓𝐪₀² + 𝐓𝐪ₙ² + 𝐓𝐫₀²;   zbB₂])      
  Σ - 𝐓ₙ - 𝐓ᵢ
end

function 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫₁, 𝐪𝐫₂)
  m₁, n₁ = size(𝐪𝐫₁)
  m₂, n₂ = size(𝐪𝐫₂)
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ¹.(Ω₁.(𝐪𝐫₁))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ².(Ω₂.(𝐪𝐫₂))))
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁), blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂))
end 

#### #### #### #### #### 
# Begin time stepping  #
#### #### #### #### ####
"""
A non-allocating implementation of the RK4 scheme
"""
function RK4_1!(M, sol)  
  X₀, k₁, k₂, k₃, k₄ = sol
  # k1 step  
  mul!(k₁, M, X₀);
  # k2 step
  mul!(k₂, M, k₁, 0.5*Δt, 0.0); mul!(k₂, M, X₀, 1, 1);
  # k3 step
  mul!(k₃, M, k₂, 0.5*Δt, 0.0); mul!(k₃, M, X₀, 1, 1);
  # k4 step
  mul!(k₄, M, k₃, Δt, 0.0); mul!(k₄, M, X₀, 1, 1);
  # Final step
  @turbo for i=1:lastindex(X₀)
    X₀[i] = X₀[i] + (Δt/6)*(k₁[i] + k₂[i] + k₃[i] + k₄[i])
  end
  X₀
end

"""
Initial conditions (Layer 1)
"""
𝐔₁(x) = @SVector [exp(-((x[1]-2.2π)^2 + (x[2]+2π)^2)), -exp(-16π*((x[1]-2.2π)^2 + (x[2]+2π)^2))]
𝐑₁(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₁(x) = @SVector [0.0, 0.0]
𝐖₁(x) = @SVector [0.0, 0.0]
𝐐₁(x) = @SVector [0.0, 0.0]

"""
Initial conditions (Layer 2)
"""
𝐔₂(x) = @SVector [exp(-((x[1]-2.2π)^2 + (x[2]+2π)^2)), -exp(-16π*((x[1]-2.2π)^2 + (x[2]+2π)^2))]
𝐑₂(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₂(x) = @SVector [0.0, 0.0]
𝐖₂(x) = @SVector [0.0, 0.0]
𝐐₂(x) = @SVector [0.0, 0.0]

"""
Function to split the solution into the corresponding variables
"""
function split_solution(X, N)  
  u1,u2 = @views X[1:N^2], @views X[N^2+1:2N^2];
  r1,r2 = @views X[2N^2+1:3N^2], @views X[3N^2+1:4N^2];
  v1,v2 = @views X[4N^2+1:5N^2], @views X[5N^2+1:6N^2];
  w1,w2 = @views X[6N^2+1:7N^2], @views X[7N^2+1:8N^2];
  q1,q2 = @views X[8N^2+1:9N^2], @views X[9N^2+1:10N^2];
  (u1,u2), (r1,r2), (v1, v2), (w1,w2), (q1,q2)
end

#############################
# Obtain Reference Solution #
#############################
N₁ = 51
N₂ = 2*N₁ - 1
𝐪𝐫₁ = generate_2d_grid((N₁, N₁));
𝐪𝐫₂ = generate_2d_grid((N₂, N₂));
xy₁ = vec(Ω₁.(𝐪𝐫₁));
xy₂ = vec(Ω₂.(𝐪𝐫₂));
const h₁ = Lₓ/(N₁-1);
const h₂ = Lₓ/(N₂-1);
stima = 𝐊2ᴾᴹᴸ_NC(𝐪𝐫₁, 𝐪𝐫₂);
massma = 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫₁, 𝐪𝐫₂);

cmax = 45.57
τ₀ = 1/2
const Δt = 0.2/(cmax*τ₀)*h₂
tf = 40.0
ntime = ceil(Int, tf/Δt)
solmax = zeros(Float64, ntime)

M = massma*stima
iter = 0
let  
  t = iter*tf
  X₀¹ = vcat(eltocols(vec(𝐔₁.(xy₁))), eltocols(vec(𝐑₁.(xy₁))), eltocols(vec(𝐕₁.(xy₁))), eltocols(vec(𝐖₁.(xy₁))), eltocols(vec(𝐐₁.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔₂.(xy₂))), eltocols(vec(𝐑₂.(xy₂))), eltocols(vec(𝐕₂.(xy₂))), eltocols(vec(𝐖₂.(xy₂))), eltocols(vec(𝐐₂.(xy₂))));
  X₀ = vcat(X₀¹, X₀²)  
  # X₀ = X₁
  # Arrays to store the RK-variables
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀))
  
  @gif for i=1:ntime
  # for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    X₀ = RK4_1!(M,sol)    
    t += Δt    
    solmax[i] = maximum(abs.(X₀))
    (i%1000==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(solmax[i]))    
    
    ## Plotting to get GIFs
    u1₁,u2₁ = split_solution(view(X₀, 1:10*N₁^2), N₁)[1];
    u1₂,u2₂ = split_solution(view(X₀, 10*N₁^2+1:10*N₁^2+10*N₂^2), N₂)[1];              
    plt1₁ = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");    
    scatter!(plt1₁, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt1₁, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],N₁)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1);
    scatter!(plt1₁, Tuple.([cᵢ(q) for q in LinRange(0,1,N₁)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))    
    title!(plt1₁, "Time t="*string(round(t,digits=4)))
    plt1₁
  # end
  end every 100
  global X₁ = X₀  
end

u1₁,u2₁ = split_solution(view(X₁, 1:10*N₁^2), N₁)[1];
u1₂,u2₂ = split_solution(view(X₁, 10*N₁^2+1:10*N₁^2+10*N₂^2), N₂)[1];

plt1 = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y", xlabel="x", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],N₁)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt1, Tuple.([cᵢ(q) for q in LinRange(0,1,N₁)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt1, "Horizontal Displacement")
plt2 = scatter(Tuple.(xy₁), zcolor=vec(u2₁), colormap=:turbo, ylabel="y", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.(xy₂), zcolor=vec(u2₂), colormap=:turbo, ylabel="y", xlabel="x", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],N₂)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt2, Tuple.([cᵢ(q) for q in LinRange(0,1,N₁)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt2, "Vertical Displacement")
plt3 = scatter(Tuple.(xy₁), zcolor=vec(σₚ.(xy₁)), colormap=:turbo, markersize=4, msw=0.01, label="", ylabel="y", xlabel="x");
scatter!(plt3, Tuple.(xy₂), zcolor=vec(σₚ.(xy₂)), colormap=:turbo, markersize=4, msw=0.01, label="", ylabel="y", xlabel="x");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],N₂)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:red, markersize=2, msw=0.1, colorbar_exponentformat="power");
scatter!(plt3, Tuple.([cᵢ(q) for q in LinRange(0,1,N₂)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800), right_margin=20*Plots.mm);
title!(plt3, "PML Function")
plt4 = plot()
plot!(plt4, LinRange(iter*tf,(iter+1)*tf,ntime), solmax, yaxis=:log10, label="||U||₍∞₎", lw=2, size=(800,800))
xlabel!(plt4, "Time (t)")
plt5 = plot(plt1, plt3, plt2, plt4, layout=(2,2));