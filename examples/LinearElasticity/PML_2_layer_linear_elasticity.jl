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
pf = 8
f(q) = 0.3*exp(-4*4.4π*(q-0.55)^2)
cᵢ(q) = [1.1*q, f(q)];
# Define the rest of the boundary
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [1.1, r]; # Right boundary
c₃¹(q) = [1.1*q, 1.0]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, r - 1.0]; # Left boundary
c₁²(q) = [1.1*q, -1.0]; # Bottom boundary. 
c₂²(r) = [1.1, r - 1.0]; # Right boundary
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
const δ = 0.1
const Lₓ = 1.0
const σ₀ = 0.4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
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
  σ = I(2) ⊗ (spdiagm(vec(σₚ.(𝐱𝐲))))
  SJr₀ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω))*J⁻¹s([q,0.0], Ω, [0,-1])) for q in LinRange(0,1,m)].^-1)⊗E1(1,1,m), m)
  SJq₀ = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω))*J⁻¹s([0.0,q], Ω, [-1,0])) for q in LinRange(0,1,m)].^-1), m)
  SJrₙ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω))*J⁻¹s([q,1.0], Ω, [0,1])) for q in LinRange(0,1,m)].^-1)⊗E1(m,m,m), m)
  SJqₙ = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω))*J⁻¹s([1.0,q], Ω, [1,0])) for q in LinRange(0,1,m)].^-1), m)
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
Redefine the marker matrix for the PML
"""
function get_marker_matrix(m)  
  W₁ = I(2) ⊗ I(m) ⊗ E1(1,1,m)
  W₂ = I(2) ⊗ I(m) ⊗ E1(m,m,m)
  Z₁ = I(2) ⊗ I(m) ⊗ E1(1,m,m)  
  Z₂ = I(2) ⊗ I(m) ⊗ E1(m,1,m) 
  # Bulk zero matrices
  Z_2_20 = spzeros(2m^2, 20m^2);
  Z_2_8 = spzeros(2m^2, 8m^2);
  Z_6_20 = spzeros(6m^2, 20m^2);

  mk1 = [Z_2_20; 
         [(-W₁) Z_2_8 (Z₁) Z_2_8]; 
         Z_6_20; 
         Z_2_20; 
         [(-Z₂) Z_2_8 (W₂) Z_2_8]; 
         Z_6_20]
  mk2 = [Z_2_20; 
         [(-W₁) Z_2_8 (Z₁) Z_2_8]; 
         Z_6_20; 
         Z_2_20; 
         [(Z₂) Z_2_8 (-W₂) Z_2_8]; 
         Z_6_20]
  
  Z_8_20 = spzeros(8m^2, 20m^2)
  mk3 = [[(-W₁)  Z_2_8   (Z₁)  Z_2_8];
         Z_8_20;
         [(-Z₂)  Z_2_8   (W₂)  Z_2_8];
         Z_8_20]

  mk1, mk2, mk3
end

function 𝐊2ᴾᴹᴸ(𝐪𝐫)
  # Obtain the properties of the first layer
  detJ₁(x) = (det∘J)(x,Ω₁)  
  P₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫) # Elasticity Bulk (For traction)
  PML₁ =  P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ₁, Ω₁, 𝐪𝐫) # PML Bulk  
  # Obtain the properties of the second layer
  detJ₂(x) = (det∘J)(x,Ω₂)  
  P₂ = P2R.(𝒫², Ω₂, 𝐪𝐫) # Elasticity Bulk (For traction)
  PML₂ =  P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ₂, Ω₂, 𝐪𝐫) # PML Bulk  
  # Get the 2d operators
  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1
  # Jacobian
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫))    
  # Bulk stiffness matrix components on Layer 1
  𝐏₁ = Pᴱ(Dᴱ(P₁))  
  𝐏ᴾᴹᴸ₁ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₁))  
  xy₁ = Ω₁.(𝐪𝐫)  
  σ₁ = I(2) ⊗ spdiagm(vec(σₚ.(xy₁)))  
  ρσ₁ = I(2) ⊗ spdiagm(vec(ρ¹.(xy₁).*σₚ.(xy₁)))
  ρσα₁ = α*ρσ₁
  Jinv_vec₁ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω₁))
  Jinv_vec_diag₁ = [spdiagm(vec(p)) for p in Jinv_vec₁] #[qx rx; qy ry]
  JD₁¹ = (I(2)⊗Jinv_vec_diag₁[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[1,2])*(I(2)⊗Dr)
  JD₂¹ = (I(2)⊗Jinv_vec_diag₁[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[2,2])*(I(2)⊗Dr)
  SJr₀¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m), m)
  SJq₀¹ = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω₁))*J⁻¹s([0.0,q], Ω₁, [-1,0])) for q in LinRange(0,1,m)]), m)
  SJrₙ¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₁))*J⁻¹s([q,1.0], Ω₁, [0,1])) for q in LinRange(0,1,m)])⊗E1(m,m,m), m)
  SJqₙ¹ = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω₁))*J⁻¹s([1.0,q], Ω₁, [1,0])) for q in LinRange(0,1,m)]), m)
  # Bulk stiffness matrix components on Layer 2
  𝐏₂ = Pᴱ(Dᴱ(P₂))  
  𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₂))
  xy₂ = Ω₂.(𝐪𝐫)
  σ₂ = I(2) ⊗ spdiagm(vec(σₚ.(xy₂)))  
  ρσ₂ = I(2) ⊗ spdiagm(vec(ρ².(xy₂).*σₚ.(xy₂)))
  ρσα₂ = α*ρσ₂
  Jinv_vec₂ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω₂))
  Jinv_vec_diag₂ = [spdiagm(vec(p)) for p in Jinv_vec₂] #[qx rx; qy ry]
  JD₁² = (I(2)⊗Jinv_vec_diag₂[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₂[1,2])*(I(2)⊗Dr) # x-Derivative operator in physical domain
  JD₂² = (I(2)⊗Jinv_vec_diag₂[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₂[2,2])*(I(2)⊗Dr) # y-Derivative operator in physical domain
  SJr₀² = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₂))*J⁻¹s([q,0.0], Ω₂, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m), m)
  SJq₀² = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω₂))*J⁻¹s([0.0,q], Ω₂, [-1,0])) for q in LinRange(0,1,m)]), m)
  SJrₙ² = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m)])⊗E1(m,m,m), m)
  SJqₙ² = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω₂))*J⁻¹s([1.0,q], Ω₂, [1,0])) for q in LinRange(0,1,m)]), m)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Z = zero(Id)  
  # Assemble the bulk stiffness matrix
  Σ₁ = [   Z      Id       Z       Z       Z;
      (spdiagm(detJ1₁.^-1)*𝐏₁+ρσα₁)  -ρσ₁     (spdiagm(detJ1₁.^-1)*𝐏ᴾᴹᴸ₁)        -ρσα₁;
      JD₁¹    Z    -(α*Id+σ₁)   Z       Z;
      JD₂¹    Z       Z      -α*Id    Z;
      α*Id    Z       Z       Z     -α*Id ]
  Σ₂ = [   Z      Id       Z       Z       Z;
      (spdiagm(detJ1₂.^-1)*𝐏₂+ρσα₂)  -ρσ₂     (spdiagm(detJ1₂.^-1)*𝐏ᴾᴹᴸ₂)        -ρσα₂;
      JD₁²    Z    -(α*Id+σ₂)   Z       Z;
      JD₂²    Z       Z      -α*Id    Z;
      α*Id   Z       Z       Z     -α*Id ]
  Σ = blockdiag(Σ₁, Σ₂)  
  # Get the traction operator of the elasticity and PML parts on Layer 1
  𝐓q₀¹ = Tᴱ(P₁, Ω₁, [-1,0]).A
  𝐓r₀¹ = Tᴱ(P₁, Ω₁, [0,-1]).A
  𝐓qₙ¹ = Tᴱ(P₁, Ω₁, [1,0]).A 
  𝐓rₙ¹ = Tᴱ(P₁, Ω₁, [0,1]).A 
  Zx₁ = blockdiag(spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₁₁¹.(xy₁)))), spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₃₃¹.(xy₁)))))
  Zy₁ = blockdiag(spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₃₃¹.(xy₁)))), spdiagm(vec(sqrt.(ρ¹.(xy₁).*c₂₂¹.(xy₁)))))  
  𝐓ᴾᴹᴸq₀¹, 𝐓ᴾᴹᴸqₙ¹, _, 𝐓ᴾᴹᴸrₙ¹  = Tᴾᴹᴸ(PML₁, (Zx₁, Zy₁), σₚ, Ω₁, 𝐪𝐫)
  # Get the traction operator of the elasticity and PML parts on Layer 2
  𝐓q₀² = Tᴱ(P₂, Ω₂, [-1,0]).A
  𝐓r₀² = Tᴱ(P₂, Ω₂, [0,-1]).A
  𝐓qₙ² = Tᴱ(P₂, Ω₂, [1,0]).A 
  𝐓rₙ² = Tᴱ(P₂, Ω₂, [0,1]).A 
  Zx₂ = blockdiag(spdiagm(vec(sqrt.(ρ².(xy₂).*c₁₁².(xy₂)))), spdiagm(vec(sqrt.(ρ².(xy₂).*c₃₃².(xy₂)))))
  Zy₂ = blockdiag(spdiagm(vec(sqrt.(ρ².(xy₂).*c₃₃².(xy₂)))), spdiagm(vec(sqrt.(ρ².(xy₂).*c₂₂².(xy₂)))))  
  𝐓ᴾᴹᴸq₀², 𝐓ᴾᴹᴸqₙ², 𝐓ᴾᴹᴸr₀², _  = Tᴾᴹᴸ(PML₂, (Zx₂, Zy₂), σₚ, Ω₂, 𝐪𝐫)
  # Norm matrices
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm  
  # Get the overall traction operator on the outer boundaries
  # Layer 1
  𝐓𝐪₀¹ = spdiagm(detJ1₁.^-1)*([-(I(2)⊗𝐇q₀)*SJq₀¹*𝐓q₀¹   Z    Z   Z   Z] + SJq₀¹*𝐓ᴾᴹᴸq₀¹)
  𝐓𝐪ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇qₙ)*SJqₙ¹*𝐓qₙ¹  Z   Z    Z   Z] + SJqₙ¹*𝐓ᴾᴹᴸqₙ¹)
  𝐓𝐫ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇rₙ)*SJrₙ¹*𝐓rₙ¹  Z  Z   Z   Z] + SJrₙ¹*𝐓ᴾᴹᴸrₙ¹)
  # Layer 2
  𝐓𝐪₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇q₀)*SJq₀²*𝐓q₀²   Z    Z   Z   Z] + SJq₀²*𝐓ᴾᴹᴸq₀²)
  𝐓𝐪ₙ² = spdiagm(detJ1₂.^-1)*([(I(2)⊗𝐇qₙ)*SJqₙ²*𝐓qₙ²  Z   Z    Z   Z] + SJqₙ²*𝐓ᴾᴹᴸqₙ²)
  𝐓𝐫₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇r₀)*SJr₀²*𝐓r₀²  Z  Z   Z   Z] + SJr₀²*𝐓ᴾᴹᴸr₀²)

  # Interface conditions: 
  zbT = spzeros(Float64, 2m^2, 10n^2)
  zbB = spzeros(Float64, 6m^2, 10n^2)
  P_vec₁ = get_property_matrix_on_grid(PML₁)
  P_vec₂ = get_property_matrix_on_grid(PML₂)
  P_vec_diag₁ = [spdiagm(vec(p)) for p in P_vec₁]  
  P_vec_diag₂ = [spdiagm(vec(p)) for p in P_vec₂]
  B₁ = SJr₀¹\([P_vec_diag₁[3,3] P_vec_diag₁[3,4]; P_vec_diag₁[4,3] P_vec_diag₁[4,4]])
  B₂ = SJrₙ²\([P_vec_diag₂[3,3] P_vec_diag₂[3,4]; P_vec_diag₂[4,3] P_vec_diag₂[4,4]] )
  𝐓𝐫₁ = spdiagm(detJ1₁.^-1)*[(𝐓r₀¹)   Z     Z    (B₁)     Z]  
  𝐓𝐫₂ = spdiagm(detJ1₂.^-1)*[(𝐓rₙ²)   Z     Z     (B₂)     Z]    
  𝐓𝐫 = blockdiag([𝐓𝐫₁; zbT; zbB], [𝐓𝐫₂; zbT; zbB])
  # Transpose matrix
  𝐓𝐫₁ᵀ = spdiagm(detJ1₁.^-1)*[(𝐓r₀¹)'   Z     Z    (B₁)'   Z]  
  𝐓𝐫₂ᵀ = spdiagm(detJ1₂.^-1)*[(𝐓rₙ²)'   Z     Z    (B₂)'   Z]  
  𝐓𝐫ᵀ = blockdiag([zbT;  𝐓𝐫₁ᵀ; zbB], [zbT;  𝐓𝐫₂ᵀ; zbB])
  BH, BT, BHᵀ = get_marker_matrix(m);
  Hq = sbp_q.norm
  Hr = sbp_r.norm  
  Hq⁻¹ = (Hq\I(m)) |> sparse
  Hr⁻¹ = (Hr\I(m)) |> sparse
  𝐃₁⁻¹ = blockdiag((I(10)⊗Hq⁻¹⊗Hr⁻¹), (I(10)⊗Hq⁻¹⊗Hr⁻¹))  
  SJ₁ = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m)])
  SJ₂ = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m)])
  𝐃 = blockdiag((I(10)⊗(SJ₁*Hr)⊗I(m))*(I(10)⊗I(m)⊗ E1(1,1,m)), (I(10)⊗(SJ₂*Hr)⊗I(m))*(I(10)⊗I(m)⊗ E1(m,m,m)))
  𝐃₂ = blockdiag((I(2)⊗(SJ₁*Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(1,1,m)), Z, Z, (I(2)⊗(SJ₁*Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(1,1,m)), Z, 
                 (I(2)⊗(SJ₂*Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(m,m,m)), Z, Z, (I(2)⊗(SJ₂*Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(m,m,m)), Z)
  ζ₀ = 800/h
  𝚯 = 𝐃₁⁻¹*𝐃*BH*𝐓𝐫
  𝚯ᵀ = -𝐃₁⁻¹*𝐓𝐫ᵀ*𝐃₂*BHᵀ
  Ju = -𝐃₁⁻¹*𝐃*BT
  𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju
  𝐓ₙ = blockdiag([zbT;   𝐓𝐪₀¹ + 𝐓𝐪ₙ¹ + 𝐓𝐫ₙ¹;   zbB], [zbT;   𝐓𝐪₀² + 𝐓𝐪ₙ² + 𝐓𝐫₀²;   zbB])      
  Σ - 𝐓ₙ - 𝐓ᵢ
end

function 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ¹.(Ω₁.(𝐪𝐫))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ².(Ω₂.(𝐪𝐫))))
  blockdiag(blockdiag(Id, ρᵥ¹, Id, Id, Id), blockdiag(Id, ρᵥ², Id, Id, Id))
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
𝐔₁(x) = @SVector [exp(-20*((x[1]-0.55)^2 + (x[2]-0.5)^2)), -exp(-20*((x[1]-0.55)^2 + (x[2]-0.5)^2))]
𝐑₁(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₁(x) = @SVector [0.0, 0.0]
𝐖₁(x) = @SVector [0.0, 0.0]
𝐐₁(x) = @SVector [0.0, 0.0]

"""
Initial conditions (Layer 2)
"""
𝐔₂(x) = @SVector [exp(-20*((x[1]-0.55)^2 + (x[2]-0.5)^2)), -exp(-20*((x[1]-0.55)^2 + (x[2]-0.5)^2))]
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
𝐍 = 41
𝐪𝐫 = generate_2d_grid((𝐍, 𝐍));
xy₁ = vec(Ω₁.(𝐪𝐫));
xy₂ = vec(Ω₂.(𝐪𝐫));
const h = Lₓ/(𝐍-1);
stima = 𝐊2ᴾᴹᴸ(𝐪𝐫);
massma = 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫);

cmax = 45.57
τ₀ = 1/4
const Δt = 0.2/(cmax*τ₀)*h
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
  
  # @gif for i=1:ntime
  for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    X₀ = RK4_1!(M,sol)    
    t += Δt    
    solmax[i] = maximum(abs.(X₀))
    (i%1000==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(solmax[i]))    
    
    ## Plotting to get GIFs
    #= u1₁,u2₁ = split_solution(view(X₀, 1:10*𝐍^2), 𝐍)[1];
    u1₂,u2₂ = split_solution(view(X₀, 10*𝐍^2+1:20*𝐍^2), 𝐍)[1];              
    plt1₁ = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");    
    scatter!(plt1₁, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt1₁, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1);
    scatter!(plt1₁, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))    
    title!(plt1₁, "Time t="*string(round(t,digits=4)))
    plt1₂ = scatter(Tuple.(xy₁), zcolor=σₚ.(vec(xy₁)), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="")
    scatter!(plt1₂, Tuple.(xy₂), zcolor=σₚ.(vec(xy₂)), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="")
    scatter!(plt1₂, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1);
    scatter!(plt1₂, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))    
    plt1 = plot(plt1₁, plt1₂, layout=(1,2)) =#
  end
  # end every 200
  global X₁ = X₀  
end 

u1₁,u2₁ = split_solution(view(X₁, 1:10*𝐍^2), 𝐍)[1];
u1₂,u2₂ = split_solution(view(X₁, 10*𝐍^2+1:20*𝐍^2), 𝐍)[1];

plt1 = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y", xlabel="x", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt1, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt1, "Horizontal Displacement")
plt2 = scatter(Tuple.(xy₁), zcolor=vec(u2₁), colormap=:turbo, ylabel="y", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.(xy₂), zcolor=vec(u2₂), colormap=:turbo, ylabel="y", xlabel="x", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt2, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt2, "Vertical Displacement")
plt3 = scatter(Tuple.(xy₁), zcolor=vec(σₚ.(xy₁)), colormap=:turbo, markersize=4, msw=0.01, label="", ylabel="y", xlabel="x");
scatter!(plt3, Tuple.(xy₂), zcolor=vec(σₚ.(xy₂)), colormap=:turbo, markersize=4, msw=0.01, label="", ylabel="y", xlabel="x");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:red, markersize=2, msw=0.1, colorbar_exponentformat="power");
scatter!(plt3, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800), right_margin=20*Plots.mm);
title!(plt3, "PML Function")
plt4 = plot()
plot!(plt4, LinRange(iter*tf,(iter+1)*tf,ntime), solmax, yaxis=:log10, label="||U||₍∞₎", lw=2, size=(800,800))
xlabel!(plt4, "Time (t)")
plt5 = plot(plt1, plt3, plt2, plt4, layout=(2,2));
# savefig(plt4, "./Images/PML/2-layer/stab.png"); 

#= 
# Use this code to remove any repetition in time-axis while plotting
plt7 = plot();
X = (plt4.series_list[1].plotattributes[:x], plt4.series_list[2].plotattributes[:x], 
     plt4.series_list[3].plotattributes[:x], plt4.series_list[5].plotattributes[:x], 
     plt4.series_list[6].plotattributes[:x])
Y = (plt4.series_list[1].plotattributes[:y], plt4.series_list[2].plotattributes[:y], 
     plt4.series_list[3].plotattributes[:y], plt4.series_list[5].plotattributes[:y],
     plt4.series_list[6].plotattributes[:y])
for i=1:lastindex(X)
  plot!(plt7, X[i], Y[i], yaxis=:log10, lw=2, size=(800,800), label="")
end
xlabel!(plt7, "Time (t)")
ylabel!(plt7, "||U||₍∞₎")
savefig(plt7, "./Images/PML/2-layer/stab.png");  
=#