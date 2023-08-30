##################################################
# Program to solve the 2 layer linear elasticity #
# Incluing the Perfectly Matched Layer Boundary  #
##################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [1,2]
# Define the parametrization for interface
f(q) = 0.12*exp(-40*(q-0.5)^2)
cᵢ(q) = 2*[q, 1.0 + f(q)];
# Define the rest of the boundary
c₀¹(r) = 2*[0.0 + 0*f(r), r+1]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = 2*[1.0 + 0*f(r), r+1]; # Right boundary
c₃¹(q) = 2*[q, 2.0 + 0*f(q)]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = 2*[0.0 + 0*f(r), r]; # Left boundary
c₁²(q) = 2*[q, 0.0 + 0*f(q)]; # Bottom boundary. 
c₂²(r) = 2*[1.0 + 0*f(r), r]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₁(qr) = S(qr, domain₁)
Ω₂(qr) = S(qr, domain₂)

##############################################
# We use the same properties for both layers #
##############################################
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
The PML damping
"""
const Lₓ = 1.6
const δ = 0.1*Lₓ
const σ₀ = 0*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀*0.05; # The frequency shift parameter

function σₚ(x)
  if((x[1] ≈ Lₓ) || x[1] > Lₓ)
    return σ₀*((x[1] - Lₓ)/δ)^3  
  else
    return 0.0
  end
end

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
Pqr::Matrix{SMatrix{m,n,Float64}} = [P(x₁₁) P(x₁₂) ... P(x₁ₙ)
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
  # v, w are included in the construction
  [D[1,1] D[1,2] D[1,3] D[1,4]; 
  D[2,1] D[2,2] D[2,3] D[2,4]] + 
  [D[3,1] D[3,2] D[3,3] D[3,4]; 
  D[4,1] D[4,2] D[4,3] D[4,4]]
end

"""
Function to obtain the PML contribution to the traction on the boundary
"""
function Tᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}}, Ω, 𝐪𝐫)
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
  𝐱𝐲 = Ω.(𝐪𝐫)
  
  # Evaluate the functions on the physical grid
  Zx = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₁₁.(𝐱𝐲)))), spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲)))))
  Zy = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲)))), spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₂₂.(𝐱𝐲)))))  
  σ = I(2) ⊗ (spdiagm(vec(σₚ.(𝐱𝐲))))
  
  # PML part of the Traction operator
  A = [P_vec_diag[1,1] P_vec_diag[1,2]; P_vec_diag[2,1] P_vec_diag[2,2]]
  B = [P_vec_diag[3,3] P_vec_diag[3,4]; P_vec_diag[4,3] P_vec_diag[4,4]]  
  Tq₀ = [Z    (I(2)⊗𝐇q₀)*Zx     -(I(2)⊗𝐇q₀)*A     Z     Z]
  Tqₙ = [Z     (I(2)⊗𝐇qₙ)*Zx     (I(2)⊗𝐇qₙ)*A     Z     Z]
  Tr₀ = [(I(2)⊗𝐇r₀)*σ*Zy    (I(2)⊗𝐇r₀)*Zy     Z     -(I(2)⊗𝐇r₀)*B     -(I(2)⊗𝐇r₀)*σ*Zy] 
  Trₙ = [(I(2)⊗𝐇rₙ)*σ*Zy     (I(2)⊗𝐇rₙ)*Zy     Z     (I(2)⊗𝐇rₙ)*B     -(I(2)⊗𝐇rₙ)*σ*Zy] 
  Tq₀, Tqₙ, Tr₀, Trₙ
end

"""
Redefine the marker matrix for the PML
"""
function get_marker_matrix(m)
  X = I(2) ⊗ I(m) ⊗ SBP.SBP_2d.E1(1,m);
  Y = I(2) ⊗ I(m) ⊗ SBP.SBP_2d.E1(m,m);
  Xind = findnz(X);
  Yind = findnz(Y);
  
  # Order = [u1, r1, v1, w1, q1], [u2, r2, v2, w2, q2]] Starting indices: ([0, 2m^2, 4m^2, 6m^2, 8m^2]; [10m^2, 12m^2, 14m^2, 16m^2, 18m^2])    
  mk2_u = -sparse(Xind[1] .+ (2m^2), Xind[1] .+ (0m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Xind[1] .+ (2m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2) -
           sparse(Yind[1] .+ (12m^2), Xind[1] .+ (0m^2), ones(length(Yind[1])), 20m^2, 20m^2) +
           sparse(Yind[1] .+ (12m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2)         
  mk2 = mk2_u

  mk3_u = -sparse(Xind[1] .+ (2m^2), Xind[1] .+ (0m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Xind[1] .+ (2m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Yind[1] .+ (12m^2), Xind[1] .+ (0m^2), ones(length(Yind[1])), 20m^2, 20m^2) -
           sparse(Yind[1] .+ (12m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2)
  mk3 = mk3_u

  mk4_u = -sparse(Xind[1] .+ (0m^2), Xind[1] .+ (0m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Xind[1] .+ (0m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2) -
           sparse(Yind[1] .+ (10m^2), Xind[1] .+ (0m^2), ones(length(Yind[1])), 20m^2, 20m^2) +
           sparse(Yind[1] .+ (10m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2)         
  mk4_w = -sparse(Xind[1] .+ (6m^2), Xind[1] .+ (6m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Xind[1] .+ (6m^2), Yind[1] .+ (16m^2), ones(length(Xind[1])), 20m^2, 20m^2) -
           sparse(Yind[1] .+ (16m^2), Xind[1] .+ (6m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
           sparse(Yind[1] .+ (16m^2), Yind[1] .+ (16m^2), ones(length(Xind[1])), 20m^2, 20m^2)
  mk4 = mk4_u + mk4_w
  
  mk2, mk3, mk4
end

function 𝐊2ᴾᴹᴸ(𝐪𝐫, Ω₁, Ω₂)
  # Obtain the properties of the first layer
  detJ₁(x) = (det∘J)(x,Ω₁)
  detJ𝒫₁(x) = detJ₁(x)*t𝒫(Ω₁, x)
  detJ𝒫ᴾᴹᴸ₁(x) = detJ₁(x)*t𝒫ᴾᴹᴸ(Ω₁, x)
  P₁ = t𝒫.(Ω₁, 𝐪𝐫) # Elasticity Bulk (For traction)
  JP₁ = detJ𝒫₁.(𝐪𝐫) # Elasticity Bulk with det(J) multiplied
  PML₁ =  t𝒫ᴾᴹᴸ.(Ω₁, 𝐪𝐫) # PML Bulk (For traction??)
  JPML₁ =  detJ𝒫ᴾᴹᴸ₁.(𝐪𝐫) # PML Bulk with det(J) multiplied

  # Obtain the properties of the second layer
  detJ₂(x) = (det∘J)(x,Ω₂)
  detJ𝒫₂(x) = detJ₂(x)*t𝒫(Ω₂, x)
  detJ𝒫ᴾᴹᴸ₂(x) = detJ₂(x)*t𝒫ᴾᴹᴸ(Ω₂, x)
  P₂ = t𝒫.(Ω₂, 𝐪𝐫) # Elasticity Bulk (For traction)
  JP₂ = detJ𝒫₂.(𝐪𝐫) # Elasticity Bulk with det(J) multiplied
  PML₂ =  t𝒫ᴾᴹᴸ.(Ω₂, 𝐪𝐫) # PML Bulk (For traction??)
  JPML₂ =  detJ𝒫ᴾᴹᴸ₂.(𝐪𝐫) # PML Bulk with det(J) multiplied
  
  # Get the 2d operators
  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1

  # Jacobian and Surface Jacobian
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫))
  sJ₁ = I(2) ⊗ spdiagm([(J⁻¹s([qᵢ,0.0], Ω₁, [0,-1])^-1) for qᵢ in LinRange(0,1,m)]) ⊗ SBP.SBP_2d.E1(1,m)
  sJ₂ = I(2) ⊗ spdiagm([(J⁻¹s([qᵢ,1.0], Ω₂, [0,1])^-1) for qᵢ in LinRange(0,1,m)]) ⊗ SBP.SBP_2d.E1(m,m)    
  
  # Bulk stiffness matrix components on Layer 1
  𝐏₁ = Pᴱ(Dᴱ(JP₁))  
  𝐏ᴾᴹᴸ₁ = Pᴾᴹᴸ(Dᴾᴹᴸ(JPML₁))  
  xy₁ = Ω₁.(𝐪𝐫)  
  σ₁ = I(2) ⊗ spdiagm(vec(σₚ.(xy₁)))  
  ρσ₁ = I(2) ⊗ spdiagm(vec(ρ.(xy₁).*σₚ.(xy₁)))
  ρσα₁ = α*ρσ₁
  Jinv_vec₁ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω₁))
  Jinv_vec_diag₁ = [spdiagm(vec(p)) for p in Jinv_vec₁] #[qx rx; qy ry]
  JD₁¹ = (I(2)⊗Jinv_vec_diag₁[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[1,2])*(I(2)⊗Dr)
  JD₂¹ = (I(2)⊗Jinv_vec_diag₁[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[2,2])*(I(2)⊗Dr)

  # Bulk stiffness matrix components on Layer 2
  𝐏₂ = Pᴱ(Dᴱ(JP₂))  
  𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Dᴾᴹᴸ(JPML₂))
  xy₂ = Ω₂.(𝐪𝐫)
  σ₂ = I(2) ⊗ spdiagm(vec(σₚ.(xy₂)))  
  ρσ₂ = I(2) ⊗ spdiagm(vec(ρ.(xy₂).*σₚ.(xy₂)))
  ρσα₂ = α*ρσ₂
  Jinv_vec₂ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω₂))
  Jinv_vec_diag₂ = [spdiagm(vec(p)) for p in Jinv_vec₂] #[qx rx; qy ry]
  JD₁² = (I(2)⊗Jinv_vec_diag₂[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₂[1,2])*(I(2)⊗Dr) # x-Derivative operator in physical domain
  JD₂² = (I(2)⊗Jinv_vec_diag₂[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₂[2,2])*(I(2)⊗Dr) # y-Derivative operator in physical domain

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
  𝐓₁ = Tᴱ(P₁) 
  𝐓q₁, 𝐓r₁ = 𝐓₁.A, 𝐓₁.B  
  𝐓ᴾᴹᴸq₀¹, 𝐓ᴾᴹᴸqₙ¹, 𝐓ᴾᴹᴸr₀¹, 𝐓ᴾᴹᴸrₙ¹  = Tᴾᴹᴸ(PML₁, Ω₁, 𝐪𝐫)
  # Get the traction operator of the elasticity and PML parts on Layer 2
  𝐓₂ = Tᴱ(P₂) 
  𝐓q₂, 𝐓r₂ = 𝐓₂.A, 𝐓₂.B  
  𝐓ᴾᴹᴸq₀², 𝐓ᴾᴹᴸqₙ², 𝐓ᴾᴹᴸr₀², 𝐓ᴾᴹᴸrₙ²  = Tᴾᴹᴸ(PML₂, Ω₂, 𝐪𝐫)
  
  # Norm matrices
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm
  
  # Get the overall traction operator on the outer boundaries of both Layer 1 and Layer 2
  𝐓𝐪₀¹ = [-(I(2)⊗𝐇q₀)*𝐓q₁   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀¹
  𝐓𝐪ₙ¹ = [(I(2)⊗𝐇qₙ)*𝐓q₁  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ¹      
  𝐓𝐫ₙ¹ = [(I(2)⊗𝐇rₙ)*𝐓r₁  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ¹    
  𝐓𝐪₀² = [-(I(2)⊗𝐇q₀)*𝐓q₂   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀²
  𝐓𝐪ₙ² = [(I(2)⊗𝐇qₙ)*𝐓q₂  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ²  
  𝐓𝐫₀² = [-(I(2)⊗𝐇r₀)*𝐓r₂  Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀² 
  
  # Interface (But not required. Will be multiplied by 0)
  𝐓𝐫₀¹ = [-(I(2)⊗𝐇r₀)*𝐓r₁  Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀¹
  𝐓𝐫ₙ² = [(I(2)⊗𝐇rₙ)*𝐓r₂  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ²  

  # Interface conditions: 
  zbT = spzeros(Float64, 2m^2, 10n^2)
  zbB = spzeros(Float64, 6m^2, 10n^2)
  P_vec₁ = get_property_matrix_on_grid(PML₁)
  P_vec₂ = get_property_matrix_on_grid(PML₂)
  P_vec_diag₁ = [spdiagm(vec(p)) for p in P_vec₁]  
  P_vec_diag₂ = [spdiagm(vec(p)) for p in P_vec₂]
  B₁ = [P_vec_diag₁[3,3] P_vec_diag₁[3,4]; P_vec_diag₁[4,3] P_vec_diag₁[4,4]] 
  B₂ = [P_vec_diag₂[3,3] P_vec_diag₂[3,4]; P_vec_diag₂[4,3] P_vec_diag₂[4,4]] 
  𝐓𝐫₁ = [sJ₁*𝐓r₁   Z     Z    sJ₁*B₁     Z]  
  𝐓𝐫₂ = [sJ₂*𝐓r₂   Z     Z    sJ₂*B₂     Z]    
  𝐃₁ = 𝐃₂ = blockdiag((I(5)⊗I(2)⊗𝐇r₀), (I(5)⊗I(2)⊗𝐇rₙ))
  𝐓𝐫 = blockdiag([𝐓𝐫₁; zbT; zbB], [𝐓𝐫₂; zbT; zbB])
  # Transpose matrix
  𝐓𝐫₁ᵀ = [(sJ₁*𝐓r₁)'   Z     Z    (sJ₁*B₁)'   Z]  
  𝐓𝐫₂ᵀ = [(sJ₂*𝐓r₂)'   Z     Z    (sJ₂*B₂)'   Z]  
  𝐓𝐫ᵀ = blockdiag([zbT;  𝐓𝐫₁ᵀ; zbB], [zbT;  𝐓𝐫₂ᵀ; zbB])
  
  BH, BT, BHᵀ = get_marker_matrix(m);

  ζ₀ = 30*(m-1)
  𝚯 = 𝐃₁*(BH*𝐓𝐫)
  𝚯ᵀ = -𝐃₁*(𝐓𝐫ᵀ*BHᵀ) 
  Ju = -𝐃₂*(BT)
  𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju

  𝐓ₙ = blockdiag([zbT;   𝐓𝐪₀¹ + 𝐓𝐪ₙ¹ + 0*𝐓𝐫₀¹ + 𝐓𝐫ₙ¹;   zbB], [zbT;   𝐓𝐪₀² + 𝐓𝐪ₙ² + 𝐓𝐫₀² + 0*𝐓𝐫ₙ²;   zbB])
    
  Σ - 𝐓ₙ - 𝐓ᵢ
end 

function 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫, Ω₁, Ω₂)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ.(Ω₁.(𝐪𝐫))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ.(Ω₂.(𝐪𝐫))))
  blockdiag(blockdiag(Id, ρᵥ¹, Id, Id, Id), blockdiag(Id, ρᵥ², Id, Id, Id))
end 

#### #### #### #### #### 
# Begin time stepping  #
#### #### #### #### ####
const Δt = 1e-4
const tf = 0.4
const ntime = ceil(Int, tf/Δt)

"""
A quick implementation of the RK4 scheme
"""
function RK4_1(M, X₀)  
  k₁ = M*X₀
  k₂ = M*(X₀ + (Δt/2)*k₁)
  k₃ = M*(X₀ + (Δt/2)*k₂)
  k₄ = M*(X₀ + (Δt)*k₃)
  X₀ + (Δt/6)*(k₁ + k₂ + k₃ + k₄)
end

"""
Initial conditions (Layer 1)
"""
𝐔₁(x) = @SVector [exp(-40*((x[1]-1.0)^2 + (x[2]-3.0)^2)), -exp(-40*((x[1]-1.0)^2 + (x[2]-3.0)^2))]
𝐑₁(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₁(x) = @SVector [0.0, 0.0]
𝐖₁(x) = @SVector [0.0, 0.0]
𝐐₁(x) = @SVector [0.0, 0.0]

"""
Initial conditions (Layer 2)
"""
𝐔₂(x) = @SVector [exp(-40*((x[1]-1.0)^2 + (x[2]-1.0)^2)), -exp(-40*((x[1]-1.0)^2 + (x[2]-1.0)^2))]
𝐑₂(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₂(x) = @SVector [0.0, 0.0]
𝐖₂(x) = @SVector [0.0, 0.0]
𝐐₂(x) = @SVector [0.0, 0.0]

"""
Function to compute the L²-Error using the reference solution
"""
function compute_l2_error(sol, ref_sol, norm, mn)
  m,n = mn 
  m = Int64(m)
  n = Int64(n)
  ar = ceil(Int64, (n-1)/(m-1))    
  sol_sq_1 = reshape(sol[1:m^2], (m,m))
  sol_sq_2 = reshape(sol[m^2+1:2m^2], (m,m))
  ref_sol_sq_1 = reshape(ref_sol[1:n^2], (n,n))
  ref_sol_sq_2 = reshape(ref_sol[n^2+1:2n^2], (n,n))
  err_1 = zero(sol_sq_1)  
  err_2 = zero(sol_sq_2)  
  for i=1:m, j=1:m
    err_1[i,j] = sol_sq_1[i,j] - ref_sol_sq_1[(i-1)*ar+1, (j-1)*ar+1]
    err_2[i,j] = sol_sq_2[i,j] - ref_sol_sq_2[(i-1)*ar+1, (j-1)*ar+1]
  end  
  err_1 = vec(err_1)
  err_2 = vec(err_2)
  err = vcat(err_1, err_2)  
  sqrt(err'*norm*err)
end

"""
Function to split the solution into the corresponding variables
"""
function split_solution(X)
  N = Int(sqrt(length(X)/10))
  u1,u2 = X[1:N^2], X[N^2+1:2N^2];
  r1,r2 = X[2N^2+1:3N^2], X[3N^2+1:4N^2];
  v1,v2 = X[4N^2+1:5N^2], X[5N^2+1:6N^2];
  w1,w2 = X[6N^2+1:7N^2], X[7N^2+1:8N^2];
  q1,q2 = X[8N^2+1:9N^2], X[9N^2+1:10N^2];
  (u1,u2), (r1,r2), (v1, v2), (w1,w2), (q1,q2)
end

#############################
# Obtain Reference Solution #
#############################
𝐍 = 321
𝐪𝐫 = generate_2d_grid((𝐍, 𝐍));
𝐱𝐲₁ = Ω₁.(𝐪𝐫);
𝐱𝐲₂ = Ω₂.(𝐪𝐫);
stima = 𝐊2ᴾᴹᴸ(𝐪𝐫, Ω₁, Ω₂);
massma = 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫, Ω₁, Ω₂);
# Begin time loop
let
  t = 0.0
  X₀¹ = vcat(eltocols(vec(𝐔₁.(𝐱𝐲₁))), eltocols(vec(𝐑₁.(𝐱𝐲₁))), eltocols(vec(𝐕₁.(𝐱𝐲₁))), eltocols(vec(𝐖₁.(𝐱𝐲₁))), eltocols(vec(𝐐₁.(𝐱𝐲₁))));
  X₀² = vcat(eltocols(vec(𝐔₂.(𝐱𝐲₂))), eltocols(vec(𝐑₂.(𝐱𝐲₂))), eltocols(vec(𝐕₂.(𝐱𝐲₂))), eltocols(vec(𝐖₂.(𝐱𝐲₂))), eltocols(vec(𝐐₂.(𝐱𝐲₂))));
  X₀ = vcat(X₀¹, X₀²)
  global Xref = zero(X₀)
  M = massma*stima
  for i=1:ntime
    Xref = RK4_1(M, X₀)
    X₀ = Xref
    t += Δt    
    (i%100==0) && println("Done t = "*string(t))
  end  
end 

############################
# Grid Refinement Analysis # 
############################
𝒩 = [21,41,81,161];
L²Error = zeros(Float64,length(𝒩))
for (N,i) ∈ zip(𝒩,1:lastindex(𝒩))
  let 
    𝐪𝐫 = generate_2d_grid((N,N));
    𝐱𝐲₁ = Ω₁.(𝐪𝐫);
    𝐱𝐲₂ = Ω₂.(𝐪𝐫);
    stima = 𝐊2ᴾᴹᴸ(𝐪𝐫, Ω₁, Ω₂);
    massma = 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫, Ω₁, Ω₂);
    # Begin time loop
    let
      t = 0.0      
      X₀¹ = vcat(eltocols(vec(𝐔₁.(𝐱𝐲₁))), eltocols(vec(𝐑₁.(𝐱𝐲₁))), eltocols(vec(𝐕₁.(𝐱𝐲₁))), eltocols(vec(𝐖₁.(𝐱𝐲₁))), eltocols(vec(𝐐₁.(𝐱𝐲₁))));
      X₀² = vcat(eltocols(vec(𝐔₂.(𝐱𝐲₂))), eltocols(vec(𝐑₂.(𝐱𝐲₂))), eltocols(vec(𝐕₂.(𝐱𝐲₂))), eltocols(vec(𝐖₂.(𝐱𝐲₂))), eltocols(vec(𝐐₂.(𝐱𝐲₂))));
      X₀ = vcat(X₀¹, X₀²)
      global X₁ = zero(X₀)
      M = massma*stima
      for i=1:ntime
        X₁ = RK4_1(M, X₀)
        X₀ = X₁
        t += Δt    
        # println("Done t = "*string(t))
      end  
    end  
    # Compute the error with the reference solution
    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    Hq = sbp_q.norm
    Hr = sbp_r.norm
    𝐇 = (I(2) ⊗ Hq ⊗ Hr)

    # Split the solution to obtain the displacement vectors (u1, u2)
    X_split₁ = split_solution(X₁[1:10m^2])    
    X_split₂ = split_solution(X₁[10m^2+1:20m^2])
    X_split_ref₁ = split_solution(Xref[1:10𝐍^2])
    X_split_ref₂ = split_solution(Xref[10𝐍^2+1:20𝐍^2])    
    u1₁, u2₁ = X_split₁[1] # Current refinement
    u1₂, u2₂ = X_split₂[1] # Current refinement
    u1ref₁,u2ref₁ = X_split_ref₁[1];
    u1ref₂,u2ref₂ = X_split_ref₂[1];
    sol₁ = vcat(u1₁, u2₁);   
    sol_ref₁ = vcat(u1ref₁, u2ref₁)
    sol₂ = vcat(u1₂, u2₂);   
    sol_ref₂ = vcat(u1ref₂, u2ref₂)    
    L²Error[i]  = sqrt(compute_l2_error(sol₁, sol_ref₁, 𝐇, (n,𝐍))^2 +
                       compute_l2_error(sol₂, sol_ref₂, 𝐇, (n,𝐍))^2)       
    println("Done N = "*string(N))
  end
end

h = 1 ./(𝒩 .- 1);
rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1]);
@show L²Error
@show rate

u1₁,u2₁ = split_solution(X₁[1:10*𝒩[end]^2])[1];
u1₂,u2₂ = split_solution(X₁[10*𝒩[end]^2+1:20*𝒩[end]^2])[1];
𝐪𝐫 = generate_2d_grid((𝒩[end], 𝒩[end]));
xy₁ = vec(Ω₁.(𝐪𝐫));
xy₂ = vec(Ω₂.(𝐪𝐫));
plt1 = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt1, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt1, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
scatter!(plt1, Tuple.([cᵢ(q) for q in LinRange(0,1,𝒩[end])]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))
title!(plt1, "Horizontal Displacement")
plt2 = scatter(Tuple.(xy₁), zcolor=vec(u2₁), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.1, label="");
scatter!(plt2, Tuple.(xy₂), zcolor=vec(u2₂), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.1, label="");
scatter!(plt2, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
scatter!(plt2, Tuple.([cᵢ(q) for q in LinRange(0,1,𝒩[end])]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))
title!(plt2, "Vertical Displacement")

plt3 = scatter(Tuple.(xy₁), zcolor=vec(σₚ.(xy₁)), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.(xy₂), zcolor=vec(σₚ.(xy₂)), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
scatter!(plt3, Tuple.([cᵢ(q) for q in LinRange(0,1,𝒩[end])]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800));
title!(plt3, "PML Function")

plt4 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
plot!(plt4, h,  h.^4, label="O(h⁴)", lw=1, xlabel="h", ylabel="L² Error");