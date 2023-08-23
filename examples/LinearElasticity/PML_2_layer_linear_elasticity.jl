##################################################
# Program to solve the 2 layer linear elasticity #
# Incluing the Perfectly Matched Layer Boundary  #
##################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine

# Define the first domain
c₀¹(r) = @SVector [0.0, r]
c₁¹(q) = @SVector [q, 0.0]
c₂¹(r) = @SVector [1.0, r]
c₃¹(q) = @SVector [q, 1.0]
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
Ω₁(qr) = S(qr, domain₁)

# Define the second domain
c₀²(r) = @SVector [0.0, r]
c₁²(q) = @SVector [q, 0.0]
c₂²(r) = @SVector [1.0, r]
c₃²(q) = @SVector [q, 1.0]
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
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
const Lₓ = 0.8
const δ = 0.1*Lₓ
const σ₀ = 4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
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
  
  mk2 = -sparse(Xind[1], Xind[1], ones(length(Xind[1])), 20m^2, 20m^2) +
         sparse(Xind[1], Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2) -
         sparse(Yind[1] .+ (10m^2), Xind[1], ones(length(Yind[1])), 20m^2, 20m^2) +
         sparse(Yind[1] .+ (10m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2)
  
  mk3 = -sparse(Xind[1], Xind[1], ones(length(Xind[1])), 20m^2, 20m^2) +
         sparse(Xind[1], Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2) +
         sparse(Yind[1] .+ (10m^2), Xind[1], ones(length(Yind[1])), 20m^2, 20m^2) -
         sparse(Yind[1] .+ (10m^2), Yind[1] .+ (10m^2), ones(length(Xind[1])), 20m^2, 20m^2)
  
  mk2, mk3
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
      (𝐏₁+ρσα₁)  -ρσ₁     (𝐏ᴾᴹᴸ₁)        -ρσα₁;
      JD₁¹    Z    -(α*Id+σ₁)   Z       Z;
      JD₂¹    Z       Z      -α*Id    Z;
      α*Id   Z       Z       Z     -α*Id ]
  Σ₂ = [   Z      Id       Z       Z       Z;
      (𝐏₂+ρσα₂)  -ρσ₂     (𝐏ᴾᴹᴸ₂)        -ρσα₂;
      JD₁²    Z    -(α*Id+σ₁)   Z       Z;
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
  𝐓𝐫₀² = [(I(2)⊗𝐇rₙ)*𝐓r₂  Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀²  

  # Interface conditions:  
  𝐓ᵢ = 
  
  # zbT = spzeros(Float64, 2m^2, 10n^2)
  # zbB = spzeros(Float64, 6m^2, 10n^2)
  # Σ - [zbT;   𝐓𝐪₀ + 𝐓𝐪ₙ + 𝐓𝐫₀ + 𝐓𝐫ₙ;   zbB]
end 