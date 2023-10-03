##################################################
# Program to solve the 2 layer linear elasticity #
# Incluing the Perfectly Matched Layer Boundary  #
##################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine
using Arpack

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [1,2]
# Define the parametrization for interface
# f(q) = 0.0*exp(-10*4π*(q-0.5)^2)
pf = 7
f(q) = 0.1*sin(pf*π*q)
cᵢ(q) = [1.1*q, f(q)];
# Define the rest of the boundary
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [1.1, r]; # Right boundary
c₃¹(q) = [1.1*q, 0.0]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, r - 1.0]; # Left boundary
c₁²(q) = [1.1*q, -1.0]; # Bottom boundary. 
c₂²(r) = [1.1, r - 1.0]; # Right boundary
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
function λ(x)
  if((x[2] ≈ cᵢ(x[1])[2]) || (x[2] > cᵢ(x[1])[2]))
    return 1.0
  else
    return 0.25
  end
end
function μ(x)
  if((x[2] ≈ cᵢ(x[1])[2]) || (x[2] > cᵢ(x[1])[2]))  
    return 1.0
  else
    return 0.25
  end
end

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
  detJ = (det∘J)(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫(x)*S)*detJ
end 

"""
Transform the PML properties to the material grid
"""
function t𝒫ᴾᴹᴸ(Ω, qr)
  x = Ω(qr)
  invJ = J⁻¹(qr, Ω)
  detJ = (det∘J)(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫ᴾᴹᴸ(x))*detJ
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

  # Inverse Jacobian
  Jinv_vec = get_property_matrix_on_grid(J.(𝐪𝐫, Ω))
  Jinv_vec_diag = [spdiagm(vec(p)) for p in Jinv_vec] #[qx rx; qy ry]      
  # Evaluate the functions on the physical grid
  Zx = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₁₁.(𝐱𝐲))))*Jinv_vec_diag[1,1], spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲))))*Jinv_vec_diag[1,1])
  Zy = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲))))*Jinv_vec_diag[2,2], spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₂₂.(𝐱𝐲))))*Jinv_vec_diag[2,2])  
  # Zx = I(2) ⊗ I(m) ⊗ I(m)
  # Zy = I(2) ⊗ I(m) ⊗ I(m)
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

function E1(i,j,m)
  X = spzeros(Float64,m,m)
  X[i,j] = 1.0
  X
end

"""
Redefine the marker matrix for the PML
"""
function get_marker_matrix(m)  
  W₁ = I(2) ⊗ I(m) ⊗ E1(1,1,m)
  W₂ = I(2) ⊗ I(m) ⊗ E1(m,m,m)
  Z₁ = I(2) ⊗ I(m) ⊗ E1(1,m,m)  
  Z₂ = I(2) ⊗ I(m) ⊗ E1(m,1,m) 
  Z = zero(W₁)
  
  mk1 = [Z   Z   Z   Z    Z    Z   Z   Z   Z   Z;
        -W₁  Z   Z   Z    Z    Z₁  Z   Z   Z   Z; 
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;        
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        -Z₂  Z   Z   Z    Z    W₂  Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z];

  mk2 = [Z   Z   Z   Z    Z    Z   Z   Z   Z   Z;
        -W₁  Z   Z   Z    Z    Z₁  Z   Z   Z   Z; 
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;                
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z₂   Z   Z   Z    Z   -W₂  Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z;
        Z    Z   Z   Z    Z    Z   Z   Z   Z   Z];

  mk3 = [-W₁   Z   Z   Z    Z    Z₁   Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z; 
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z;
         -Z₂   Z   Z   Z    Z    W₂   Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z; 
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z;
          Z    Z   Z   Z    Z    Z    Z   Z   Z   Z];

  mk1, mk2, mk3
end

function 𝐊2ᴾᴹᴸ(𝐪𝐫, Ω₁, Ω₂)
  # Obtain the properties of the first layer
  detJ₁(x) = (det∘J)(x,Ω₁)  
  P₁ = t𝒫.(Ω₁, 𝐪𝐫) # Elasticity Bulk (For traction)
  PML₁ =  t𝒫ᴾᴹᴸ.(Ω₁, 𝐪𝐫) # PML Bulk  

  # Obtain the properties of the second layer
  detJ₂(x) = (det∘J)(x,Ω₂)  
  P₂ = t𝒫.(Ω₂, 𝐪𝐫) # Elasticity Bulk (For traction)
  PML₂ =  t𝒫ᴾᴹᴸ.(Ω₂, 𝐪𝐫) # PML Bulk
  
  # Get the 2d operators
  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1

  # Jacobian and Surface Jacobian
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫))    
  
  # Bulk stiffness matrix components on Layer 1
  𝐏₁ = Pᴱ(Dᴱ(P₁))  
  𝐏ᴾᴹᴸ₁ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₁))  
  xy₁ = Ω₁.(𝐪𝐫)  
  σ₁ = I(2) ⊗ spdiagm(vec(σₚ.(xy₁)))  
  ρσ₁ = I(2) ⊗ spdiagm(vec(ρ.(xy₁).*σₚ.(xy₁)))
  ρσα₁ = α*ρσ₁
  Jinv_vec₁ = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω₁))
  Jinv_vec_diag₁ = [spdiagm(vec(p)) for p in Jinv_vec₁] #[qx rx; qy ry]
  JD₁¹ = (I(2)⊗Jinv_vec_diag₁[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[1,2])*(I(2)⊗Dr)
  JD₂¹ = (I(2)⊗Jinv_vec_diag₁[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag₁[2,2])*(I(2)⊗Dr)

  # Bulk stiffness matrix components on Layer 2
  𝐏₂ = Pᴱ(Dᴱ(P₂))  
  𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML₂))
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
  𝐓𝐪₀¹ = spdiagm(detJ1₁.^-1)*([-(I(2)⊗𝐇q₀)*𝐓q₁   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀¹)
  𝐓𝐪ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇qₙ)*𝐓q₁  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ¹)
  𝐓𝐫ₙ¹ = spdiagm(detJ1₁.^-1)*([(I(2)⊗𝐇rₙ)*𝐓r₁  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ¹)
  𝐓𝐪₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇q₀)*𝐓q₂   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀²)
  𝐓𝐪ₙ² = spdiagm(detJ1₂.^-1)*([(I(2)⊗𝐇qₙ)*𝐓q₂  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ²)
  𝐓𝐫₀² = spdiagm(detJ1₂.^-1)*([-(I(2)⊗𝐇r₀)*𝐓r₂  Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀²)
  
  # Interface (But not required. Will be multiplied by 0)
  𝐓𝐫₀¹ = spdiagm(detJ1₁.^-1)*([-(I(2)⊗𝐇r₀)*𝐓r₁  Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀¹)
  𝐓𝐫ₙ² = spdiagm(detJ1₂.^-1)*([(I(2)⊗𝐇rₙ)*𝐓r₂  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ²)

  # Interface conditions: 
  zbT = spzeros(Float64, 2m^2, 10n^2)
  zbB = spzeros(Float64, 6m^2, 10n^2)
  P_vec₁ = get_property_matrix_on_grid(PML₁)
  P_vec₂ = get_property_matrix_on_grid(PML₂)
  P_vec_diag₁ = [spdiagm(vec(p)) for p in P_vec₁]  
  P_vec_diag₂ = [spdiagm(vec(p)) for p in P_vec₂]
  B₁ = [P_vec_diag₁[3,3] P_vec_diag₁[3,4]; P_vec_diag₁[4,3] P_vec_diag₁[4,4]] 
  B₂ = [P_vec_diag₂[3,3] P_vec_diag₂[3,4]; P_vec_diag₂[4,3] P_vec_diag₂[4,4]] 
  𝐓𝐫₁ = spdiagm(detJ1₁.^-1)*[(𝐓r₁)   Z     Z    (B₁)     Z]  
  𝐓𝐫₂ = spdiagm(detJ1₂.^-1)*[(𝐓r₂)   Z     Z    (B₂)     Z]    
  
  𝐓𝐫 = blockdiag([𝐓𝐫₁; zbT; zbB], [𝐓𝐫₂; zbT; zbB])
  # Transpose matrix
  𝐓𝐫₁ᵀ = spdiagm(detJ1₁.^-1)*[(𝐓r₁)'   Z     Z    (B₁)'   Z]  
  𝐓𝐫₂ᵀ = spdiagm(detJ1₂.^-1)*[(𝐓r₂)'   Z     Z    (B₂)'   Z]  
  𝐓𝐫ᵀ = blockdiag([zbT;  𝐓𝐫₁ᵀ; zbB], [zbT;  𝐓𝐫₂ᵀ; zbB])
  
  BH, BT, BHᵀ = get_marker_matrix(m);
  Hq⁻¹ = (sbp_q.norm\I(m)) |> sparse
  Hr⁻¹ = (sbp_r.norm\I(m)) |> sparse
  # Hq = sbp_q.norm
  Hr = sbp_q.norm
  𝐃₁⁻¹ = blockdiag((I(10)⊗Hq⁻¹⊗Hr⁻¹), (I(10)⊗Hq⁻¹⊗Hr⁻¹))
  𝐃 = blockdiag((I(10)⊗(Hr)⊗ I(m))*(I(10)⊗I(m)⊗ E1(1,1,m)), (I(10)⊗(Hr)⊗I(m))*(I(10)⊗I(m)⊗ E1(m,m,m)))
  𝐃₂ = blockdiag((I(2)⊗(Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(1,1,m)), Z, Z, (I(2)⊗(Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(1,1,m)), Z, 
                 (I(2)⊗(Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(m,m,m)), Z, Z, (I(2)⊗(Hr)⊗I(m))*(I(2)⊗I(m)⊗ E1(m,m,m)), Z)

  ζ₀ = 40/h
  𝚯 = 𝐃₁⁻¹*𝐃*BH*𝐓𝐫
  𝚯ᵀ = -𝐃₁⁻¹*𝐓𝐫ᵀ*BHᵀ*𝐃₂
  Ju = -𝐃₁⁻¹*𝐃*BT
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
"""
A non-allocating implementation of the RK4 scheme
"""
function RK4_1!(M, sol)  
  X₀, k₁, k₂, k₃, k₄, tmp = sol
  # k1 step  
  mul!(k₁, M, X₀)  
  # k2 step    
  for i=1:lastindex(k₂)
    tmp[i] = (X₀[i] + (Δt/2)*k₁[i])  
  end  
  mul!(k₂, M, tmp)  
  # k3 step
  for i=1:lastindex(k₃)
    tmp[i] = (X₀[i] + (Δt/2)*k₂[i])  
  end
  mul!(k₃, M, tmp)
  # k4 step
  for i=1:lastindex(k₄)
    tmp[i] = (X₀[i] + (Δt)*k₃[i])  
  end
  mul!(k₄, M, tmp)
  for i=1:lastindex(X₀)
    X₀[i] = (X₀[i] + (Δt/6)*(k₁[i] + k₂[i] + k₃[i] + k₄[i]))
    tmp[i] = 0.0
  end
  X₀
end

"""
Initial conditions (Layer 1)
"""
𝐔₁(x) = @SVector [exp(-100*((x[1]-0.55)^2 + (x[2]-0.5)^2)), -exp(-100*((x[1]-0.55)^2 + (x[2]-0.5)^2))]
𝐑₁(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕₁(x) = @SVector [0.0, 0.0]
𝐖₁(x) = @SVector [0.0, 0.0]
𝐐₁(x) = @SVector [0.0, 0.0]

"""
Initial conditions (Layer 2)
"""
𝐔₂(x) = @SVector [exp(-100*((x[1]-0.55)^2 + (x[2]-0.5)^2)), -exp(-100*((x[1]-0.55)^2 + (x[2]-0.5)^2))]
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
𝐍 = 61
𝐪𝐫 = generate_2d_grid((𝐍, 𝐍));
𝐱𝐲₁ = Ω₁.(𝐪𝐫);
𝐱𝐲₂ = Ω₂.(𝐪𝐫);
const h = Lₓ/(𝐍-1)
stima = 𝐊2ᴾᴹᴸ(𝐪𝐫, Ω₁, Ω₂);
massma = 𝐌2ᴾᴹᴸ⁻¹(𝐪𝐫, Ω₁, Ω₂);

cmax = sqrt(2^2+1^2)
τ₀ = 0.8
const Δt = 0.2/(cmax*τ₀)*h
const tf = 2Δt
const ntime = ceil(Int, tf/Δt)
solmax = zeros(Float64, ntime)

xy₁ = vec(Ω₁.(𝐪𝐫));
xy₂ = vec(Ω₂.(𝐪𝐫));

M = massma*stima 
iter = 0
let  
  t = iter*tf
  X₀¹ = vcat(eltocols(vec(𝐔₁.(𝐱𝐲₁))), eltocols(vec(𝐑₁.(𝐱𝐲₁))), eltocols(vec(𝐕₁.(𝐱𝐲₁))), eltocols(vec(𝐖₁.(𝐱𝐲₁))), eltocols(vec(𝐐₁.(𝐱𝐲₁))));
  X₀² = vcat(eltocols(vec(𝐔₂.(𝐱𝐲₂))), eltocols(vec(𝐑₂.(𝐱𝐲₂))), eltocols(vec(𝐕₂.(𝐱𝐲₂))), eltocols(vec(𝐖₂.(𝐱𝐲₂))), eltocols(vec(𝐐₂.(𝐱𝐲₂))));
  X₀ = vcat(X₀¹, X₀²)  
  # X₀ = X₁
  # Arrays to store the RK-variables
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀))
  tmp = zeros(Float64, length(X₀))   
  for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄, tmp
    X₀ = RK4_1!(M,sol)    
    t += Δt    
    solmax[i] = maximum(abs.(X₀))
    (i%1000==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(solmax[i]))    
    
    ## Plotting to get GIFs
    # u1₁,u2₁ = split_solution(view(X₀, 1:10*𝐍^2), 𝐍)[1];
    # u1₂,u2₂ = split_solution(view(X₀, 10*𝐍^2+1:20*𝐍^2), 𝐍)[1];              
    # plt1₁ = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");    
    # scatter!(plt1₁, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    # scatter!(plt1₁, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1);
    # scatter!(plt1₁, Tuple.([cᵢ(q) for q in LinRange(0,1,𝒩[end])]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))    
    # title!(plt1₁, "Time t="*string(round(t,digits=4)))
    # plt1₂ = scatter(Tuple.(xy₁), zcolor=σₚ.(vec(Ω₁.(𝐪𝐫))), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="")
    # scatter!(plt1₂, Tuple.(xy₂), zcolor=σₚ.(vec(Ω₂.(𝐪𝐫))), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="")
    # scatter!(plt1₂, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([0.0,0.0])[2],Ω₁([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1);
    # scatter!(plt1₂, Tuple.([cᵢ(q) for q in LinRange(0,1,𝒩[end])]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800))    
    # plt1 = plot(plt1₁, plt1₂, layout=(1,2))
  end
  global X₁ = X₀  
end 

u1₁,u2₁ = split_solution(view(X₁, 1:10*𝐍^2), 𝐍)[1];
u1₂,u2₂ = split_solution(view(X₁, 10*𝐍^2+1:20*𝐍^2), 𝐍)[1];

plt1 = scatter(Tuple.(xy₁), zcolor=vec(u1₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.(xy₂), zcolor=vec(u1₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt1, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt1, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt1, "Horizontal Displacement")
plt2 = scatter(Tuple.(xy₁), zcolor=vec(u2₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.(xy₂), zcolor=vec(u2₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.1, label="");
scatter!(plt2, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), markercolor=:blue, markersize=3, msw=0.1, label="");
scatter!(plt2, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), markercolor=:green, markersize=2, msw=0.1, label="", right_margin=20*Plots.mm)
title!(plt2, "Vertical Displacement")
plt3 = scatter(Tuple.(xy₁), zcolor=vec(σₚ.(xy₁)), colormap=:turbo, markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.(xy₂), zcolor=vec(σₚ.(xy₂)), colormap=:turbo, markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω₂([1.0,0.0])[2],Ω₁([1.0,1.0])[2],𝐍)]), label="x ≥ "*string(round(Lₓ,digits=4))*" (PML)", markercolor=:white, markersize=2, msw=0.1, colorbar_exponentformat="power");
scatter!(plt3, Tuple.([cᵢ(q) for q in LinRange(0,1,𝐍)]), label="Interface", markercolor=:green, markersize=2, msw=0.1, size=(800,800), right_margin=20*Plots.mm);
title!(plt3, "PML Function")
# plt4 = plot()
# plot!(plt4, LinRange(iter*tf,(iter+1)*tf,ntime), solmax, yaxis=:log10, label="||U||₍∞₎,  f(q) = sin("*string(pf)*"πq), τ = 40/h", lw=2)
# plt5 = plot(plt1, plt3, plt2, plt4, layout=(2,2));
# savefig(plt5, "C:\\Users\\baka0042\\OneDrive - Umeå universitet\\Postdoc-Balaje\\PML_elastic_layered_media\\PML GIFs\\Steady-State\\eg6.pdf"); 