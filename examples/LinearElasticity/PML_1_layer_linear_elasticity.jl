###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine

# Define the domain
c₀(r) = @SVector [0.0, r]
c₁(q) = @SVector [q, 0.0]
c₂(r) = @SVector [1.0, r]
c₃(q) = @SVector [q, 1.0]
domain = domain_2d(c₀, c₁, c₂, c₃)
Ω(qr) = S(qr, domain)

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
The PML damping function
"""
const Lₓ = 0.9
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

"""
Transform the material property matrix to the reference grid
"""
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
  Pqr::Matrix{SMatrix{m,n,Float64}}
    = [P(x₁₁) P(x₁₂) ... P(x₁ₙ)
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
function Tᴾᴹᴸ(Pqr::Matrix{SMatrix{4,4,Float64,16}}, Ω)
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

function 𝐊ᴾᴹᴸ(𝐪𝐫, Ω)
  detJ(x) = (det∘J)(x,Ω)
  detJ𝒫(x) = detJ(x)*t𝒫(Ω, x)
  detJ𝒫ᴾᴹᴸ(x) = detJ(x)*t𝒫ᴾᴹᴸ(Ω, x)

  P = t𝒫.(Ω, 𝐪𝐫) # Elasticity Bulk (For traction)
  JP = detJ𝒫.(𝐪𝐫) # Elasticity Bulk with det(J) multiplied
  PML =  t𝒫ᴾᴹᴸ.(Ω, 𝐪𝐫) # PML Bulk (For traction??)
  JPML =  detJ𝒫ᴾᴹᴸ.(𝐪𝐫) # PML Bulk with det(J) multiplied

  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1

  # Bulk stiffness matrix
  𝐏 = Pᴱ(Dᴱ(JP))  
  𝐏ᴾᴹᴸ = Pᴾᴹᴸ(Dᴾᴹᴸ(JPML))
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Z = zero(Id)  
  xy = Ω.(𝐪𝐫)
  σ = I(2) ⊗ spdiagm(vec(σₚ.(xy)))  
  ρσ = I(2) ⊗ spdiagm(vec(ρ.(xy).*σₚ.(xy)))
  ρσα = α*ρσ

  # Get the derivative operator transformed to the reference grid
  Jinv_vec = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω))
  Jinv_vec_diag = [spdiagm(vec(p)) for p in Jinv_vec] #[qx rx; qy ry]
  JD₁ = (I(2)⊗Jinv_vec_diag[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag[1,2])*(I(2)⊗Dr)
  JD₂ = (I(2)⊗Jinv_vec_diag[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag[2,2])*(I(2)⊗Dr)

  # Assemble the bulk stiffness matrix
  Σ = [   Z      Id       Z       Z       Z;
       (𝐏+ρσα)  -ρσ     (𝐏ᴾᴹᴸ)        -ρσα;
          JD₁    Z    -(α*Id+σ)   Z       Z;
          JD₂    Z       Z      -α*Id    Z;
          α*Id   Z       Z       Z     -α*Id ]

  # Get the traction operator of the elasticity part
  𝐓 = Tᴱ(P) 
  𝐓q, 𝐓r = 𝐓.A, 𝐓.B  
  # Get the traction operator of the PML part
  𝐓ᴾᴹᴸq₀, 𝐓ᴾᴹᴸqₙ, 𝐓ᴾᴹᴸr₀, 𝐓ᴾᴹᴸrₙ  = Tᴾᴹᴸ(PML, Ω)

  # Norm matrices
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm

  # Get the overall traction operator  
  𝐓𝐪₀ = [-(I(2)⊗𝐇q₀)*𝐓q   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀
  𝐓𝐪ₙ = [(I(2)⊗𝐇qₙ)*𝐓q  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ
  𝐓𝐫₀ = [-(I(2)⊗𝐇r₀)*𝐓r   Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀  
  𝐓𝐫ₙ = [(I(2)⊗𝐇rₙ)*𝐓r  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ  

  zbT = spzeros(Float64, 2m^2, 10n^2)
  zbB = spzeros(Float64, 6m^2, 10n^2)
  Σ - [zbT;   𝐓𝐪₀ + 𝐓𝐪ₙ + 𝐓𝐫₀ + 𝐓𝐫ₙ;   zbB]
end 

function 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫, Ω)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  ρᵥ = I(2)⊗spdiagm(vec(1 ./ρ.(Ω.(𝐪𝐫))))
  blockdiag(Id, ρᵥ, Id, Id, Id)
end 

N = 61
𝐪𝐫 = generate_2d_grid((N,N));
𝐱𝐲 = Ω.(𝐪𝐫);
stima = 𝐊ᴾᴹᴸ(𝐪𝐫, Ω);
massma = 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫, Ω);

#### #### #### #### #### 
# Begin time stepping  #
#### #### #### #### #### 
Δt = 10^-3
tf = 0.2
ntime = ceil(Int, tf/Δt)
# Initial conditions
𝐔(x) = @SVector [exp(-20*((x[1]-0.5)^2 + (x[2]-0.5)^2)), exp(-30*((x[1]-0.5)^2 + (x[2]-0.5)^2))]
𝐑(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]

# Raw initial condition vector


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

# Begin time loop
plt1 = plot()
let
  t = 0.0
  X₀ = vcat(eltocols(vec(𝐔.(𝐱𝐲))), eltocols(vec(𝐑.(𝐱𝐲))), eltocols(vec(𝐕.(𝐱𝐲))), eltocols(vec(𝐖.(𝐱𝐲))), eltocols(vec(𝐐.(𝐱𝐲))));
  global X₁ = zero(X₀)
  M = massma*stima
  for i=1:ntime
    X₁ = RK4_1(M, X₀)
    X₀ = X₁
    t += Δt    
    println("Done t = "*string(t))
  end  
end
X₀ = vcat(eltocols(vec(𝐔.(𝐱𝐲))), eltocols(vec(𝐑.(𝐱𝐲))), eltocols(vec(𝐕.(𝐱𝐲))), eltocols(vec(𝐖.(𝐱𝐲))), eltocols(vec(𝐐.(𝐱𝐲))));
𝐔₀,𝐕₀ = X₀[1:N^2], X₀[N^2+1:2N^2];
𝐔₁,𝐕₁ = X₁[1:N^2], X₁[N^2+1:2N^2];
m, n = N, N
q,r = LinRange(0,1,m), LinRange(0,1,n);
plt1 = contourf(q, r, reshape(𝐔₁, (N,N)))
plt2 = contourf(q, r, reshape(𝐕₁, (N,N)))
plt = plot(plt1, plt2, layout=(1,2), size=(800,800))