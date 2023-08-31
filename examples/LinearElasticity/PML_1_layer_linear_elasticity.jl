###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine

# Define the domain
c₀(r) = @SVector [0.0, 2r]
c₁(q) = @SVector [2q, 0.0]
c₂(r) = @SVector [2.0, 2r]
c₃(q) = @SVector [2q, 2.0]
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
The PML damping
"""
const Lₓ = 1.6
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

  # Determinant of the Jacobian Matrix
  detJ1 = [1,1] ⊗ vec(detJ.(𝐪𝐫))
  
  # Get the derivative operator transformed to the reference grid
  Jinv_vec = get_property_matrix_on_grid(J⁻¹.(𝐪𝐫, Ω))
  Jinv_vec_diag = [spdiagm(vec(p)) for p in Jinv_vec] #[qx rx; qy ry]
  JD₁ = (I(2)⊗Jinv_vec_diag[1,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag[1,2])*(I(2)⊗Dr)
  JD₂ = (I(2)⊗Jinv_vec_diag[2,1])*(I(2)⊗Dq) + (I(2)⊗Jinv_vec_diag[2,2])*(I(2)⊗Dr)
  
  # Assemble the bulk stiffness matrix
  Σ = [   Z      Id       Z       Z       Z;
  (spdiagm(detJ1.^-1)*𝐏+ρσα)  -ρσ     (spdiagm(detJ1.^-1)*𝐏ᴾᴹᴸ)        -ρσα;
  JD₁    Z    -(α*Id+σ)   Z       Z;
  JD₂    Z       Z      -α*Id    Z;
  α*Id   Z       Z       Z     -α*Id ]
  
  # Get the traction operator of the elasticity part
  𝐓 = Tᴱ(P) 
  𝐓q, 𝐓r = 𝐓.A, 𝐓.B  
  # Get the traction operator of the PML part
  𝐓ᴾᴹᴸq₀, 𝐓ᴾᴹᴸqₙ, 𝐓ᴾᴹᴸr₀, 𝐓ᴾᴹᴸrₙ  = Tᴾᴹᴸ(PML, Ω, 𝐪𝐫)
  
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

#### #### #### #### #### 
# Begin time stepping  #
#### #### #### #### ####
const Δt = 5e-4
const tf = 40.0
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
Initial conditions
"""
𝐔(x) = @SVector [exp(-10*((x[1]-1.0)^2 + (x[2]-1.0)^2)), -exp(-10*((x[1]-1.0)^2 + (x[2]-1.0)^2))]
𝐑(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]

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
N = 81
𝐪𝐫 = generate_2d_grid((N,N));
𝐱𝐲 = Ω.(𝐪𝐫);
stima = 𝐊ᴾᴹᴸ(𝐪𝐫, Ω);
massma = 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫, Ω);
# Begin time loop
let
  t = 0.0
  X₀ = vcat(eltocols(vec(𝐔.(𝐱𝐲))), eltocols(vec(𝐑.(𝐱𝐲))), eltocols(vec(𝐕.(𝐱𝐲))), eltocols(vec(𝐖.(𝐱𝐲))), eltocols(vec(𝐐.(𝐱𝐲))));
  global Xref = zero(X₀)
  M = massma*stima
  @gif for i=1:ntime
    Xref = RK4_1(M, X₀)
    X₀ = Xref
    t += Δt    
    (i%100==0) && println("Done t = "*string(t))

    u1ref,u2ref = split_solution(Xref)[1];
    𝐪𝐫 = generate_2d_grid((N,N));
    xy = vec(Ω.(𝐪𝐫));
    plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);  
    title!(plt3, "Time t="*string(t))    
  end  every 100
end  

#= 
############################
# Grid Refinement Analysis # 
############################
𝒩 = [21,41,81,161]
L²Error = zeros(Float64,length(𝒩))
for (N,i) ∈ zip(𝒩,1:lastindex(𝒩))
  let 
    𝐪𝐫 = generate_2d_grid((N,N));
    𝐱𝐲 = Ω.(𝐪𝐫);
    stima = 𝐊ᴾᴹᴸ(𝐪𝐫, Ω);
    massma = 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫, Ω);
    # Begin time loop
    let
      t = 0.0
      X₀ = vcat(eltocols(vec(𝐔.(𝐱𝐲))), eltocols(vec(𝐑.(𝐱𝐲))), eltocols(vec(𝐕.(𝐱𝐲))), eltocols(vec(𝐖.(𝐱𝐲))), eltocols(vec(𝐐.(𝐱𝐲))));
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
    u1, u2 = split_solution(X₁)[1] # Current refinement
    u1ref, u2ref = split_solution(Xref)[1] # Reference solution
    sol = vcat(u1, u2);   sol_ref = vcat(u1ref, u2ref)
    L²Error[i]  = compute_l2_error(sol, sol_ref, 𝐇, (sqrt(length(u1)), sqrt(length(u1ref))))
    println("Done N = "*string(N))
  end
end

h = 1 ./(𝒩 .- 1);
rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
@show L²Error
@show rate

###############################################
# Plot the solution and the convergence rates #
###############################################
u1,u2 = split_solution(X₁)[1];
𝐪𝐫 = generate_2d_grid((𝒩[end], 𝒩[end]));
xy = vec(Ω.(𝐪𝐫));
plt1 = scatter(Tuple.(xy), zcolor=vec(u1), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt1, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
title!(plt1, "Horizontal Displacement (App. Sol.)")
plt2 = scatter(Tuple.(xy), zcolor=vec(u2), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.1, label="");
scatter!(plt2, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1)
title!(plt2, "Vertical Displacement (App. Sol.)") =#
#
u1ref,u2ref = split_solution(Xref)[1];
𝐪𝐫 = generate_2d_grid((N,N));
xy = vec(Ω.(𝐪𝐫));
plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
title!(plt3, "Horizontal Displacement (Ref. Sol.)")
plt4 = scatter(Tuple.(xy), zcolor=vec(u2ref), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.1, label="");
scatter!(plt4, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1)
title!(plt4, "Vertical Displacement (Ref. Sol.)")
#
# plt5 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
# plot!(plt5, h,  h.^4, label="O(h⁴)", lw=1, xlabel="h", ylabel="L² Error");
#
plt6 = scatter(Tuple.(xy), zcolor=σₚ.(xy), colormap=:redsblues, xlabel="x(=q)", ylabel="y(=r)", title="PML Damping Function", label="", ms=2, msw=0.1)
scatter!(plt6, Tuple.([[Lₓ,q] for q in LinRange(0,2,𝒩[end])]), mc=:white, label="x ≥ "*string(Lₓ)*" (PML)")
#
𝐪𝐫 = generate_2d_grid((𝒩[end], 𝒩[end]));
xy = vec(Ω.(𝐪𝐫));
X₀ = vcat(eltocols(vec(𝐔.(xy))), eltocols(vec(𝐑.(xy))), eltocols(vec(𝐕.(xy))), eltocols(vec(𝐖.(xy))), eltocols(vec(𝐐.(xy))));
u0,v0 = split_solution(X₀)[1];
plt7 = scatter(Tuple.(xy), zcolor=vec(u0), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt7, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);
title!(plt7, "Horizontal Displacement (Init. Cond.)")
plt8 = scatter(Tuple.(xy), zcolor=vec(v0), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.1, label="");
scatter!(plt8, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],𝒩[end])]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1)
title!(plt8, "Vertical Displacement (Init. Cond.)")

plt13 = plot(plt1, plt3, layout=(2,1), size=(800,800));
plt24 = plot(plt2, plt4, layout=(2,1), size=(800,800));
plt78 = plot(plt7, plt8, layout=(2,1), size=(800,800)); 

#=savefig(plt13, "./Images/PML/1-layer/horizontal-disp.png")
savefig(plt24, "./Images/PML/1-layer/vertical-disp.png")
savefig(plt7, "./Images/PML/1-layer/init-cond-1.png")
savefig(plt8, "./Images/PML/1-layer/init-cond-2.png")
savefig(plt5, "./Images/PML/1-layer/rate.png")
savefig(plt6, "./Images/PML/1-layer/damping-function.png")=#
