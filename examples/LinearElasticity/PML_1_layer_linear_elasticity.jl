###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine
using LoopVectorization

# Define the domain
c₀(r) = @SVector [0.0, r]
c₁(q) = @SVector [1.1*q, 0.0 + 0.1*sin(π*q)]
c₂(r) = @SVector [1.1, r]
c₃(q) = @SVector [1.1*q, 1.0]
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
const Lₓ = 1.0
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
  # Zx = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₁₁.(𝐱𝐲)))), spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲)))))
  # Zy = blockdiag(spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₃₃.(𝐱𝐲)))), spdiagm(vec(sqrt.(ρ.(𝐱𝐲).*c₂₂.(𝐱𝐲)))))  
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
Stiffness matrix for the PML problem
"""
function 𝐊ᴾᴹᴸ(𝐪𝐫)
  detJ(x) = (det∘J)(x,Ω)    
  P = P2R.(𝒫, Ω, 𝐪𝐫) # Elasticity Bulk  
  PML =  P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ, Ω, 𝐪𝐫) # PML Bulk 
  # For the first derivative difference operators of the auxiliary variables
  m,n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  Dq, Dr = sbp_2d.D1
  # Bulk stiffness matrices
  𝐏 = Pᴱ(Dᴱ(P))  
  𝐏ᴾᴹᴸ = Pᴾᴹᴸ(Dᴾᴹᴸ(PML))
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Z = zero(Id)  
  xy = Ω.(𝐪𝐫)
  σ = I(2) ⊗ spdiagm(vec(σₚ.(xy)))  
  ρσ = I(2) ⊗ spdiagm(vec(ρ.(xy).*σₚ.(xy)))
  ρσα = α*ρσ
  # Determinant of the Jacobian Matrix
  detJ1 = [1,1] ⊗ vec(detJ.(𝐪𝐫))  
  SJr₀ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω))*J⁻¹s([q,0.0], Ω, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m), m)
  SJq₀ = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω))*J⁻¹s([0.0,q], Ω, [-1,0])) for q in LinRange(0,1,m)]), m)
  SJrₙ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω))*J⁻¹s([q,1.0], Ω, [0,1])) for q in LinRange(0,1,m)])⊗E1(m,m,m), m)
  SJqₙ = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω))*J⁻¹s([1.0,q], Ω, [1,0])) for q in LinRange(0,1,m)]), m)
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
  # Get the traction operators of the elasticity part 
  𝐓q₀ = Tᴱ(P, Ω, [-1,0]).A
  𝐓r₀ = Tᴱ(P, Ω, [0,-1]).A
  𝐓qₙ = Tᴱ(P, Ω, [1,0]).A 
  𝐓rₙ = Tᴱ(P, Ω, [0,1]).A 
  # Get the traction operator of the PML part
  Zx = blockdiag(spdiagm(vec(sqrt.(ρ.(xy).*c₁₁.(xy)))), spdiagm(vec(sqrt.(ρ.(xy).*c₃₃.(xy)))))
  Zy = blockdiag(spdiagm(vec(sqrt.(ρ.(xy).*c₃₃.(xy)))), spdiagm(vec(sqrt.(ρ.(xy).*c₂₂.(xy)))))  
  𝐓ᴾᴹᴸq₀, 𝐓ᴾᴹᴸqₙ, 𝐓ᴾᴹᴸr₀, 𝐓ᴾᴹᴸrₙ  = Tᴾᴹᴸ(PML, (Zx, Zy), σₚ, Ω, 𝐪𝐫)  
  # Norm matrices
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm  
  # Get the overall traction operator  
  𝐓𝐪₀ = spdiagm(detJ1.^-1)*([-(I(2)⊗𝐇q₀)*SJq₀*𝐓q₀   Z    Z   Z   Z] + 𝐓ᴾᴹᴸq₀)
  𝐓𝐪ₙ = spdiagm(detJ1.^-1)*([(I(2)⊗𝐇qₙ)*SJqₙ*𝐓qₙ  Z   Z    Z   Z] + 𝐓ᴾᴹᴸqₙ)
  𝐓𝐫₀ = spdiagm(detJ1.^-1)*([-(I(2)⊗𝐇r₀)*SJr₀*𝐓r₀   Z  Z   Z   Z] + 𝐓ᴾᴹᴸr₀)
  𝐓𝐫ₙ = spdiagm(detJ1.^-1)*([(I(2)⊗𝐇rₙ)*SJrₙ*𝐓rₙ  Z  Z   Z   Z] + 𝐓ᴾᴹᴸrₙ)
  # The final system  
  zbT = spzeros(Float64, 2m^2, 10n^2)
  zbB = spzeros(Float64, 6m^2, 10n^2)
  Σ - [zbT;   𝐓𝐪₀ + 𝐓𝐪ₙ + 𝐓𝐫₀ + 𝐓𝐫ₙ;   zbB]
end 

"""
Inverse of the mass matrix
"""
function 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  ρᵥ = I(2)⊗spdiagm(vec(1 ./ρ.(Ω.(𝐪𝐫))))
  blockdiag(Id, ρᵥ, Id, Id, Id)
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
Initial conditions
"""
𝐔(x) = @SVector [exp(-20*((x[1]-0.5)^2 + (x[2]-0.5)^2)), -exp(-20*((x[1]-0.5)^2 + (x[2]-0.5)^2))]
𝐑(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]

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

const Δt = 1e-3
tf = 1e-3
ntime = ceil(Int, tf/Δt)
N = 21
𝐪𝐫 = generate_2d_grid((N,N));
xy = Ω.(𝐪𝐫);
stima = 𝐊ᴾᴹᴸ(𝐪𝐫);
massma = 𝐌ᴾᴹᴸ⁻¹(𝐪𝐫);
# Begin time loop
let
  t = 0.0
  X₀ = vcat(eltocols(vec(𝐔.(xy))), eltocols(vec(𝐑.(xy))), eltocols(vec(𝐕.(xy))), eltocols(vec(𝐖.(xy))), eltocols(vec(𝐐.(xy))));
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  M = massma*stima
  @gif for i=1:ntime
  # for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    X₀ = RK4_1!(M, sol)    
    t += Δt    
    (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))

    # Plotting part for 
    u1ref,u2ref = split_solution(X₀, N)[1];
    𝐪𝐫 = generate_2d_grid((N,N));
    xy = vec(Ω.(𝐪𝐫));
    plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);  
    title!(plt3, "Time t="*string(t))
  # end
  end  every 100      
  global Xref = X₀
end  

u1ref,u2ref = split_solution(Xref,N)[1];
xy = vec(xy)
plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=4, msw=0.1);
title!(plt3, "Horizontal Displacement")
plt4 = scatter(Tuple.(xy), zcolor=vec(u2ref), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.1, label="");
scatter!(plt4, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=4, msw=0.1)
title!(plt4, "Vertical Displacement")
plt5 = scatter(Tuple.(xy), zcolor=σₚ.(xy), colormap=:redsblues, xlabel="x(=q)", ylabel="y(=r)", title="PML Damping Function", label="", ms=2, msw=0.1)
scatter!(plt5, Tuple.([[Lₓ,q] for q in LinRange(0,2,N)]), mc=:white, label="x ≥ "*string(Lₓ)*" (PML)")

X₀ = vcat(eltocols(vec(𝐔.(xy))), eltocols(vec(𝐑.(xy))), eltocols(vec(𝐕.(xy))), eltocols(vec(𝐖.(xy))), eltocols(vec(𝐐.(xy))));
u0,v0 = split_solution(X₀,N)[1];
plt6 = scatter(Tuple.(xy), zcolor=vec(u0), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt6, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=4, msw=0.1);
title!(plt6, "Horizontal Displacement (Init. Cond.)")
plt7 = scatter(Tuple.(xy), zcolor=vec(v0), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.1, label="");
scatter!(plt7, Tuple.([[Lₓ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₓ)*" (PML)", markercolor=:white, markersize=4, msw=0.1)
title!(plt7, "Vertical Displacement (Init. Cond.)")

plt34 = plot(plt3, plt4, layout=(2,1), size=(800,800));
plt78 = plot(plt6, plt7, layout=(2,1), size=(800,800)); 