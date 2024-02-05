# include("2d_elasticity_problem.jl");
using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Plots

using LoopVectorization
using SplitApplyCombine

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Function to generate the stiffness matrices
"""
function 𝐊!(Pqr, 𝛀::DiscreteDomain, 𝐪𝐫)
  Ω(qr) = S(qr, 𝛀.domain)
  detJ(x) = (det∘J)(x,Ω)    

  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(n)
  sbp_r = SBP_1_2_CONSTANT_0_1(m)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  
  # Get the material property matrix evaluated at grid points    
  # Pqr = P2R.(𝒫,Ω,𝐪𝐫) 

  # Elasticity bulk differential operator  
  𝐏 = Pᴱ(Pqr).A 

  # Elasticity Traction Operators
  𝐓q₀, 𝐓r₀, 𝐓qₙ, 𝐓rₙ = Tᴱ(Pqr, 𝛀, [-1,0]).A, Tᴱ(Pqr, 𝛀, [0,-1]).A, Tᴱ(Pqr, 𝛀, [1,0]).A, Tᴱ(Pqr, 𝛀, [0,1]).A   

  # The surface Jacobians on the boundary
  SJr₀, SJq₀, SJrₙ, SJqₙ = Js(𝛀, [0,-1];  X=I(2)), Js(𝛀, [-1,0];  X=I(2)), Js(𝛀, [0,1];  X=I(2)), Js(𝛀, [1,0];  X=I(2))   
  
  # The norm-inverse on the boundary
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
  # Bulk Jacobian
  𝐉 = Jb(𝛀, 𝐪𝐫)
  𝐉⁻¹ = 𝐉\I(size(𝐉,1))

  SAT = (-(I(2) ⊗ 𝐇q₀⁻¹)*SJq₀*(𝐓q₀) + (I(2) ⊗ 𝐇qₙ⁻¹)*SJqₙ*(𝐓qₙ) -(I(2) ⊗ 𝐇r₀⁻¹)*SJr₀*(𝐓r₀) + (I(2) ⊗ 𝐇rₙ⁻¹)*SJrₙ*(𝐓rₙ))

  # The SBP-SAT Formulation    
  𝐉⁻¹*(𝐏 - SAT)
end


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
Function to split the solution into the corresponding variables
"""
function split_solution(X, MN)  
  M,N = MN
  res = splitdimsview(reshape(X, (M*N, 4)))
  u1, u2 = res[1:2]
  (u1,u2)
end

using MAT
vars = matread("./examples/MarmousiModel/marmousi2_downsampled_10.mat");
X = vars["X_e"]
Z = vars["Z_e"]
x = X[1,:]
z = Z[:,1]
m, n = size(X);
XZ = [[X[i,j], Z[i,j]] for i=1:m, j=1:n] # X-Z coordinates from data

## Define the physical domain
c₀(r) = @SVector [x[1], z[1] + (z[end]-z[1])*r] # Left boundary 
c₁(q) = @SVector [x[1] + (x[end]-x[1])*q, z[1]] # Bottom boundary
c₂(r) = @SVector [x[end], z[1] + (z[end]-z[1])*r] # Right boundary
c₃(q) = @SVector [x[1] + (x[end]-x[1])*q, z[end]] # Top boundary
domain = domain_2d(c₀, c₁, c₂, c₃)
𝛀 = DiscreteDomain(domain, (n,m));
Ω(qr) = S(qr, 𝛀.domain);

𝐪𝐫 = generate_2d_grid((n,m));
using Test
@test Ω.(𝐪𝐫) ≈ XZ;

##### ##### ##### ##### ##### ##### ##### ##### 
#   Build the material properties function    #
##### ##### ##### ##### ##### ##### ##### #####
function Pt(𝒫, 𝒮, qr)    
  invJ = J⁻¹(qr, 𝒮)
  detJ = (det∘J)(qr, 𝒮)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫*S)*detJ
end

vp = vars["vp_e"];
vs = vars["vs_e"];
rho = vars["rho_e"];
mu = (vs.^2).*rho;
lambda = (vp.^2).*rho - 2*mu;
C₁₁ = C₂₂ = 2*mu + lambda;
C₃₃ = mu;
C₁₂ = lambda;

P = [@SMatrix [C₁₁[i,j] 0 0 C₁₂[i,j]; 0 C₃₃[i,j] C₃₃[i,j] 0; 0 C₃₃[i,j] C₃₃[i,j] 0; C₁₂[i,j] 0  0 C₂₂[i,j]] for i=1:m, j=1:n]
P_t= [Pt(P[i,j], Ω, 𝐪𝐫[i,j]) for i=1:m, j=1:n];

stima = 𝐊!(P_t, 𝛀, 𝐪𝐫);
massma = I(2) ⊗ spdiagm(vec(rho).^-1);
U₀(p) = @SVector [exp(-1e-3*((p[1]-8493.2)^2 + 4*(p[2]-(-1973.42))^2)) + exp(-1e-3*((p[1]-3000)^2 + 4*(p[2]-(-1000))^2)) + exp(-1e-3*((p[1]-14000)^2 + 4*(p[2]-(-3000))^2)), 
                 -exp(-1e-3*((p[1]-8493.2)^2 + 4*(p[2]-(-1973.42))^2)) - exp(-1e-3*((p[1]-3000)^2 + 4*(p[2]-(-1000))^2)) + exp(-1e-3*((p[1]-14000)^2 + 4*(p[2]-(-3000))^2))];
V₀(p) = @SVector [0.0,0.0]

const Δt = 1e-3
tf = 1.0
ntime = ceil(Int, tf/Δt)
let
  t = 0.0
  X₀ = eltocols(vec(U₀.(XZ)))
  Y₀ = eltocols(vec(V₀.(XZ)))
  global Z₀ = vcat(X₀, Y₀)
  global maxvals = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma*stima
  K = [zero(M) I(size(M,1)); M zero(M)]
  # @gif for i=1:ntime
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(K, sol)    
    t += Δt        
    (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = split_solution(Z₀, (m,n))
    # plt3 = scatter(Tuple.(XZ |> vec), zcolor=vec(u1ref₁), colormap=:redsblues, ylabel="y(=r)", markersize=4, msw=0.0, label="", size=(3200,800));  
    # title!(plt3, "Time t="*string(t))

    maxvals[i] = max(maximum(abs.(u1ref₁)), maximum(abs.(u2ref₁)))
  end
  # end  every 10 
end  

using ColorSchemes
u1ref₁,u2ref₁ = split_solution(Z₀, (m,n))
absu1 = sqrt.(u1ref₁.^2 + u2ref₁.^2);
plt3 = heatmap(x, z, reshape(absu1, (m,n)), colormap=:matter, ylabel="y(=r)", label="", size=(800,800), xtickfontsize=18, ytickfontsize=18, bottommargin=12*Plots.mm, topmargin=15*Plots.mm, rightmargin=20*Plots.mm, titlefontsize=20, clims=(0, 0.02));  
xlims!(plt3, (x[1], x[end]))  
ylims!(plt3, (z[1], z[end]))  
title!(plt3, "\$|u(x,y)|\$ at Time t="*string(tf))

plt4 = heatmap(x, z, vp, ylabel="y(=r)", markersize=4, msw=0.0, label="", size=(800,800), xtickfontsize=18, ytickfontsize=18, bottommargin=12*Plots.mm, titlefontsize=18, topmargin=15*Plots.mm, rightmargin=12*Plots.mm);   
xlims!(plt4, (x[1], x[end]))  
ylims!(plt4, (z[1], z[end])) 
title!(plt4, "Density of the material")

plot(plt4, plt3, layout=(2,1), size=(1600,1600), rightmargin=12*Plots.mm)