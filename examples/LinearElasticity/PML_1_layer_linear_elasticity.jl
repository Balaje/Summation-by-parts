###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
# -------------- CORRECTION WORK IN PROGRESS.... -----------------
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine
using LoopVectorization

# Define the domain
c₀(r) = @SVector [0.0, 4.4π*r]
c₁(q) = @SVector [4.4π*q, 0.0 + 0.0*sin(π*q)]
c₂(r) = @SVector [4.4π + 0.0*sin(π*r), 4.4π*r]
c₃(q) = @SVector [4.4π*q, 4.4π - 0.0*sin(π*q)]
domain = domain_2d(c₀, c₁, c₂, c₃)

"""
The Lamé parameters μ, λ
"""
λ(x) = 2.0
μ(x) = 1.0

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
const Lᵥ = 4π
const Lₕ = 4π
const δ = 0.1*Lᵥ
const σ₀ᵛ = 0*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ Lᵥ) || x[1] > Lᵥ)
    return σ₀ᵛ*((x[1] - Lᵥ)/δ)^3  
  else
    return 0.0
  end
end

function σₕ(x)
  if((x[2] ≈ Lₕ) || x[2] > Lₕ)
    return σ₀ʰ*((x[2] - Lₕ)/δ)^3  
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
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁(x) + σₕ(x)*c₁₁(x) 0 0 0; 0 -σᵥ(x)*c₃₃(x) + σₕ(x)*c₃₃(x) 0 0; 0 0 σᵥ(x)*c₃₃(x) - σₕ(x)*c₃₃(x)  0; 0 0 0 σᵥ(x)*c₂₂(x) - σₕ(x)*c₂₂(x)];

"""
Density function 
"""
ρ(x) = 1.0

"""
Material velocity tensors
"""
Z₁(x) = @SMatrix [√(c₁₁(x)/ρ(x))  0;  0 √(c₃₃(x)/ρ(x))]
Z₂(x) = @SMatrix [√(c₃₃(x)/ρ(x))  0;  0 √(c₂₂(x)/ρ(x))]

"""
Function to obtain the PML stiffness matrix
"""
function 𝐊ₚₘₗ(𝒫, 𝒫ᴾᴹᴸ, Z₁₂, 𝛀::DiscreteDomain, 𝐪𝐫)
  Ω(qr) = S(qr, 𝛀.domain);
  Z₁, Z₂ = Z₁₂

  Pqr = P2R.(𝒫,Ω,𝐪𝐫);
  Pᴾᴹᴸqr = P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ, Ω, 𝐪𝐫);
  𝐏 = Pᴱ(Pqr).A;
  𝐏ᴾᴹᴸ₁, 𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr).A;

  # Obtain some quantities on the grid points
  𝐙₁ = 𝐙(Z₁, Ω, 𝐪𝐫);  𝐙₂ = 𝐙(Z₂, Ω, 𝐪𝐫);
  𝛔ᵥ = I(2) ⊗ spdiagm(σᵥ.(Ω.(vec(𝐪𝐫))));  𝛔ₕ = I(2) ⊗ spdiagm(σₕ.(Ω.(vec(𝐪𝐫))));
  𝛒 = I(2) ⊗ spdiagm(ρ.(Ω.(vec(𝐪𝐫))))
  # Get the transformed gradient
  Jqr = J⁻¹.(𝐪𝐫, Ω);
  J_vec = get_property_matrix_on_grid(Jqr, 2);
  J_vec_diag = [I(2)⊗spdiagm(vec(p)) for p in J_vec];
  # Get the 2d SBP operators on the reference grid
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  Dq, Dr = sbp_2d.D1
  Dqr = [I(2)⊗Dq, I(2)⊗Dr]
  Dx, Dy = J_vec_diag*Dqr;
  # Bulk Jacobian
  𝐉 = Jb(𝛀, 𝐪𝐫)
  𝐉⁻¹ = 𝐉\(I(size(𝐉,1))) 

  # Surface Jacobian Matrices
  SJr₀, SJq₀, SJrₙ, SJqₙ =  𝐉⁻¹*Js(𝛀, [0,-1];  X=I(2)), 𝐉⁻¹*Js(𝛀, [-1,0];  X=I(2)), 𝐉⁻¹*Js(𝛀, [0,1];  X=I(2)), 𝐉⁻¹*Js(𝛀, [1,0];  X=I(2))

  # Equation 1: ∂u/∂t = p
  EQ1 = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m)⊗I(m))

  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [E1(2,i,(6,6)) for i=1:6];
  eq2s = [(𝐉⁻¹*𝐏)+α*𝛒*(𝛔ᵥ+𝛔ₕ)-𝛒*𝛔ᵥ*𝛔ₕ, -𝛒*(𝛔ᵥ+𝛔ₕ), 𝐉⁻¹*𝐏ᴾᴹᴸ₁, 𝐉⁻¹*𝐏ᴾᴹᴸ₂, -α*𝛒*(𝛔ᵥ+𝛔ₕ)+𝛒*𝛔ᵥ*𝛔ₕ, 𝛒*𝛔ᵥ*𝛔ₕ];
  EQ2 = sum(es .⊗ eq2s);

  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [E1(3,i,(6,6)) for i=[1,3]];
  eq3s = [Dx, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ᵥ)];
  EQ3 = sum(es .⊗ eq3s);

  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [E1(4,i,(6,6)) for i=[1,4]]
  eq4s = [Dy, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ₕ)]
  EQ4 = sum(es .⊗ eq4s)

  # Equation 5 ∂q/∂t = α(u-q)
  es = [E1(5,i,(6,6)) for i=[1,5]]
  eq5s = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ5 = sum(es .⊗ eq5s)

  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [E1(6,i,(6,6)) for i=[1,5,6]]
  eq6s = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ6 = sum(es .⊗ eq6s)

  # PML characteristic boundary conditions
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ = Pqr, Pᴾᴹᴸqr, 𝐙₁, 𝐙₂, 𝛔ᵥ, 𝛔ₕ;
  χq₀, χr₀, χqₙ, χrₙ = χᴾᴹᴸ(PQRᵪ, 𝛀, [-1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,-1]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹ = (fill(SJq₀,6).*fill((I(2)⊗𝐇q₀⁻¹),6));
  SJ_𝐇qₙ⁻¹ = (fill(SJqₙ,6).*fill((I(2)⊗𝐇qₙ⁻¹),6));
  SJ_𝐇r₀⁻¹ = (fill(SJr₀,6).*fill((I(2)⊗𝐇r₀⁻¹),6));
  SJ_𝐇rₙ⁻¹ = (fill(SJrₙ,6).*fill((I(2)⊗𝐇rₙ⁻¹),6));
  SAT = sum(es.⊗(SJ_𝐇q₀⁻¹.*χq₀)) + sum(es.⊗(SJ_𝐇qₙ⁻¹.*χqₙ)) + sum(es.⊗(SJ_𝐇r₀⁻¹.*χr₀)) + sum(es.⊗(SJ_𝐇rₙ⁻¹.*χrₙ));

  # The SBP-SAT Formulation
  bulk = (EQ1 + EQ2 + EQ3 + EQ4 + EQ5 + EQ6);
  bulk - SAT;
end

"""
Inverse of the mass matrix
"""
function 𝐌⁻¹ₚₘₗ(𝛀::DiscreteDomain, 𝐪𝐫)
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Ω(qr) = S(qr, 𝛀.domain);
  ρᵥ = I(2)⊗spdiagm(vec(1 ./ρ.(Ω.(𝐪𝐫))))
  blockdiag(Id, ρᵥ, Id, Id, Id, Id)
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
function split_solution(X, N)  
  res = splitdimsview(reshape(X, (N^2, 12)))
  u1, u2 = res[1:2]
  p1, p2 = res[3:4]
  v1, v2 = res[5:6]
  w1, w2 = res[7:8]
  q1, q2 = res[9:10]
  r1, r2 = res[11:12]
  (u1,u2), (p1,p2), (v1, v2), (w1,w2), (q1,q2), (r1,r2)
end

"""
Initial conditions
"""
𝐔(x) = @SVector [exp(-2*((x[1]-2.2π)^2 + (x[2]-2.2π)^2)), -exp(-2*((x[1]-2.2π)^2 + (x[2]-2.2π)^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

const Δt = 1e-3
tf = 10.0
ntime = ceil(Int, tf/Δt)
N = 81;
𝛀 = DiscreteDomain(domain, (N,N));
Ω(qr) = S(qr, 𝛀.domain);
𝐪𝐫 = generate_2d_grid((N,N));
xy = Ω.(𝐪𝐫);
stima = 𝐊ₚₘₗ(𝒫, 𝒫ᴾᴹᴸ, (Z₁, Z₂), 𝛀, 𝐪𝐫);
massma = 𝐌⁻¹ₚₘₗ(𝛀, 𝐪𝐫)

# Begin time loop
let
  t = 0.0
  X₀ = vcat(eltocols(vec(𝐔.(xy))), eltocols(vec(𝐏.(xy))), eltocols(vec(𝐕.(xy))), eltocols(vec(𝐖.(xy))), eltocols(vec(𝐐.(xy))), eltocols(vec(𝐑.(xy))));
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
    plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.([[Lᵥ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lᵥ)*" (PML)", markercolor=:white, markersize=2, msw=0.1);  
    title!(plt3, "Time t="*string(t))
  # end
  end  every 50      
  global Xref = X₀
end  

# Plotting
u1ref,u2ref = split_solution(Xref,N)[1];
xy = vec(xy)
plt3 = scatter(Tuple.(xy), zcolor=vec(u1ref), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.([[Lₕ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₕ)*" (PML)", markercolor=:white, markersize=4, msw=0.1);
title!(plt3, "Horizontal Displacement")
plt4 = scatter(Tuple.(xy), zcolor=vec(u2ref), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.1, label="");
scatter!(plt4, Tuple.([[Lₕ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), label="x ≥ "*string(Lₕ)*" (PML)", markercolor=:white, markersize=4, msw=0.1)
title!(plt4, "Vertical Displacement")

plt34 = plot(plt3, plt4, layout=(2,1), size=(800,800));

plt5 = scatter(Tuple.(xy), zcolor=σₕ.(xy), colormap=:turbo, xlabel="x(=q)", ylabel="y(=r)", title="PML Damping Function", label="", ms=4, msw=0.1)
scatter!(plt5, Tuple.([[q,Lᵥ] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), mc=:white, label="x ≥ "*string(Lᵥ)*" (PML)")
plt6 = scatter(Tuple.(xy), zcolor=σᵥ.(xy), colormap=:turbo, xlabel="x(=q)", ylabel="y(=r)", title="PML Damping Function", label="", ms=4, msw=0.1)
scatter!(plt6, Tuple.([[Lₕ,q] for q in LinRange(Ω([0.0,0.0])[2],Ω([1.0,1.0])[2],N)]), mc=:white, label="x ≥ "*string(Lᵥ)*" (PML)")
