include("2d_elasticity_problem.jl")

using SplitApplyCombine
using LoopVectorization

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
f(q) = 0.2*exp(-4*4π*(q-0.5)^2)
cᵢ¹(q) = 4π*[q, 1 + f(q)];
cᵢ²(r) = 4π*[2.0 + f(r), r];
# Define the rest of the boundary
c₀¹(r) = 4π*[0.0 , 1+r]; # Left boundary
c₁¹(q) = cᵢ¹(q) # Bottom boundary. (Interface 1)
c₂¹(r) = 4π*[1.0, 1+r]; # Right boundary
c₃¹(q) = 4π*[q, 2.0 - f(q)]; # Top boundary
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = 4π*[0.0, r]; # Left boundary
c₁²(q) = 4π*[q, 0.0]; # Bottom boundary. 
c₂²(r) = cᵢ²(r); # Right boundary (Interface 2)
c₃²(q) = cᵢ¹(q); # Top boundary. (Interface 1)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₂(qr) = S(qr, domain₂)
c₀³(r) = cᵢ²(r) # Left boundary (Interface 2)
c₁³(q) = 4π*[1.0 + q, 0.0] # Bottom boundary
c₂³(r) = 4π*[2.0 - f(r), r] # Right boundary
c₃³(q) = 4π*[1.0 + q, 1.0] # Top boundary
domain₃ = domain_2d(c₀³, c₁³, c₂³, c₃³)

## Define the material properties on the physical grid
"""
The Lamé parameters μ, λ
"""
λ¹(x) = 2.0
μ¹(x) = 1.0
λ²(x) = 1.0
μ²(x) = 0.5
λ³(x) = 0.5
μ³(x) = 0.25

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

c₁₁³(x) = 2*μ³(x)+λ³(x)
c₂₂³(x) = 2*μ³(x)+λ³(x)
c₃₃³(x) = μ³(x)
c₁₂³(x) = λ³(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫¹(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫²(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];
𝒫³(x) = @SMatrix [c₁₁³(x) 0 0 c₁₂³(x); 0 c₃₃³(x) c₃₃³(x) 0; 0 c₃₃³(x) c₃₃³(x) 0; c₁₂³(x) 0 0 c₂₂³(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ¹(∇u,x) = 𝒫¹(x)*∇u
σ²(∇u,x) = 𝒫²(x)*∇u
σ³(∇u,x) = 𝒫³(x)*∇u

"""
Density function 
"""
ρ¹(x) = 1.0
ρ²(x) = 1.0
ρ³(x) = 1.0

"""
Stiffness matrix function
"""
function 𝐊3!(𝒫, 𝛀::Tuple{DiscreteDomain, DiscreteDomain, DiscreteDomain},  𝐪𝐫)
  𝒫¹, 𝒫², 𝒫³ = 𝒫
  𝛀₁, 𝛀₂, 𝛀₃ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  Ω₃(qr) = S(qr, 𝛀₃.domain)
  @assert 𝛀₁.mn == 𝛀₂.mn == 𝛀₃.mn "Grid size need to be equal"
  (size(𝐪𝐫) != 𝛀₁.mn) && begin
    @warn "Grid not same size. Using the grid size in DiscreteDomain and overwriting the reference grid.."
    𝐪𝐫 = generate_2d_grid(𝛀.mn)
  end
  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Pqr₁) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹, 𝐓r₀¹, 𝐓qₙ¹, 𝐓rₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [-1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,-1]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,1]; X=I(2)).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Pqr₂) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀², 𝐓r₀², 𝐓qₙ², 𝐓rₙ² = Tᴱ(Pqr₂, 𝛀₂, [-1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,-1]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,1]; X=I(2)).A 

  # Get the bulk and the traction operator for the 3rd layer
  detJ₃(x) = (det∘J)(x, Ω₃)    
  Pqr₃ = P2R.(𝒫³, Ω₃, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₃ = Pᴱ(Pqr₃) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀³, 𝐓r₀³, 𝐓qₙ³, 𝐓rₙ³ = Tᴱ(Pqr₃, 𝛀₃, [-1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,-1]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,1]; X=I(2)).A 
  
  # Get the norm matrices (Same for all layers)
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
  # Determinants of the transformation
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫)
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫) 
  𝐉₃ = Jb(𝛀₃, 𝐪𝐫) 
  𝐉 = blockdiag(𝐉₁, 𝐉₂, 𝐉₃)   
  
  # Surface Jacobians of the outer boundaries
  # - Layer 1  
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ = Js(𝛀₁, [0,-1]; X=I(2)), Js(𝛀₁, [-1,0]; X=I(2)), Js(𝛀₁, [0,1]; X=I(2)), Js(𝛀₁, [1,0]; X=I(2))   
  # - Layer 2
  SJr₀², SJq₀², _, _ = Js(𝛀₂, [0,-1]; X=I(2)), Js(𝛀₂, [-1,0]; X=I(2)), Js(𝛀₂, [0,1]; X=I(2)), Js(𝛀₂, [1,0]; X=I(2))   
  # - Layer 3
  SJr₀³, _, SJrₙ³, SJqₙ³ = Js(𝛀₃, [0,-1]; X=I(2)), Js(𝛀₃, [-1,0]; X=I(2)), Js(𝛀₃, [0,1]; X=I(2)), Js(𝛀₃, [1,0]; X=I(2))   

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁.A, 𝐏₂.A, 𝐏₃.A)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀⁻¹)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ⁻¹)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ⁻¹)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀⁻¹)*SJq₀²*(𝐓q₀²) + -(I(2)⊗𝐇r₀⁻¹)*SJr₀²*(𝐓r₀²), 
                 (I(2)⊗𝐇qₙ⁻¹)*SJqₙ³*(𝐓qₙ³) + -(I(2)⊗𝐇r₀⁻¹)*SJr₀³*(𝐓r₀³) + (I(2)⊗𝐇rₙ⁻¹)*SJrₙ³*(𝐓rₙ³))
  𝐓rᵢ¹ = blockdiag(𝐓r₀¹, 𝐓rₙ²)            
  𝐓qᵢ² = blockdiag(𝐓qₙ², 𝐓q₀³)            
  
  # Get the Interface SAT for Conforming Interface
  B̂₁, B̃₁, 𝐇⁻¹₁ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=I(2))
  B̂₂, B̃₂, 𝐇⁻¹₂ = SATᵢᴱ(𝛀₂, 𝛀₃, [1; 0], [-1; 0], ConformingInterface(); X=I(2))
  
  h = 1/(m-1)
  ζ₀ = 40/h
  𝐓ᵢ¹ = blockdiag((I(2)⊗𝐇⁻¹₁)*(0.5*B̂₁*𝐓rᵢ¹ - 0.5*𝐓rᵢ¹'*B̂₁ - ζ₀*B̃₁), zero(𝐏₃.A))
  𝐓ᵢ² = blockdiag(zero(𝐏₁.A), (I(2)⊗𝐇⁻¹₂)*(-0.5*B̂₂*𝐓qᵢ² + 0.5*𝐓qᵢ²'*B̂₂ - ζ₀*B̃₂))
    
  𝐉\(𝐏 - 𝐓 - 𝐓ᵢ¹ - 𝐓ᵢ²)
end
  
m = 81;
𝐪𝐫 = generate_2d_grid((m,m))
𝛀₁ = DiscreteDomain(domain₁, (m,m))
𝛀₂ = DiscreteDomain(domain₂, (m,m))
𝛀₃ = DiscreteDomain(domain₃, (m,m))
Ω₁(qr) = S(qr, 𝛀₁.domain)
Ω₂(qr) = S(qr, 𝛀₂.domain)
Ω₃(qr) = S(qr, 𝛀₃.domain)
𝐱𝐲₁ = Ω₁.(𝐪𝐫)
𝐱𝐲₂ = Ω₂.(𝐪𝐫)
𝐱𝐲₃ = Ω₃.(𝐪𝐫)
stima3 = 𝐊3!((𝒫¹, 𝒫², 𝒫³), (𝛀₁, 𝛀₂, 𝛀₃), 𝐪𝐫);
massma3 = blockdiag((I(2)⊗spdiagm(vec(ρ¹.(𝐱𝐲₁)))), (I(2)⊗spdiagm(vec(ρ².(𝐱𝐲₂)))), (I(2)⊗spdiagm(vec(ρ³.(𝐱𝐲₃)))))

const Δt = 1e-3
tf = 5.0
ntime = ceil(Int, tf/Δt)

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
Extract solution vector from the raw vector
"""
function get_sol_vector_from_raw_vector(sol, m)
  ((reshape(sol[1:m^2], (m,m)), reshape(sol[m^2+1:2m^2], (m,m))), 
  (reshape(sol[2m^2+1:3m^2], (m,m)), reshape(sol[3m^2+1:4m^2], (m,m))),
  (reshape(sol[4m^2+1:5m^2], (m,m)), reshape(sol[5m^2+1:6m^2], (m,m))))
end

U₀(x) = @SVector [exp(-((x[1]-2π)^2 + (x[2]-6π)^2)), -exp(-((x[1]-2π)^2 + (x[2]-6π)^2))]
V₀(x) = @SVector [0.0,0.0]

# Begin time loop
let
  t = 0.0
  X₀ = vcat(eltocols(vec(U₀.(𝐱𝐲₁))), eltocols(vec(U₀.(𝐱𝐲₂))), eltocols(vec(U₀.(𝐱𝐲₃))));
  Y₀ = vcat(eltocols(vec(V₀.(𝐱𝐲₁))), eltocols(vec(V₀.(𝐱𝐲₂))), eltocols(vec(V₀.(𝐱𝐲₃))));
  global Z₀ = vcat(X₀, Y₀)
  global maxvals = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma3\stima3
  K = [zero(M) I(size(M,1)); M zero(M)]
  # @gif for i=1:ntime
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(K, sol)    
    t += Δt        
    (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[1];
    u1ref₂,u2ref₂ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[2];
    u1ref₃,u2ref₃ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[3];
    
    #=  plt3 = scatter(Tuple.(𝐱𝐲₁ |> vec), zcolor=vec(u1ref₁), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
    scatter!(plt3, Tuple.(𝐱𝐲₂ |> vec), zcolor=vec(u1ref₂), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
    scatter!(plt3, Tuple.(𝐱𝐲₃ |> vec), zcolor=vec(u1ref₃), colormap=:redsblues, ylabel="y(=r)", markersize=2, msw=0.01, label="");
    scatter!(plt3, Tuple.([Ω₁([q,0.0]) for q in LinRange(0,1,m)]), label="", msw=0.01, ms=2)
    scatter!(plt3, Tuple.([Ω₃([0.0,r]) for r in LinRange(0,1,m)]), label="", msw=0.01, ms=2, right_margin=20*Plots.mm)
    title!(plt3, "Time t="*string(t)) =#

    maxvals[i] = max(maximum(abs.(u1ref₁)), maximum(abs.(u1ref₂)), maximum(abs.(u1ref₃)))
  end
  # end  every 100 
end  

u1ref₁,u2ref₁ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[1];
u1ref₂,u2ref₂ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[2];
u1ref₃,u2ref₃ = get_sol_vector_from_raw_vector(Z₀[1:6m^2], m)[3];
plt3 = scatter(Tuple.(𝐱𝐲₁ |> vec), zcolor=vec(u1ref₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.(𝐱𝐲₂ |> vec), zcolor=vec(u1ref₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.(𝐱𝐲₃ |> vec), zcolor=vec(u1ref₃), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt3, Tuple.([Ω₁([q,0.0]) for q in LinRange(0,1,m)]), label="", msw=0.01, ms=2)
scatter!(plt3, Tuple.([Ω₃([0.0,r]) for r in LinRange(0,1,m)]), label="", msw=0.01, ms=2, right_margin=10*Plots.mm, size=(800,800))

plt4 = plot(LinRange(0,tf,ntime), maxvals, lw=2, label="", xlabel="t", ylabel="||U||∞")