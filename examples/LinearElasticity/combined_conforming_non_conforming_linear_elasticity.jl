include("2d_elasticity_problem.jl")

using SplitApplyCombine
using LoopVectorization

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
f(q) = 0.1*exp(-4*4π*(q-0.5)^2)
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
c₀³(r) = cᵢ²(r) # Left boundary (Interface 2)
c₁³(q) = 4π*[1.0 + q, f(q)] # Bottom boundary (Interface 3)
c₂³(r) = 4π*[2.0, r] # Right boundary
c₃³(q) = 4π*[1.0 + q, 1.0] # Top boundary
domain₃ = domain_2d(c₀³, c₁³, c₂³, c₃³)
c₀⁴(r) = 4π*[1.0, r-1] # Left boundary
c₁⁴(q) = 4π*[1.0 + q, -1 + f(q)] # Bottom boundary
c₂⁴(r) = 4π*[2.0, r-1] # Right boundary
c₃⁴(q) = 4π*[1.0 + q, f(q)] # Top boundary (Interface 3)
domain₄ = domain_2d(c₀⁴, c₁⁴, c₂⁴, c₃⁴)

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
λ⁴(x) = 0.5
μ⁴(x) = 0.25

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

c₁₁⁴(x) = 2*μ⁴(x)+λ⁴(x)
c₂₂⁴(x) = 2*μ⁴(x)+λ⁴(x)
c₃₃⁴(x) = μ⁴(x)
c₁₂⁴(x) = λ⁴(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫¹(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫²(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];
𝒫³(x) = @SMatrix [c₁₁³(x) 0 0 c₁₂³(x); 0 c₃₃³(x) c₃₃³(x) 0; 0 c₃₃³(x) c₃₃³(x) 0; c₁₂³(x) 0 0 c₂₂³(x)];
𝒫⁴(x) = @SMatrix [c₁₁⁴(x) 0 0 c₁₂⁴(x); 0 c₃₃⁴(x) c₃₃⁴(x) 0; 0 c₃₃⁴(x) c₃₃⁴(x) 0; c₁₂⁴(x) 0 0 c₂₂⁴(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ¹(∇u,x) = 𝒫¹(x)*∇u
σ²(∇u,x) = 𝒫²(x)*∇u
σ³(∇u,x) = 𝒫³(x)*∇u
σ⁴(∇u,x) = 𝒫⁴(x)*∇u

"""
Density function 
"""
ρ¹(x) = 1.0
ρ²(x) = 1.0
ρ³(x) = 1.0
ρ⁴(x) = 1.0

"""
Stiffness matrix function
"""
function 𝐊4!(𝒫, 𝛀::Tuple{DiscreteDomain, DiscreteDomain, DiscreteDomain, DiscreteDomain},  𝐪𝐫)
  𝒫¹, 𝒫², 𝒫³, 𝒫⁴ = 𝒫
  𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄ = 𝛀
  qr₁, qr₂, qr₃, qr₄ = 𝐪𝐫
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  Ω₃(qr) = S(qr, 𝛀₃.domain)
  Ω₄(qr) = S(qr, 𝛀₄.domain)

  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, qr₁) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Pqr₁) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹, 𝐓r₀¹, 𝐓qₙ¹, 𝐓rₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [-1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,-1]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,1]; X=I(2)).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, qr₂) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Pqr₂) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀², 𝐓r₀², 𝐓qₙ², 𝐓rₙ² = Tᴱ(Pqr₂, 𝛀₂, [-1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,-1]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,1]; X=I(2)).A 

  # Get the bulk and the traction operator for the 3rd layer
  detJ₃(x) = (det∘J)(x, Ω₃)    
  Pqr₃ = P2R.(𝒫³, Ω₃, qr₃) # Property matrix evaluated at grid points
  𝐏₃ = Pᴱ(Pqr₃) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀³, 𝐓r₀³, 𝐓qₙ³, 𝐓rₙ³ = Tᴱ(Pqr₃, 𝛀₃, [-1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,-1]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,1]; X=I(2)).A 

  # Get the bulk and the traction operator for the 4th layer
  detJ₄(x) = (det∘J)(x, Ω₄)    
  Pqr₄ = P2R.(𝒫⁴, Ω₄, qr₄) # Property matrix evaluated at grid points
  𝐏₄ = Pᴱ(Pqr₄) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀⁴, 𝐓r₀⁴, 𝐓qₙ⁴, 𝐓rₙ⁴ = Tᴱ(Pqr₄, 𝛀₄, [-1,0]; X=I(2)).A, Tᴱ(Pqr₄, 𝛀₄, [0,-1]; X=I(2)).A, Tᴱ(Pqr₄, 𝛀₄, [1,0]; X=I(2)).A, Tᴱ(Pqr₄, 𝛀₄, [0,1]; X=I(2)).A 
  
  # Get the norm matrices (Same for Layer 2 and Layer 3)
  # Layer 1
  m₁, n₁ = size(qr₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀⁻¹₁, 𝐇qₙ⁻¹₁, _, 𝐇rₙ⁻¹₁ = sbp_2d₁.norm
  # Same for Layer 2 and Layer 3
  (m₂, n₂) = (m₃, n₃) = size(qr₂)
  sbp_q₂ = sbp_q₃ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = sbp_r₃ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = sbp_2d₃ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  (𝐇q₀⁻¹₂, _, 𝐇r₀⁻¹₂, _) = (_, 𝐇qₙ⁻¹₃, _, 𝐇rₙ⁻¹₃) = sbp_2d₂.norm
  # Layer 4
  m₄, n₄ = size(qr₄)
  sbp_q₄ = SBP_1_2_CONSTANT_0_1(m₄)
  sbp_r₄ = SBP_1_2_CONSTANT_0_1(n₄)
  sbp_2d₄ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₄, sbp_r₄)
  𝐇q₀⁻¹₄, 𝐇qₙ⁻¹₄, 𝐇r₀⁻¹₄, _ = sbp_2d₄.norm
  
  # Determinants of the transformation
  𝐉₁ = Jb(𝛀₁, qr₁)
  𝐉₂ = Jb(𝛀₂, qr₂) 
  𝐉₃ = Jb(𝛀₃, qr₃) 
  𝐉₄ = Jb(𝛀₄, qr₄) 
  𝐉 = blockdiag(𝐉₁, 𝐉₂, 𝐉₃, 𝐉₄)   
  
  # Surface Jacobians of the outer boundaries
  # - Layer 1  
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ = Js(𝛀₁, [0,-1]; X=I(2)), Js(𝛀₁, [-1,0]; X=I(2)), Js(𝛀₁, [0,1]; X=I(2)), Js(𝛀₁, [1,0]; X=I(2))   
  # - Layer 2
  SJr₀², SJq₀², _, _ = Js(𝛀₂, [0,-1]; X=I(2)), Js(𝛀₂, [-1,0]; X=I(2)), Js(𝛀₂, [0,1]; X=I(2)), Js(𝛀₂, [1,0]; X=I(2))   
  # - Layer 3
  _, _, SJrₙ³, SJqₙ³ = Js(𝛀₃, [0,-1]; X=I(2)), Js(𝛀₃, [-1,0]; X=I(2)), Js(𝛀₃, [0,1]; X=I(2)), Js(𝛀₃, [1,0]; X=I(2))   
  # - Layer 4
  SJr₀⁴, SJq₀⁴, _, SJqₙ⁴ = Js(𝛀₄, [0,-1]; X=I(2)), Js(𝛀₄, [-1,0]; X=I(2)), Js(𝛀₄, [0,1]; X=I(2)), Js(𝛀₄, [1,0]; X=I(2))   

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁.A, 𝐏₂.A, 𝐏₃.A, 𝐏₄.A)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀⁻¹₁)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ⁻¹₁)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ⁻¹₁)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀⁻¹₂)*SJq₀²*(𝐓q₀²) + -(I(2)⊗𝐇r₀⁻¹₂)*SJr₀²*(𝐓r₀²), 
                (I(2)⊗𝐇qₙ⁻¹₃)*SJqₙ³*(𝐓qₙ³) + (I(2)⊗𝐇rₙ⁻¹₃)*SJrₙ³*(𝐓rₙ³), 
                -(I(2)⊗𝐇q₀⁻¹₄)*SJq₀⁴*(𝐓q₀⁴)  + (I(2)⊗𝐇qₙ⁻¹₄)*SJqₙ⁴*(𝐓qₙ⁴) + -(I(2)⊗𝐇r₀⁻¹₄)*SJr₀⁴*(𝐓r₀⁴) )
  𝐓rᵢ¹ = blockdiag(𝐓r₀¹, 𝐓rₙ²)            
  𝐓qᵢ² = blockdiag(𝐓qₙ², 𝐓q₀³)            
  𝐓rᵢ³ = blockdiag(𝐓r₀³, 𝐓rₙ⁴)            
  
  # Get the Interface SAT for Conforming Interface
  B̂₁, B̃₁, 𝐇⁻¹₁ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], NonConformingInterface(); X=I(2))
  B̂₂, B̃₂, 𝐇⁻¹₂ = SATᵢᴱ(𝛀₂, 𝛀₃, [1; 0], [-1; 0], ConformingInterface(); X=I(2))
  B̂₃, B̃₃, 𝐇⁻¹₃ = SATᵢᴱ(𝛀₃, 𝛀₄, [0; -1], [0; 1], NonConformingInterface(); X=I(2))  
  
  h = 1/(max(m₁,m₂,m₃,m₄)-1)
  ζ₀ = 40/h
  𝐓ᵢ¹ = blockdiag((𝐇⁻¹₁)*(0.5*B̂₁*𝐓rᵢ¹ - 0.5*𝐓rᵢ¹'*B̂₁ - ζ₀*B̃₁), zero(𝐏₃.A), zero(𝐏₄.A))
  𝐓ᵢ² = blockdiag(zero(𝐏₁.A), (I(2)⊗𝐇⁻¹₂)*(-0.5*B̂₂*𝐓qᵢ² + 0.5*𝐓qᵢ²'*B̂₂ - ζ₀*B̃₂), zero(𝐏₄.A))
  𝐓ᵢ³ = blockdiag(zero(𝐏₁.A), zero(𝐏₂.A), (𝐇⁻¹₃)*(0.5*B̂₃*𝐓rᵢ³ - 0.5*𝐓rᵢ³'*B̂₃ - ζ₀*B̃₃))

  𝐉\(𝐏 - 𝐓 - 𝐓ᵢ¹ - 𝐓ᵢ² - 𝐓ᵢ³)
end

############################
# Begin solving the problem

m₁ = 41;
m₂ = 81;
m₃ = 81;
m₄ = 41;
qr₁ = generate_2d_grid((m₁,m₁))
qr₂ = generate_2d_grid((m₂,m₂))
qr₃ = generate_2d_grid((m₃,m₃))
qr₄ = generate_2d_grid((m₄,m₄))
𝛀₁ = DiscreteDomain(domain₁, (m₁,m₁))
𝛀₂ = DiscreteDomain(domain₂, (m₂,m₂))
𝛀₃ = DiscreteDomain(domain₃, (m₃,m₃))
𝛀₄ = DiscreteDomain(domain₄, (m₄,m₄))
Ω₁(qr) = S(qr, 𝛀₁.domain)
Ω₂(qr) = S(qr, 𝛀₂.domain)
Ω₃(qr) = S(qr, 𝛀₃.domain)
Ω₄(qr) = S(qr, 𝛀₄.domain)
xy₁ = Ω₁.(qr₁)
xy₂ = Ω₂.(qr₂)
xy₃ = Ω₃.(qr₃)
xy₄ = Ω₄.(qr₄)
stima4 = 𝐊4!((𝒫¹, 𝒫², 𝒫³, 𝒫⁴), (𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄), (qr₁, qr₂, qr₃, qr₄));
massma4 = blockdiag((I(2)⊗spdiagm(vec(ρ¹.(xy₁)))), 
                    (I(2)⊗spdiagm(vec(ρ².(xy₂)))), 
                    (I(2)⊗spdiagm(vec(ρ³.(xy₃)))),
                    (I(2)⊗spdiagm(vec(ρ⁴.(xy₄)))),)

const Δt = 1e-3
tf = 40.0
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
  m₁, m₂, m₃, m₄ = m
  ((reshape(sol[1:m₁^2], (m₁,m₁)), reshape(sol[m₁^2+1:2m₁^2], (m₁,m₁))), 
  (reshape(sol[2m₁^2+1:2m₁^2+m₂^2], (m₂,m₂)), reshape(sol[2m₁^2+m₂^2+1:2m₁^2+2m₂^2], (m₂,m₂))),
  (reshape(sol[(2m₁^2+2m₂^2)+1:(2m₁^2+2m₂^2)+m₃^2], (m₃,m₃)), reshape(sol[(2m₁^2+2m₂^2)+m₃^2+1:(2m₁^2+2m₂^2)+2m₃^2], (m₃,m₃))),
  (reshape(sol[(2m₁^2+2m₂^2+2m₃^2)+1:(2m₁^2+2m₂^2+2m₃^2)+m₄^2], (m₄,m₄)), reshape(sol[(2m₁^2+2m₂^2+2m₃^2)+m₄^2+1:(2m₁^2+2m₂^2+2m₃^2)+2m₄^2], (m₄,m₄))))
end

U₀(x) = @SVector [exp(-((x[1]-2π)^2 + (x[2]-6π)^2)), -exp(-((x[1]-2π)^2 + (x[2]-6π)^2))]
V₀(x) = @SVector [0.0,0.0]

# Begin time loop
let
  t = 0.0
  X₀ = vcat(eltocols(vec(U₀.(xy₁))), eltocols(vec(U₀.(xy₂))), eltocols(vec(U₀.(xy₃))), eltocols(vec(U₀.(xy₄))));
  Y₀ = vcat(eltocols(vec(V₀.(xy₁))), eltocols(vec(V₀.(xy₂))), eltocols(vec(V₀.(xy₃))), eltocols(vec(V₀.(xy₄))));
  global Z₀ = vcat(X₀, Y₀)
  global maxvals = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma4\stima4
  K = [zero(M) I(size(M,1)); M zero(M)]
  @gif for i=1:ntime
  # for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(K, sol)    
    t += Δt        
    (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))
    
    u1ref₁,u2ref₁ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[1];
    u1ref₂,u2ref₂ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[2];
    u1ref₃,u2ref₃ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[3];
    u1ref₄,u2ref₄ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[4];
    # Plotting part
    plt3 = scatter(Tuple.(xy₁ |> vec), zcolor=vec(u1ref₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.(xy₂ |> vec), zcolor=vec(u1ref₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.(xy₃ |> vec), zcolor=vec(u1ref₃), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.(xy₄ |> vec), zcolor=vec(u1ref₄), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
    scatter!(plt3, Tuple.([Ω₁([q,0.0]) for q in LinRange(0,1,m₁)]), label="", msw=0.01, ms=2)
    scatter!(plt3, Tuple.([Ω₃([0.0,r]) for r in LinRange(0,1,m₂)]), label="", msw=0.01, ms=2, right_margin=10*Plots.mm, size=(800,800))
    scatter!(plt3, Tuple.([Ω₄([r,1.0]) for r in LinRange(0,1,m₃)]), label="", msw=0.01, ms=2, right_margin=10*Plots.mm, size=(800,800))

    maxvals[i] = max(maximum(abs.(u1ref₁)), maximum(abs.(u1ref₂)), maximum(abs.(u1ref₃)))
  # end
  end  every 100 
end  

u1ref₁,u2ref₁ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[1];
u1ref₂,u2ref₂ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[2];
u1ref₃,u2ref₃ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[3];
u1ref₄,u2ref₄ = get_sol_vector_from_raw_vector(Z₀[1:(2m₁^2 + 2m₂^2 + 2m₃^2 + 2m₄^2)], (m₁, m₂, m₃, m₄))[4];
plt3 = scatter(Tuple.(xy₁ |> vec), zcolor=vec(u1ref₁), colormap=:turbo, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.(xy₂ |> vec), zcolor=vec(u1ref₂), colormap=:turbo, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.(xy₃ |> vec), zcolor=vec(u1ref₃), colormap=:turbo, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.(xy₄ |> vec), zcolor=vec(u1ref₄), colormap=:turbo, ylabel="y(=r)", markersize=2, msw=0.01, label="");
scatter!(plt3, Tuple.([Ω₁([q,0.0]) for q in LinRange(0,1,m₁)]), label="", msw=0.01, ms=2)
scatter!(plt3, Tuple.([Ω₃([0.0,r]) for r in LinRange(0,1,m₂)]), label="", msw=0.01, ms=2, right_margin=10*Plots.mm, size=(800,800))
scatter!(plt3, Tuple.([Ω₄([r,1.0]) for r in LinRange(0,1,m₃)]), label="", msw=0.01, ms=2, right_margin=10*Plots.mm, size=(800,800))

plt4 = plot(LinRange(0,tf,ntime), maxvals, lw=2, label="", xlabel="t", ylabel="||U||∞")