include("2d_elasticity_problem.jl")

using SplitApplyCombine

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
f(q) = 1 + 0.1*sin(2π*q)
cᵢ(q) = [q, f(q)];
# Define the rest of the boundary
c₀¹(r) = [0.0 , 1+r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [1.0, 1+r]; # Right boundary
c₃¹(q) = [q, 2.0 + 0.1*sin(2π*q)]; # Top boundary
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, r]; # Left boundary
c₁²(q) = [q, 0.0]; # Bottom boundary. 
c₂²(r) = [1.0, r]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface 
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)

###################################################################
# In this problem, we have two reference grids on the two domains #
# For example:                                                    #
#                                                                 #
# N = 21;                                                         #
# 𝐪𝐫₁ = generate_2d_grid((21,21)); # Coarser grid                 #
# 𝐪𝐫₂ = generate_2d_grid((2*N-1,2*N-1)); # Finer grid             #
# xy₁ = Ω₁.(𝐪𝐫₁)                                                  #
# xy₂ = Ω₂.(𝐪𝐫₂)                                                  #
###################################################################

###############################################
# We use different properties for both layers #
###############################################
"""
The Lamé parameters μ, λ
"""
λ¹(x) = 2.0
μ¹(x) = 1.0
λ²(x) = 2.0
μ²(x) = 1.0

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

"""
Density function 
"""
ρ¹(x) = 1.0
ρ²(x) = 0.5

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫¹(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫²(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ¹(∇u,x) = 𝒫¹(x)*∇u
σ²(∇u,x) = 𝒫²(x)*∇u

function 𝐊2_NC!(𝒫, 𝛀::Tuple{DiscreteDomain, DiscreteDomain},  𝐪𝐫)
  𝒫¹, 𝒫² = 𝒫
  𝛀₁, 𝛀₂ = 𝛀
  𝐪𝐫₁, 𝐪𝐫₂ = 𝐪𝐫
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)  

  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫₁) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Pqr₁) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹, 𝐓r₀¹, 𝐓qₙ¹, 𝐓rₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [-1,0]).A, Tᴱ(Pqr₁, 𝛀₁, [0,-1]).A, Tᴱ(Pqr₁, 𝛀₁, [1,0]).A, Tᴱ(Pqr₁, 𝛀₁, [0,1]).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫₂) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Pqr₂) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀², 𝐓r₀², 𝐓qₙ², 𝐓rₙ² = Tᴱ(Pqr₂, 𝛀₂, [-1,0]).A, Tᴱ(Pqr₂, 𝛀₂, [0,-1]).A, Tᴱ(Pqr₂, 𝛀₂, [1,0]).A, Tᴱ(Pqr₂, 𝛀₂, [0,1]).A 
  
  # Get the norm matrices (Different on the two layers)
  # Layer 1
  m₁, n₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀⁻¹₁, 𝐇qₙ⁻¹₁, _, 𝐇rₙ⁻¹₁ = sbp_2d₁.norm  
  # Layer 2
  m₂, n₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀⁻¹₂, 𝐇qₙ⁻¹₂, 𝐇r₀⁻¹₂, _ = sbp_2d₂.norm
  
  # Determinants of the transformation
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫₁)
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫₂) 
  𝐉 = blockdiag(𝐉₁, 𝐉₂)   
  
  # Surface Jacobians of the outer boundaries
  # - Layer 1  
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ = Js(𝛀₁, [0,-1]; X=I(2)), Js(𝛀₁, [-1,0]; X=I(2)), Js(𝛀₁, [0,1]; X=I(2)), Js(𝛀₁, [1,0]; X=I(2))   
  # - Layer 2
  SJr₀², SJq₀², _, SJqₙ² = Js(𝛀₂, [0,-1]; X=I(2)), Js(𝛀₂, [-1,0]; X=I(2)), Js(𝛀₂, [0,1]; X=I(2)), Js(𝛀₂, [1,0]; X=I(2))   

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁.A, 𝐏₂.A)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀⁻¹₁)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ⁻¹₁)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ⁻¹₁)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀⁻¹₂)*SJq₀²*(𝐓q₀²) + (I(2)⊗𝐇qₙ⁻¹₂)*SJqₙ²*(𝐓qₙ²) + -(I(2)⊗𝐇r₀⁻¹₂)*SJr₀²*(𝐓r₀²))
  𝐓rᵢ = blockdiag(𝐓r₀¹, 𝐓rₙ²)            
  
  # Get the Interface SAT for Conforming Interface
  B̂, B̃, 𝐇₁⁻¹, 𝐇₂⁻¹ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], NonConformingInterface(); X=I(2))
  
  h = 1/(max(m₁,m₂)-1)
  ζ₀ = 40/h
  𝐓ᵢ = (blockdiag(I(2)⊗𝐇₁⁻¹, I(2)⊗𝐇₂⁻¹))*(0.5*B̂*𝐓rᵢ - 0.5*𝐓rᵢ'*B̂ - ζ₀*B̃)
  
  𝐉\(𝐏 - 𝐓 - 𝐓ᵢ)
end

"""
Neumann boundary condition vector
"""
function 𝐠(t::Float64, mn::Tuple{Int64,Int64}, norm, Ω, P, C, σ)
  m,n= mn
  q = LinRange(0,1,m); r = LinRange(0,1,n)
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = norm
  P1, P2, P3, P4 = P
  c₀, c₁, c₂, c₃ = C
    
  bvals_q₀ = reduce(hcat, [J⁻¹s([0.0,rᵢ], Ω, [-1,0])*g(t, c₀, rᵢ, σ, P1) for rᵢ in r])
  bvals_r₀ = reduce(hcat, [J⁻¹s([qᵢ,0.0], Ω, [0,-1])*g(t, c₁, qᵢ, σ, P2) for qᵢ in q])
  bvals_qₙ = reduce(hcat, [J⁻¹s([1.0,rᵢ], Ω, [1,0])*g(t, c₂, rᵢ, σ, P3) for rᵢ in r])
  bvals_rₙ = reduce(hcat, [J⁻¹s([qᵢ,1.0], Ω, [0,1])*g(t, c₃, qᵢ, σ, P4) for qᵢ in q])
    
  E1(i,M) = diag(SBP.SBP_2d.E1(i,i,M))
  bq₀ = (E1(1,2) ⊗ E1(1,m) ⊗ (bvals_q₀[1,:])) + (E1(2,2) ⊗ E1(1,m) ⊗ (bvals_q₀[2,:]))
  br₀ = (E1(1,2) ⊗ (bvals_r₀[1,:]) ⊗ E1(1,n)) + (E1(2,2) ⊗ (bvals_r₀[2,:]) ⊗ E1(1,n))
  bqₙ = (E1(1,2) ⊗ E1(m,n) ⊗ (bvals_qₙ[1,:])) + (E1(2,2) ⊗ E1(m,n) ⊗ (bvals_qₙ[2,:]))
  brₙ = (E1(1,2) ⊗ (bvals_rₙ[1,:]) ⊗ E1(m,n)) + (E1(2,2) ⊗ (bvals_rₙ[2,:]) ⊗ E1(m,n))
    
  collect((I(2)⊗𝐇r₀)*br₀ + (I(2)⊗𝐇rₙ)*brₙ + (I(2)⊗𝐇q₀)*bq₀ + (I(2)⊗𝐇qₙ)*bqₙ)
end
  
#################################
# Now begin solving the problem #
#################################
N = [21,31,41]
h1 = 1 ./(N .- 1)
L²Error = zeros(Float64, length(N))
const Δt = 1e-3
tf = 1.0
ntime = ceil(Int, tf/Δt)
max_err = zeros(Float64, ntime, length(N))
  
for (m,Ni) in zip(N, 1:length(N))
  let
    m₁ = m
    m₂ = 2m-1;
    qr₁ = generate_2d_grid((m₁, m₁));
    qr₂ = generate_2d_grid((m₂, m₂));
    𝛀₁ = DiscreteDomain(domain₁, (m₁,m₁));
    𝛀₂ = DiscreteDomain(domain₂, (m₂,m₂));
    Ω₁(qr) = S(qr, 𝛀₁.domain);
    Ω₂(qr) = S(qr, 𝛀₂.domain);

    global stima2_nc = 𝐊2_NC!((𝒫¹, 𝒫²), (𝛀₁, 𝛀₂), (qr₁, qr₂));
    𝐱𝐲₁ = Ω₁.(qr₁)
    𝐱𝐲₂ = Ω₂.(qr₂)        
    massma2_nc = blockdiag((I(2)⊗spdiagm(vec(ρ¹.(𝐱𝐲₁)))), (I(2)⊗spdiagm(vec(ρ².(𝐱𝐲₂)))))
    M⁺ = (massma2_nc - (Δt/2)^2*stima2_nc)
    M⁻ = (massma2_nc + (Δt/2)^2*stima2_nc)
    luM⁺ = factorize(M⁺)
      
    # Get the norm matrices (Different on the two layers)
    # Layer 1    
    sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
    sbp_r₁ = SBP_1_2_CONSTANT_0_1(m₁)
    sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)  
    # Layer 2    
    sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
    sbp_r₂ = SBP_1_2_CONSTANT_0_1(m₂)
    sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
      
    let
      u₀ = vcat(eltocols(vec(U.(𝐱𝐲₁,0.0))), eltocols(vec(U.(𝐱𝐲₂,0.0))))
      v₀ = vcat(eltocols(vec(Uₜ.(𝐱𝐲₁,0.0))), eltocols(vec(Uₜ.(𝐱𝐲₂,0.0))))
      global u₁ = zero(u₀)
      global v₁ = zero(v₀)            
      t = 0.0
      for i=1:ntime
        Fₙ = vcat(eltocols(vec(F.(𝐱𝐲₁, t, σ¹, ρ¹))), eltocols(vec(F.(𝐱𝐲₂, t, σ², ρ²))))
        Fₙ₊₁ = vcat(eltocols(vec(F.(𝐱𝐲₁, t+Δt, σ¹, ρ¹))), eltocols(vec(F.(𝐱𝐲₂, t+Δt, σ², ρ²))))
        normals(Ω) = (r->Ω([0.0,r]), q->Ω([q,0.0]), r->Ω([1.0,r]), q->Ω([q,1.0]))
        gₙ = vcat(𝐠(t, (m₁,m₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                 𝐠(t, (m₂,m₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
        gₙ₊₁ = vcat(𝐠(t+Δt, (m₁,m₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                   𝐠(t+Δt, (m₂,m₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
          
        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(luM⁺, M⁻, massma2_nc, fargs) # Function in "time-stepping.jl"
        (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(abs.(u₁))))
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
        max_err[i,Ni] = maximum(abs.(u₁ - vcat(eltocols(vec(U.(𝐱𝐲₁, t))), eltocols(vec(U.(𝐱𝐲₂, t))))))
      end
    end
      
    Hq₁ = sbp_q₁.norm
    Hr₁ = sbp_r₁.norm
    Hq₂ = sbp_q₂.norm
    Hr₂ = sbp_r₂.norm
    𝐇 = blockdiag((I(2) ⊗ Hq₁ ⊗ Hr₁), (I(2) ⊗ Hq₂ ⊗ Hr₂))
    e = u₁ - vcat(eltocols(vec(U.(𝐱𝐲₁, tf))), eltocols(vec(U.(𝐱𝐲₂, tf))))    
    L²Error[Ni] = sqrt(e'*𝐇*e)
    println("Done N = "*string(m)*", L²Error = "*string(L²Error[Ni]))
  end
end