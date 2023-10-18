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
Ω₁(qr) = S(qr, domain₁)
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, r]; # Left boundary
c₁²(q) = [q, 0.0]; # Bottom boundary. 
c₂²(r) = [1.0, r]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface 
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₂(qr) = S(qr, domain₂)

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

"""
Function to get the marker matrix for implementing the jump conditions on the interface.
The resulting matrix uses an interpolation operator used in SBP techniques.
"""
function get_marker_matrix(N_C)
  C2F, F2C = INTERPOLATION_4(N_C)
  N_F = 2*N_C-1
  
  I_N_C = spzeros(Float64, N_C, N_F)  
  I_N_C[1, N_F] = 1.0

  I_N_F = spzeros(Float64, N_F, N_C)  
  I_N_F[N_F, 1] = 1.0

  J_N_C = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for  q in LinRange(0,1,N_C)].^(0.5))
  J_N_F = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for  q in LinRange(0,1,N_F)].^(0.5))

  W₁ = I(2) ⊗ I(N_C) ⊗ E1(1, 1, N_C)
  W₂ = I(2) ⊗ I(N_F) ⊗ E1(N_F, N_F, N_F)
  Z₁ = I(2) ⊗ (J_N_C\(F2C*J_N_F)) ⊗ I_N_C
  Z₂ = I(2) ⊗ (J_N_F\(C2F*J_N_C)) ⊗ I_N_F 
  mk1 = [-W₁  Z₁; -Z₂  W₂]
  mk2 = [-W₁  Z₁; Z₂  -W₂]
  mk1, mk2
end


"""
Stiffness matrix function for non-conforming interface
"""
function 𝐊2_NC(𝐪𝐫₁, 𝐪𝐫₂)
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫₁) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Dᴱ(Pqr₁)) # Elasticity bulk differential operator
  𝐓q₀¹ = Tᴱ(Pqr₁, Ω₁, [-1,0]).A
  𝐓r₀¹ = Tᴱ(Pqr₁, Ω₁, [0,-1]).A
  𝐓qₙ¹ = Tᴱ(Pqr₁, Ω₁, [1,0]).A 
  𝐓rₙ¹ = Tᴱ(Pqr₁, Ω₁, [0,1]).A 
  # Second layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫₂) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Dᴱ(Pqr₂)) # Elasticity bulk differential operator 
  𝐓q₀² = Tᴱ(Pqr₂, Ω₂, [-1,0]).A
  𝐓r₀² = Tᴱ(Pqr₂, Ω₂, [0,-1]).A
  𝐓qₙ² = Tᴱ(Pqr₂, Ω₂, [1,0]).A 
  𝐓rₙ² = Tᴱ(Pqr₂, Ω₂, [0,1]).A 
  # Get the 2d operators
  m₁,n₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀¹, 𝐇qₙ¹, _, 𝐇rₙ¹ = sbp_2d₁.norm
  m₂,n₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀², 𝐇qₙ², 𝐇r₀², _ = sbp_2d₂.norm
  # Determinants of the transformation
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫₁))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫₂)) 
  Jbulk⁻¹ = blockdiag(spdiagm(detJ1₁.^-1), spdiagm(detJ1₂.^-1))
  # SJr₀¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m₁), m₁)
  SJq₀¹ = get_surf_J(I(2)⊗E1(1,1,m₁)⊗spdiagm([(det(J([0.0,q], Ω₁))*J⁻¹s([0.0,q], Ω₁, [-1,0])) for q in LinRange(0,1,m₁)]), m₁)
  SJrₙ¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₁))*J⁻¹s([q,1.0], Ω₁, [0,1])) for q in LinRange(0,1,m₁)])⊗E1(m₁,m₁,m₁), m₁)
  SJqₙ¹ = get_surf_J(I(2)⊗E1(m₁,m₁,m₁)⊗spdiagm([(det(J([1.0,q], Ω₁))*J⁻¹s([1.0,q], Ω₁, [1,0])) for q in LinRange(0,1,m₁)]), m₁)

  SJr₀² = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₂))*J⁻¹s([q,0.0], Ω₂, [0,-1])) for q in LinRange(0,1,m₂)])⊗E1(1,1,m₂), m₂)
  SJq₀² = get_surf_J(I(2)⊗E1(1,1,m₂)⊗spdiagm([(det(J([0.0,q], Ω₂))*J⁻¹s([0.0,q], Ω₂, [-1,0])) for q in LinRange(0,1,m₂)]), m₂)
  # SJrₙ² = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m)])⊗E1(m₂,m₂,m₂), m₂)
  SJqₙ² = get_surf_J(I(2)⊗E1(m₂,m₂,m₂)⊗spdiagm([(det(J([1.0,q], Ω₂))*J⁻¹s([1.0,q], Ω₂, [1,0])) for q in LinRange(0,1,m₂)]), m₂)
  # Combine the operators    
  𝐏 = blockdiag(𝐏₁, 𝐏₂)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀¹)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ¹)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ¹)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀²)*SJq₀²*(𝐓q₀²) + (I(2)⊗𝐇qₙ²)*SJqₙ²*(𝐓qₙ²) + -(I(2)⊗𝐇r₀²)*SJr₀²*(𝐓r₀²))

  # Traction on the interface      
  Hq₁ = sbp_q₁.norm;  Hr₁ = sbp_r₁.norm    
  Hq₂ = sbp_q₂.norm;  Hr₂ = sbp_r₂.norm    
  Hq₁⁻¹ = (Hq₁)\I(m₁) |> sparse;  Hr₁⁻¹ = (Hr₁)\I(n₁) |> sparse
  Hq₂⁻¹ = (Hq₂)\I(m₂) |> sparse;  Hr₂⁻¹ = (Hr₂)\I(n₂) |> sparse  
  SJ₁ = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for  q in LinRange(0,1,m₁)])
  SJ₂ = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for  q in LinRange(0,1,m₂)])
  𝐃 = blockdiag((I(2)⊗(SJ₁*Hr₁)⊗I(m₁))*(I(2)⊗I(m₁)⊗(E1(1,1,m₁))), (I(2)⊗(SJ₂*Hr₂)⊗I(m₂))*(I(2)⊗I(m₂)⊗E1(m₂,m₂,m₂)))
  𝐃⁻¹ = blockdiag((I(2)⊗Hq₁⁻¹⊗Hr₁⁻¹), (I(2)⊗Hq₂⁻¹⊗Hr₂⁻¹))
  BHᵀ, BT = get_marker_matrix(m₁) # Assuming coarse mesh in layer 1
   
  𝐓r = blockdiag(𝐓r₀¹, 𝐓rₙ²)
  𝐓rᵀ = blockdiag(𝐓r₀¹, 𝐓rₙ²)'      
  
  X = 𝐃*BHᵀ*𝐓r;
  Xᵀ = 𝐓rᵀ*𝐃*BHᵀ;
  
  𝚯 = 𝐃⁻¹*X
  𝚯ᵀ = -𝐃⁻¹*Xᵀ
  Ju = -𝐃⁻¹*𝐃*BT;   
  
  h = cᵢ(1)[1]/(m₂-1)
  ζ₀ = 40/h
  𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju
  
  Jbulk⁻¹*(𝐏 - 𝐓 - 𝐓ᵢ)
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


#############################
# Begin solving the problem #
#############################
N = [21,31,41]
h1 = 1 ./(N .- 1)
L²Error = zeros(Float64, length(N))
const Δt = 1e-3
tf = 1.0
ntime = ceil(Int, tf/Δt)
max_err = zeros(Float64, ntime, length(N))
  
for (m,Ni) in zip(N, 1:length(N))
  let    
    𝐪𝐫₁ = generate_2d_grid((m, m)); # Coarser grid
    𝐪𝐫₂ = generate_2d_grid((2*m-1, 2*m-1)); # Finer grid
    xy₁ = Ω₁.(𝐪𝐫₁)
    xy₂ = Ω₂.(𝐪𝐫₂)   
    global stima2 = 𝐊2_NC(𝐪𝐫₁, 𝐪𝐫₂);     
    u₀ = vcat(eltocols(vec(U.(xy₁,0.0))), eltocols(vec(U.(xy₂,0.0)))) # Function in "2d_elasticity_problem.jl"
    v₀ = vcat(eltocols(vec(Uₜ.(xy₁,0.0))), eltocols(vec(Uₜ.(xy₂,0.0)))) # Function in "2d_elasticity_problem.jl"        
    massma2 = blockdiag((I(2)⊗spdiagm(vec(ρ¹.(xy₁)))), (I(2)⊗spdiagm(vec(ρ².(xy₂)))))
    M⁺ = (massma2 - (Δt/2)^2*stima2)
    M⁻ = (massma2 + (Δt/2)^2*stima2)
    luM⁺ = factorize(M⁺)
      
    m₁, n₁ = size(𝐪𝐫₁)
    m₂, n₂ = size(𝐪𝐫₂)
    sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁);    sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
    sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂);    sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
    sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
    sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
      
    let
      u₀ = vcat(eltocols(vec(U.(xy₁,0.0))), eltocols(vec(U.(xy₂,0.0)))) # Function in "2d_elasticity_problem.jl"
      v₀ = vcat(eltocols(vec(Uₜ.(xy₁,0.0))), eltocols(vec(Uₜ.(xy₂,0.0)))) # Function in "2d_elasticity_problem.jl"
      global u₁ = zero(u₀)
      global v₁ = zero(v₀)            
      t = 0.0
      for i=1:ntime
        Fₙ = vcat(eltocols(vec(F.(xy₁, t, σ¹, ρ¹))), eltocols(vec(F.(xy₂, t, σ², ρ²))))
        Fₙ₊₁ = vcat(eltocols(vec(F.(xy₁, t+Δt, σ¹, ρ¹))), eltocols(vec(F.(xy₂, t+Δt, σ², ρ²))))
        normals(Ω) = (r->Ω([0.0,r]), q->Ω([q,0.0]), r->Ω([1.0,r]), q->Ω([q,1.0]))
        gₙ = vcat(𝐠(t, (m₁,n₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                 𝐠(t, (m₂,n₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
        gₙ₊₁ = vcat(𝐠(t+Δt, (m₁,n₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                  𝐠(t+Δt, (m₂,n₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
          
        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(luM⁺, M⁻, massma2, fargs) # Function in "time-stepping.jl"
        (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(abs.(u₁))))
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
        max_err[i,Ni] = maximum(abs.(u₁ - vcat(eltocols(vec(U.(xy₁, t))), eltocols(vec(U.(xy₂, t))))))
      end
    end
      
    Hq₁ = sbp_q₁.norm;  Hr₁ = sbp_r₁.norm
    Hq₂ = sbp_r₂.norm;  Hr₂ = sbp_r₂.norm;
    𝐇 = blockdiag((I(2) ⊗ Hq₁ ⊗ Hr₁), (I(2) ⊗ Hq₂ ⊗ Hr₂))
    e = u₁ - vcat(eltocols(vec(U.(xy₁, tf))), eltocols(vec(U.(xy₂, tf))))
    L²Error[Ni] = sqrt(e'*𝐇*e)
    println("Done N = "*string(m)*", L²Error = "*string(L²Error[Ni]))
  end
end

function get_sol_vector_from_raw_vector(sol, mn₁, mn₂)
  m₁, n₁ = mn₁
  m₂, n₂ = mn₂
  (reshape(sol[1:m₁^2], (m₁, m₁)), 
   reshape(sol[m₁^2+1:m₁^2+n₁^2], (n₁,n₁)),
   reshape(sol[m₁^2+n₁^2+1:m₁^2+n₁^2+m₂^2], (m₂,m₂)), 
   reshape(sol[m₁^2+n₁^2+m₂^2+1:m₁^2+n₁^2+m₂^2+n₂^2], (n₂,n₂)))
end

𝐪𝐫₁ = generate_2d_grid((N[end],N[end])); # Coarser grid
𝐪𝐫₂ = generate_2d_grid((2*N[end]-1,2*N[end]-1)); # Finer grid
xy₁ = Ω₁.(𝐪𝐫₁);
xy₂ = Ω₂.(𝐪𝐫₂);
Uap₁, Vap₁, Uap₂, Vap₂ = get_sol_vector_from_raw_vector(u₁, (N[end],N[end]), (2*N[end]-1, 2*N[end]-1));

plt1 = scatter(Tuple.(xy₁ |> vec), zcolor=vec(Uap₁), label="", title="Approx. solution (u(x,y))", markersize=4, msw=0.1);
scatter!(plt1, Tuple.(xy₂ |> vec), zcolor=vec(Uap₂), label="", markersize=4, msw=0.1);
plt3 = scatter(Tuple.(xy₁ |> vec), zcolor=vec(Vap₁), label="", title="Approx. solution (v(x,y))", markersize=4, msw=0.1);
scatter!(plt3, Tuple.(xy₂ |> vec), zcolor=vec(Vap₂), label="", markersize=4, msw=0.1);
plt1_3 = plot(plt1, plt3, layout=(1,2), size=(800,800))

plt4 = plot();
for i=1:lastindex(h1)
  t_arr = LinRange(0,tf,ntime)
  plot!(plt4, t_arr, max_err[:,i], label="h="*string(h1[i]), yscale=:log10, lw=1.5, legend=:bottomright)
end