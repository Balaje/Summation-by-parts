include("2d_elasticity_problem.jl")

using SplitApplyCombine

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
f(q) = 1 + 0.0*sin(2π*q)
cᵢ(q) = [q, f(q)];
# Define the rest of the boundary
c₀¹(r) = [0.0 , 1+r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [1.0, 1+r]; # Right boundary
c₃¹(q) = [q, 2.0]; # Top boundary
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
function λ(x)
  if((x[2] ≈ cᵢ(x[1])[2]) || (x[2] > cᵢ(x[1])[2]))
    return 2.0
  else
    return 2.0
  end
end
function μ(x)
  if((x[2] ≈ cᵢ(x[1])[2]) || (x[2] > cᵢ(x[1])[2]))  
    return 1.0
  else
    return 1.0
  end
end
"""
The density of the material, ρ
"""
function ρ(x) 
  if((x[2] ≈ cᵢ(x[1])[2]) || (x[2] > cᵢ(x[1])[2]))  
    return 1.0
  else
    return 1.0
  end
end 

"""
Material properties coefficients of an anisotropic material
"""
c₁₁(x) = 2*μ(x)+λ(x)
c₂₂(x) = 2*μ(x)+λ(x)
c₃₃(x) = μ(x)
c₁₂(x) = λ(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫(x) = @SMatrix [c₁₁(x) 0 0 c₁₂(x); 0 c₃₃(x) c₃₃(x) 0; 0 c₃₃(x) c₃₃(x) 0; c₁₂(x) 0 0 c₂₂(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ(∇u,x) = 𝒫(x)*∇u

"""
Function to get the marker matrix for implementing the jump conditions on the interface.
The resulting matrix uses an interpolation operator used in SBP techniques.
"""
function get_marker_matrix(N_C::Int64)
  C2F, F2C = INTERPOLATION_4(N_C)
  N_F = 2*N_C-1
  
  I_N_C = spzeros(Float64, N_C, N_F)  
  I_N_C[1, N_F] = 1.0

  I_N_F = spzeros(Float64, N_F, N_C)  
  I_N_F[N_F, 1] = 1.0

  W₁ = I(2) ⊗ I(N_C) ⊗ E1(1, 1, N_C)
  W₂ = I(2) ⊗ I(N_F) ⊗ E1(N_F, N_F, N_F)
  Z₁ = I(2) ⊗ F2C ⊗ I_N_C
  Z₂ = I(2) ⊗ C2F ⊗ I_N_F 
  mk1 = [-W₁  Z₁; -Z₂  W₂]
  mk2 = [-W₁  Z₁; Z₂  -W₂]
  mk1, mk2
end

function 𝐊2_NC(𝐪𝐫₁, 𝐪𝐫₂)
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫, Ω₁, 𝐪𝐫₁) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Dᴱ(Pqr₁)) # Elasticity bulk differential operator
  𝐓₁ = Tᴱ(Pqr₁) # Elasticity Traction operator
  𝐓q₁ = 𝐓₁.A
  𝐓r₁ = 𝐓₁.B
  # Second layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫, Ω₂, 𝐪𝐫₂) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Dᴱ(Pqr₂)) # Elasticity bulk differential operator
  𝐓₂ = Tᴱ(Pqr₂) # Elasticity Traction operator
  𝐓q₂ = 𝐓₂.A
  𝐓r₂ = 𝐓₂.B
  # Get the 2d operators
  m₁,n₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀¹, 𝐇qₙ¹, 𝐇r₀¹, 𝐇rₙ¹ = sbp_2d₁.norm
  m₂,n₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀², 𝐇qₙ², 𝐇r₀², 𝐇rₙ² = sbp_2d₂.norm
  # Determinants of the transformation
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫₁))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫₂)) 
  Jbulk⁻¹ = blockdiag(spdiagm(detJ1₁.^-1), spdiagm(detJ1₂.^-1))
  # Combine the operators    
  𝐏 = blockdiag(𝐏₁, 𝐏₂)
  𝐓 = blockdiag(-(I(2) ⊗ 𝐇q₀¹)*(𝐓q₁) + (I(2) ⊗ 𝐇qₙ¹)*(𝐓q₁) + (I(2) ⊗ 𝐇rₙ¹)*(𝐓r₁),
                 -(I(2) ⊗ 𝐇q₀²)*(𝐓q₂) + (I(2) ⊗ 𝐇qₙ²)*(𝐓q₂) + -(I(2) ⊗ 𝐇r₀²)*(𝐓r₂)) 

  # Traction on the interface      
  Hq₁ = sbp_q₁.norm;  Hr₁ = sbp_r₁.norm    
  Hq₂ = sbp_q₂.norm;  Hr₂ = sbp_r₂.norm    
  Hq₁⁻¹ = (Hq₁)\I(m₁) |> sparse;  Hr₁⁻¹ = (Hr₁)\I(n₁) |> sparse
  Hq₂⁻¹ = (Hq₂)\I(m₂) |> sparse;  Hr₂⁻¹ = (Hr₂)\I(n₂) |> sparse  
  𝐃 = blockdiag((I(2)⊗(Hr₁)⊗I(m₁))*(I(2)⊗I(m₁)⊗(E1(1,1,m₁))), (I(2)⊗(Hr₂)⊗I(m₂))*(I(2)⊗I(m₂)⊗E1(m₂,m₂,m₂)))
  𝐃⁻¹ = blockdiag((I(2)⊗Hq₁⁻¹⊗Hr₁⁻¹), (I(2)⊗Hq₂⁻¹⊗Hr₂⁻¹))
  BHᵀ, BT = get_marker_matrix(m₁) # Assuming coarse mesh in layer 1
  
  𝐓r = blockdiag(𝐓r₁, 𝐓r₂)
  𝐓rᵀ = blockdiag(𝐓r₁, 𝐓r₂)'    
  
  X = 𝐃*BHᵀ*𝐓r;
  Xᵀ = 𝐃*𝐓rᵀ*BHᵀ;
  
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
function 𝐠(t::Float64, mn::Tuple{Int64,Int64}, norm, Ω, P, C)
  m,n= mn
  q = LinRange(0,1,m); r = LinRange(0,1,n) # Reference coordinate axes
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = norm # The inverse of the norm matrices
  P1, P2, P3, P4 = P # A parameter to indicate the nature of the boundary; 0: Interface, 1: CW, -1: CCW
  c₀, c₁, c₂, c₃ = C # The parametric representation of the boundary
  bvals_q₀ = reduce(hcat, [J⁻¹s(@SVector[0.0, rᵢ], Ω, @SVector[-1.0,0.0])*g(t, c₀, rᵢ, P1) for rᵢ in r])
  bvals_r₀ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 0.0], Ω, @SVector[0.0,-1.0])*g(t, c₁, qᵢ, P2) for qᵢ in q])
  bvals_qₙ = reduce(hcat, [J⁻¹s(@SVector[1.0, rᵢ], Ω, @SVector[1.0,0.0])*g(t, c₂, rᵢ, P3) for rᵢ in r])
  bvals_rₙ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 1.0], Ω, @SVector[0.0,1.0])*g(t, c₃, qᵢ, P4) for qᵢ in q])    
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
N = [21,41]
h1 = 1 ./(N .- 1)
L²Error = zeros(Float64, length(N))
const Δt = 1e-3
tf = 0.5
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
    massma2 = blockdiag((I(2)⊗spdiagm(vec(ρ.(xy₁)))), (I(2)⊗spdiagm(vec(ρ.(xy₂)))))
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
        Fₙ = vcat(eltocols(vec(F.(xy₁, t))), eltocols(vec(F.(xy₂, t))))
        Fₙ₊₁ = vcat(eltocols(vec(F.(xy₁, t+Δt))), eltocols(vec(F.(xy₂, t+Δt))))
        normals(Ω) = (r->Ω([0.0,r]), q->Ω([q,0.0]), r->Ω([1.0,r]), q->Ω([q,1.0]))
        gₙ = vcat(𝐠(t, (m₁,n₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁)),
                 𝐠(t, (m₂,n₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂)))
        gₙ₊₁ = vcat(𝐠(t+Δt, (m₁,n₁), sbp_2d₁.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁)),
                  𝐠(t+Δt, (m₂,n₂), sbp_2d₂.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂)))
          
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

plt10_1 = scatter(Tuple.(xy₁ |> vec), size=(800,800), markersize=4, xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Layer 1", msw=0.1)
plt10_2 = scatter!(plt10_1,Tuple.(xy₂ |> vec), size=(800,800), markersize=2, markercolor="red", xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Layer 2", msw=0.1)
plt10_12 = plot(plt10_1, plt10_2, layout=(2,1))
plt10_3 = scatter(Tuple.(𝐪𝐫₁ |> vec), xlabel="q", ylabel="r", label="Reference Domain", markersize=4, markercolor="white", aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), msw=0.1);
plt10_4 = scatter(Tuple.(𝐪𝐫₂ |> vec), xlabel="q", ylabel="r", label="Reference Domain", markersize=4, markercolor="white", aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), msw=0.1);
plt10 = plot(plt10_1, plt10_3, plt10_2, plt10_4, layout=(2,2));