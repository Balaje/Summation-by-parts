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

## Define the material properties on the physical grid
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
Stiffness matrix function
"""
function 𝐊2(𝐪𝐫)
  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Dᴱ(Pqr₁)) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹ = Tᴱ(Pqr₁, Ω₁, [-1,0]).A
  𝐓r₀¹ = Tᴱ(Pqr₁, Ω₁, [0,-1]).A
  𝐓qₙ¹ = Tᴱ(Pqr₁, Ω₁, [1,0]).A 
  𝐓rₙ¹ = Tᴱ(Pqr₁, Ω₁, [0,1]).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Dᴱ(Pqr₂)) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀² = Tᴱ(Pqr₂, Ω₂, [-1,0]).A
  𝐓r₀² = Tᴱ(Pqr₂, Ω₂, [0,-1]).A
  𝐓qₙ² = Tᴱ(Pqr₂, Ω₂, [1,0]).A 
  𝐓rₙ² = Tᴱ(Pqr₂, Ω₂, [0,1]).A 
  
  # Get the norm matrices (Same for both layers)
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm
  
  # Determinants of the transformation
  detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
  detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫)) 
  Jbulk⁻¹ = blockdiag(spdiagm(detJ1₁.^-1), spdiagm(detJ1₂.^-1))
  
  # Surface Jacobians
  # SJr₀¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m), m)
  SJq₀¹ = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω₁))*J⁻¹s([0.0,q], Ω₁, [-1,0])) for q in LinRange(0,1,m)]), m)
  SJrₙ¹ = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₁))*J⁻¹s([q,1.0], Ω₁, [0,1])) for q in LinRange(0,1,m)])⊗E1(m,m,m), m)
  SJqₙ¹ = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω₁))*J⁻¹s([1.0,q], Ω₁, [1,0])) for q in LinRange(0,1,m)]), m)

  SJr₀² = get_surf_J(I(2)⊗spdiagm([(det(J([q,0.0], Ω₂))*J⁻¹s([q,0.0], Ω₂, [0,-1])) for q in LinRange(0,1,m)])⊗E1(1,1,m), m)
  SJq₀² = get_surf_J(I(2)⊗E1(1,1,m)⊗spdiagm([(det(J([0.0,q], Ω₂))*J⁻¹s([0.0,q], Ω₂, [-1,0])) for q in LinRange(0,1,m)]), m)
  # SJrₙ² = get_surf_J(I(2)⊗spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m)])⊗E1(m,m,m), m)
  SJqₙ² = get_surf_J(I(2)⊗E1(m,m,m)⊗spdiagm([(det(J([1.0,q], Ω₂))*J⁻¹s([1.0,q], Ω₂, [1,0])) for q in LinRange(0,1,m)]), m)

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁, 𝐏₂)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀)*SJq₀²*(𝐓q₀²) + (I(2)⊗𝐇qₙ)*SJqₙ²*(𝐓qₙ²) + -(I(2)⊗𝐇r₀)*SJr₀²*(𝐓r₀²))
  
  # Traction on the interface      
  Hq = sbp_q.norm
  Hr = sbp_q.norm    
  Hq⁻¹ = (Hq)\I(m) |> sparse
  Hr⁻¹ = (Hr)\I(n) |> sparse
  # Hq = sbp_q.norm
  Hr = sbp_r.norm
  sjr₀ = spdiagm([(det(J([q,0.0], Ω₁))*J⁻¹s([q,0.0], Ω₁, [0,-1])) for q in LinRange(0,1,m)])
  sjrₙ = spdiagm([(det(J([q,1.0], Ω₂))*J⁻¹s([q,1.0], Ω₂, [0,1])) for q in LinRange(0,1,m)])
  𝐃 = blockdiag((I(2)⊗(sjr₀*Hr)⊗I(m))*(I(2)⊗I(m)⊗(E1(1,1,m))), (I(2)⊗(sjrₙ*Hr)⊗I(m))*(I(2)⊗I(m)⊗E1(m,m,m))) # # The inverse is contained in the 2d stencil struct                
  𝐃⁻¹ = blockdiag((I(2)⊗Hq⁻¹⊗Hr⁻¹), (I(2)⊗Hq⁻¹⊗Hr⁻¹))
  BHᵀ, BT = get_marker_matrix(m)
  
  𝐓r = blockdiag(𝐓r₀¹, 𝐓rₙ²)
  𝐓rᵀ = blockdiag(𝐓r₀¹, 𝐓rₙ²)'    
  
  X = 𝐃*BHᵀ*𝐓r;
  Xᵀ = 𝐓rᵀ*𝐃*BHᵀ;
  
  𝚯 = 𝐃⁻¹*X
  𝚯ᵀ = -𝐃⁻¹*Xᵀ
  Ju = -𝐃⁻¹*𝐃*BT;   
  
  h = cᵢ(1)[1]/(m-1)
  ζ₀ = 40/h
  𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju
  
  Jbulk⁻¹*(𝐏 - 𝐓 - 𝐓ᵢ)
end

"""
Function to get the marker matrix for implementing the jump conditions on the interface
"""
function get_marker_matrix(m)
  W₁ = I(2) ⊗ I(m) ⊗ E1(1,1,m)
  W₂ = I(2) ⊗ I(m) ⊗ E1(m,m,m)
  Z₁ = I(2) ⊗ I(m) ⊗ E1(1,m,m)  
  Z₂ = I(2) ⊗ I(m) ⊗ E1(m,1,m)  
  mk1 = [-W₁  Z₁; -Z₂  W₂]
  mk2 = [-W₁  Z₁; Z₂  -W₂]
  mk1, mk2
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
    𝐪𝐫 = generate_2d_grid((m,m))
    global stima2 = 𝐊2(𝐪𝐫)
    𝐱𝐲₁ = Ω₁.(𝐪𝐫)
    𝐱𝐲₂ = Ω₂.(𝐪𝐫)        
    massma2 = blockdiag((I(2)⊗spdiagm(vec(ρ¹.(𝐱𝐲₁)))), (I(2)⊗spdiagm(vec(ρ².(𝐱𝐲₂)))))
    M⁺ = (massma2 - (Δt/2)^2*stima2)
    M⁻ = (massma2 + (Δt/2)^2*stima2)
    luM⁺ = factorize(M⁺)
      
    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
      
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
        gₙ = vcat(𝐠(t, (m,n), sbp_2d.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                 𝐠(t, (m,n), sbp_2d.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
        gₙ₊₁ = vcat(𝐠(t+Δt, (m,n), sbp_2d.norm, Ω₁, [1, 0, -1, 1], normals(Ω₁), σ¹),
                   𝐠(t+Δt, (m,n), sbp_2d.norm, Ω₂, [1, -1, -1, 0], normals(Ω₂), σ²))
          
        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(luM⁺, M⁻, massma2, fargs) # Function in "time-stepping.jl"
        (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(abs.(u₁))))
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
        max_err[i,Ni] = maximum(abs.(u₁ - vcat(eltocols(vec(U.(𝐱𝐲₁, t))), eltocols(vec(U.(𝐱𝐲₂, t))))))
      end
    end
      
    Hq = sbp_q.norm
    Hr = sbp_r.norm
    𝐇 = blockdiag((I(2) ⊗ Hq ⊗ Hr), (I(2) ⊗ Hq ⊗ Hr))
    e = u₁ - vcat(eltocols(vec(U.(𝐱𝐲₁, tf))), eltocols(vec(U.(𝐱𝐲₂, tf))))
    L²Error[Ni] = sqrt(e'*𝐇*e)
    println("Done N = "*string(m)*", L²Error = "*string(L²Error[Ni]))
  end
end
  
#= rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
@show L²Error
@show rate
=#
function get_sol_vector_from_raw_vector(sol, m, n)
  (reshape(sol[1:m^2], (m,m)), reshape(sol[m^2+1:m^2+n^2], (n,n)),
  reshape(sol[m^2+n^2+1:m^2+n^2+m^2], (m,m)), reshape(sol[m^2+n^2+m^2+1:m^2+n^2+m^2+n^2], (n,n)))
end
  
𝐪𝐫 = generate_2d_grid((N[end],N[end]));
q = LinRange(0,1,N[end]); r = LinRange(0,1,N[end]);
Uap₁, Vap₁, Uap₂, Vap₂ = get_sol_vector_from_raw_vector(u₁, N[end], N[end]);
𝐱𝐲₁ = vec(Ω₁.(𝐪𝐫));
𝐱𝐲₂ = vec(Ω₂.(𝐪𝐫));
Ue₁, Ue₂, Ve₁, Ve₂ = get_sol_vector_from_raw_vector(vcat(reduce(hcat, U.(𝐱𝐲₁,tf))', reduce(hcat, U.(𝐱𝐲₂,tf))'), N[end], N[end]);
  
# Plot the horizontal solution on the physical grid
plt1 = scatter(Tuple.(𝐱𝐲₁), zcolor=vec(Uap₁), label="", title="Approx. solution (u(x,y))", markersize=4, msw=0.1);
scatter!(plt1, Tuple.(𝐱𝐲₂), zcolor=vec(Uap₂), label="", markersize=4, msw=0.1);
plt2 = scatter(Tuple.(𝐱𝐲₁), zcolor=vec(Ue₁), label="", title="Exact solution (u(x,y))", markersize=4, msw=0.1);
scatter!(plt2, Tuple.(𝐱𝐲₂), zcolor=vec(Ue₂), label="", markersize=4, msw=0.1);
  
# Plot the vertical solution on the physical grid
plt3 = scatter(Tuple.(𝐱𝐲₁), zcolor=vec(Vap₁), label="", title="Approx. solution (v(x,y))", markersize=4, msw=0.1);
scatter!(plt3, Tuple.(𝐱𝐲₂), zcolor=vec(Vap₂), label="", markersize=4, msw=0.1);
plt4 = scatter(Tuple.(𝐱𝐲₁), zcolor=vec(Ve₁), label="", title="Exact solution (v(x,y))", markersize=4, msw=0.1);
scatter!(plt4, Tuple.(𝐱𝐲₂), zcolor=vec(Ve₂), label="", markersize=4, msw=0.1);
  
# Plot the exact solution and the approximate solution together.
plt1_3 = plot(plt1, plt2, layout=(1,2), size=(800,400));
plt2_4 = plot(plt3, plt4, layout=(1,2), size=(800,400));
  
plt5 = plot(h1, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2, size=(800,800));
scatter!(plt5, h1, L²Error, markersize=4, label="");
plot!(plt5, h1, h1.^4, label="O(h⁴)", lw=2);

plt6_1 = scatter(Tuple.(𝐱𝐲₁), size=(800,800), markersize=4, xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Layer 1", msw=0.1)
plt6_2 = scatter(Tuple.(𝐱𝐲₂), size=(800,800), markersize=2, markercolor="red", xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Layer 2", msw=0.1)
plt6_1_2 = plot(plt6_1, plt6_2, layout=(2,1))
plt6_3 = scatter(Tuple.(𝐪𝐫 |> vec), xlabel="q", ylabel="r", label="Reference Domain", markersize=4, markercolor="white", aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), msw=0.1);
plt6 = plot(plt6_1_2, plt6_3, layout=(1,2));
  
#= # Run these from the Project folder
savefig(plt13, "./Images/2-layer/horizontal-disp.png")
savefig(plt24, "./Images/2-layer/vertical-disp.png")
savefig(plt9, "./Images/2-layer/rate.png")
savefig(plt10, "./Images/2-layer/domain.png") =#
  
plt7 = scatter(Tuple.(𝐱𝐲₁ |> vec), zcolor=vec(abs.(Uap₁-Ue₁)), label="", title="ΔU", markersize=4, msw=0.1);
scatter!(plt7, Tuple.(𝐱𝐲₂ |> vec), zcolor=vec(abs.(Uap₂-Ue₂)), label="", markersize=4, msw=0.1);
plt8 = scatter(Tuple.(𝐱𝐲₁ |> vec), zcolor=vec(abs.(Vap₁-Ve₁)), label="", title="ΔV", markersize=4, msw=0.1);
scatter!(plt8, Tuple.(𝐱𝐲₂ |> vec), zcolor=vec(abs.(Vap₂-Ve₂)), label="", markersize=4, msw=0.1);
plt7_8 = plot(plt7, plt8, layout=(1,2))

# plt14 = plot();
# for i=1:lastindex(h1)
#   t_arr = LinRange(0,tf,ntime)
#   plot!(plt14, t_arr, max_err[:,i], label="h="*string(h1[i]), yscale=:log10, lw=1.5, legend=:bottomright)
# end