include("2d_elasticity_problem.jl");

## Define the physical domain
c₀(r) = @SVector [0.0 + 0.1*sin(π*r), r] # Left boundary 
c₁(q) = @SVector [q, 0.0 + 0.1*sin(2π*q)] # Bottom boundary
c₂(r) = @SVector [1.0 + 0.1*sin(π*r), r] # Right boundary
c₃(q) = @SVector [q, 1.0 + 0.1*sin(2π*q)] # Top boundary
domain = domain_2d(c₀, c₁, c₂, c₃)
Ω(qr) = S(qr, domain)

## Define the material properties on the physical grid
const E = 1.0;
const ν = 0.33;

"""
The Lamé parameters μ, λ
"""
μ(x) = E/(2*(1+ν)) + 0.1*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;
λ(x) = E*ν/((1+ν)*(1-2ν)) + 0.1*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;

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
Function to generate the stiffness matrices
"""
function 𝐊(𝐪𝐫)
  detJ(x) = (det∘J)(x,Ω)    
  
  Pqr = P2R.(𝒫,Ω,𝐪𝐫) # Property matrix evaluated at grid points    
  𝐏 = Pᴱ(Dᴱ(Pqr)) # Elasticity bulk differential operator
  𝐓 = Tᴱ(Pqr) # Elasticity Traction operator
  𝐓q = 𝐓.A
  𝐓r = 𝐓.B
  
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm
  
  detJ1 = [1,1] ⊗ vec(detJ.(𝐪𝐫))
  spdiagm(detJ1.^-1)*(𝐏 - (-(I(2) ⊗ 𝐇q₀)*(𝐓q) + (I(2) ⊗ 𝐇qₙ)*(𝐓q) - (I(2) ⊗ 𝐇r₀)*(𝐓r) + (I(2) ⊗ 𝐇rₙ)*(𝐓r)))
end

"""
Neumann boundary condition vector
"""
function 𝐠(t::Float64, mn::Tuple{Int64,Int64}, norm)
  m,n= mn
  q = LinRange(0,1,m); r = LinRange(0,1,n)
  𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = norm
  
  bvals_q₀ = reduce(hcat, [J⁻¹s(@SVector[0.0, rᵢ], Ω, @SVector[-1.0,0.0])*g(t, c₀, rᵢ, 1) for rᵢ in r])
  bvals_r₀ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 0.0], Ω, @SVector[0.0,-1.0])*g(t, c₁, qᵢ, -1) for qᵢ in q])
  bvals_qₙ = reduce(hcat, [J⁻¹s(@SVector[1.0, rᵢ], Ω, @SVector[1.0,0.0])*g(t, c₂, rᵢ, -1) for rᵢ in r])
  bvals_rₙ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 1.0], Ω, @SVector[0.0,1.0])*g(t, c₃, qᵢ, 1) for qᵢ in q])
  
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
N = [21,31,41,51]
h = 1 ./(N .- 1)
L²Error = zeros(Float64, length(N))
tf = 0.5
const Δt = 1e-3
ntime = ceil(Int, tf/Δt)

for (m,i) in zip(N, 1:length(N))
  let
    𝐪𝐫 = generate_2d_grid((m,m))
    global stima = 𝐊(𝐪𝐫)
    𝐱𝐲 = Ω.(𝐪𝐫)
    ρᵢ = ρ.(𝐱𝐲)
    massma = I(2) ⊗ spdiagm(vec(ρᵢ))
    M⁺ = (massma - (Δt/2)^2*stima)
    M⁻ = (massma + (Δt/2)^2*stima)
    luM⁺ = factorize(M⁺)
    
    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
    
    let
      u₀ = eltocols(vec(U.(𝐱𝐲,0.0)))
      v₀ = eltocols(vec(Uₜ.(𝐱𝐲,0.0)))
      global u₁ = zero(u₀)
      global v₁ = zero(v₀)
      t = 0.0
      for i=1:ntime
        Fₙ = eltocols(vec(F.(𝐱𝐲, t)))
        Fₙ₊₁ = eltocols(vec(F.(𝐱𝐲, t+Δt)))
        gₙ = 𝐠(t, (m,n), sbp_2d.norm)
        gₙ₊₁ = 𝐠(t+Δt, (m,n), sbp_2d.norm)
        
        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(luM⁺, M⁻, massma, fargs) # Function in "time-stepping.jl"
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
      end
    end
    
    Hq = sbp_q.norm
    Hr = sbp_r.norm
    𝐇 = (I(2) ⊗ Hq ⊗ Hr)
    e = u₁ - eltocols(vec(U.(𝐱𝐲, tf)))
    L²Error[i] = sqrt(e'*𝐇*e)
    println("Done N = "*string(m)*", L²Error = "*string(L²Error[i]))
  end
end

rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
@show L²Error
@show rate

function get_sol_vector_from_raw_vector(sol, m, n)
  (reshape(sol[1:m^2], (m,m)), reshape(sol[1:n^2], (n,n)))
end

𝐪𝐫 = generate_2d_grid((N[end],N[end]));
q = LinRange(0,1,N[end]); r = LinRange(0,1,N[end]);
Uap, Vap = get_sol_vector_from_raw_vector(u₁, N[end], N[end]);
𝐱𝐲 = vec(Ω.(𝐪𝐫));
Ue, Ve = get_sol_vector_from_raw_vector(reduce(hcat, U.(𝐱𝐲,tf))', N[end], N[end]);
plt1 = contourf(q, r, Uap, title="u₁ Approximate");
plt2 = contourf(q, r, Ue, title="u₁ Exact");
plt3 = contourf(q, r, Vap, title="v₁ Approximate");
plt4 = contourf(q, r, Ve, title="v₁ Exact");
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));

plt5 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
plot!(plt5, h, h.^4, label="O(h⁴)", lw=1);
plt6_1 = scatter(Tuple.(𝐱𝐲 |> vec), size=(700,800), markersize=0.5, xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Physical Domain")
plt6_2 = scatter(Tuple.(𝐪𝐫 |> vec), xlabel="q", ylabel="r", label="Reference Domain", markersize=0.5);
plt6 = plot(plt6_1, plt6_2, layout=(1,2));
plt56 = plot(plt6, plt5, layout=(2,1), size=(700,800));
