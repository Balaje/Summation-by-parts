include("2d_elasticity_problem.jl")

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
cᵢ(q) = [q, 0.0 + 0.0*sin(2π*q)];
# Define the rest of the boundary
c₀¹(r) = [0.0 + 0.0*sin(2π*r), r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. Also the interface
c₂¹(r) = [1.0 + 0.0*sin(2π*r), r]; # Right boundary
c₃¹(q) = [q, 1.0 + 0.0*sin(2π*q)]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [-1,0]
c₀²(r) = [0.0 + 0.0*sin(2π*r), r-1]; # Left boundary
c₁²(q) = [q, -1.0 + 0.0*sin(2π*q)]; # Bottom boundary. 
c₂²(r) = [1.0 + 0.0*sin(2π*r), r-1]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₁(qr) = S(qr, domain₁)
Ω₂(qr) = S(qr, domain₂)

## Define the material properties on the physical grid
const E = 1.0;
const ν = 0.33;

"""
The Lamé parameters μ, λ
"""
μ(x) = E/(2*(1+ν))
λ(x) = E*ν/((1+ν)*(1-2ν))

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


## Transform the material properties to the reference grid
function t𝒫(𝒮, qr)
    x = 𝒮(qr)
    invJ = J⁻¹(qr, 𝒮)
    S = invJ ⊗ I(2)
    m,n = size(S)
    SMatrix{m,n,Float64}(S'*𝒫(x)*S)
end

function 𝐊2(𝐪𝐫)
    # Get the bulk and the traction operator for the 1st layer
    detJ₁(x) = (det∘J)(x,Ω₁)
    detJ₁𝒫(x) = detJ₁(x)*t𝒫(Ω₁, x)
    Pqr₁ = t𝒫.(Ω₁,𝐪𝐫) # Property matrix evaluated at grid points
    JPqr₁ = detJ₁𝒫.(𝐪𝐫) # Property matrix * det(J)
    𝐏₁ = Pᴱ(Dᴱ(JPqr₁)) # Elasticity bulk differential operator
    𝐓₁ = Tᴱ(Pqr₁) # Elasticity Traction operator
    𝐓q₁ = 𝐓₁.A
    𝐓r₁ = 𝐓₁.B

    # Get the bulk and the traction operator for the 2nd layer
    detJ₂(x) = (det∘J)(x,Ω₂)
    detJ₂𝒫(x) = detJ₂(x)*t𝒫(Ω₂, x)
    Pqr₂ = t𝒫.(Ω₂,𝐪𝐫) # Property matrix evaluated at grid points
    JPqr₂ = detJ₂𝒫.(𝐪𝐫) # Property matrix * det(J)
    𝐏₂ = Pᴱ(Dᴱ(JPqr₂)) # Elasticity bulk differential operator
    𝐓₂ = Tᴱ(Pqr₂) # Elasticity Traction operator
    𝐓q₂ = 𝐓₂.A
    𝐓r₂ = 𝐓₂.B

    # Get the norm matrices (Same for both layers)
    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
    𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm

    # Determinants of the transformation
    detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
    detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫))    

    # Combine the operators
    𝐏 = blockdiag(spdiagm(detJ1₁.^-1)*𝐏₁, spdiagm(detJ1₂.^-1)*𝐏₂)
    𝐓 = blockdiag(-(I(2) ⊗ 𝐇q₀)*(𝐓q₁) + (I(2) ⊗ 𝐇qₙ)*(𝐓q₁) + (I(2) ⊗ 𝐇rₙ)*(𝐓r₁),
                  -(I(2) ⊗ 𝐇q₀)*(𝐓q₂) + (I(2) ⊗ 𝐇qₙ)*(𝐓q₂) + -(I(2) ⊗ 𝐇r₀)*(𝐓r₂))

    # Traction on the interface
    q = LinRange(0,1,m)
    sJ₁ = spdiagm([J⁻¹s([qᵢ,0.0], Ω₁, [0,-1])^-1 for qᵢ in q]) ⊗ SBP.SBP_2d.E1(1,n)
    sJ₂ = spdiagm([J⁻¹s([qᵢ,1.0], Ω₂, [0,1])^-1 for qᵢ in q]) ⊗ SBP.SBP_2d.E1(m,n)

    Id₃ = spdiagm(ones(2*m*n))
    𝐃 = blockdiag((I(2)⊗𝐇r₀), (I(2)⊗𝐇rₙ))
    BH = [-Id₃ -Id₃; Id₃ Id₃]
    BHᵀ = [Id₃ -Id₃; Id₃ -Id₃]
    BT = [Id₃ -Id₃; -Id₃ Id₃]
    
    𝐓r = blockdiag(([1 1; 1 1]⊗sJ₁).*𝐓r₁, ([1 1; 1 1]⊗sJ₂).*𝐓r₂)

    𝚯 = 𝐃*BHᵀ*𝐓r;
    𝚯ᵀ = 𝐃*𝐓r'*BH;
    Ju = 𝐃*BT

    ζ₀ = 10*(m-1)
    𝐓ᵢ = 0.5*𝚯 + 0.5*𝚯ᵀ + ζ₀*Ju

    𝐏 - 𝐓 - 𝐓ᵢ
end

"""
Neumann boundary condition vector
"""
function 𝐠(t::Float64, mn::Tuple{Int64,Int64}, norm, Ω, P, C)
    m,n= mn
    q = LinRange(0,1,m); r = LinRange(0,1,n)
    𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = norm
    P1, P2, P3, P4 = P
    c₀, c₁, c₂, c₃ = C
    
    bvals_q₀ = reduce(hcat, [J⁻¹s(@SVector[0.0, rᵢ], Ω, @SVector[-1.0,0.0])*g(t, c₀, rᵢ, P1) for rᵢ in r])
    bvals_r₀ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 0.0], Ω, @SVector[0.0,-1.0])*g(t, c₁, qᵢ, P2) for qᵢ in q])
    bvals_qₙ = reduce(hcat, [J⁻¹s(@SVector[1.0, rᵢ], Ω, @SVector[1.0,0.0])*g(t, c₂, rᵢ, P3) for rᵢ in r])
    bvals_rₙ = reduce(hcat, [J⁻¹s(@SVector[qᵢ, 1.0], Ω, @SVector[0.0,1.0])*g(t, c₃, qᵢ, P4) for qᵢ in q])

    E1(i,M) = diag(SBP.SBP_2d.E1(i,M))
    bq₀ = (E1(1,2) ⊗ E1(1,m) ⊗ (bvals_q₀[1,:])) + (E1(2,2) ⊗ E1(1,m) ⊗ (bvals_q₀[2,:]))
    br₀ = (E1(1,2) ⊗ (bvals_r₀[1,:]) ⊗ E1(1,n)) + (E1(2,2) ⊗ (bvals_r₀[2,:]) ⊗ E1(1,n))
    bqₙ = (E1(1,2) ⊗ E1(m,n) ⊗ (bvals_qₙ[1,:])) + (E1(2,2) ⊗ E1(m,n) ⊗ (bvals_qₙ[2,:]))
    brₙ = (E1(1,2) ⊗ (bvals_rₙ[1,:]) ⊗ E1(m,n)) + (E1(2,2) ⊗ (bvals_rₙ[2,:]) ⊗ E1(m,n))

    collect((I(2)⊗𝐇r₀)*br₀ + (I(2)⊗𝐇rₙ)*brₙ + (I(2)⊗𝐇q₀)*bq₀ + (I(2)⊗𝐇qₙ)*bqₙ)
end

#################################
# Now begin solving the problem #
#################################
N = [21,31,41,51,61]
h = 1 ./(N .- 1)
L²Error = zeros(Float64, length(N))
tf = 0.5
Δt = 1e-3
ntime = ceil(Int, tf/Δt)

for (m,i) in zip(N, 1:length(N))
    let
        𝐪𝐫 = generate_2d_grid((m,m))
        stima2 = 𝐊2(𝐪𝐫)
        𝐱𝐲₁ = Ω₁.(𝐪𝐫)
        𝐱𝐲₂ = Ω₂.(𝐪𝐫)        
        massma2 = blockdiag((I(2)⊗spdiagm(vec(ρ.(𝐱𝐲₁)))), (I(2)⊗spdiagm(vec(ρ.(𝐱𝐲₂)))))
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
                Fₙ = vcat(eltocols(vec(F.(𝐱𝐲₁, t))), eltocols(vec(F.(𝐱𝐲₂, t))))
                Fₙ₊₁ = vcat(eltocols(vec(F.(𝐱𝐲₁, t+Δt))), eltocols(vec(F.(𝐱𝐲₂, t+Δt))))
                gₙ = vcat(𝐠(t, (m,n), sbp_2d.norm, Ω₁, [1, 0, -1, 1], [c₀¹, c₁¹, c₂¹, c₃¹]),
                          𝐠(t, (m,n), sbp_2d.norm, Ω₂, [1, -1, -1, 0], [c₀², c₁², c₂², c₃²]))
                gₙ₊₁ = vcat(𝐠(t+Δt, (m,n), sbp_2d.norm, Ω₁, [1, 0, -1, 1], [c₀¹, c₁¹, c₂¹, c₃¹]),
                            𝐠(t+Δt, (m,n), sbp_2d.norm, Ω₂, [1, -1, -1, 0], [c₀², c₁², c₂², c₃²]))

                rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
                fargs = Δt, u₀, v₀, rhs
                u₁,v₁ = CN(luM⁺, M⁻, massma2, fargs) # Function in "time-stepping.jl"
                t = t+Δt
                u₀ = u₁
                v₀ = v₁
            end
        end

        Hq = sbp_q.norm
        Hr = sbp_r.norm
        𝐇 = blockdiag((I(2) ⊗ Hq ⊗ Hr), (I(2) ⊗ Hq ⊗ Hr))
        e = u₁ - vcat(eltocols(vec(U.(𝐱𝐲₁, tf))), eltocols(vec(U.(𝐱𝐲₂, tf))))
        L²Error[i] = sqrt(e'*𝐇*e)
        println("Done N = "*string(m)*", L²Error = "*string(L²Error[i]))
    end
end

rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
@show L²Error
@show rate

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
plt1 = contourf(q, r, Uap₁, title="u₁ Approximate (Layer 1)");
plt2 = contourf(q, r, Ue₁, title="u₁ Exact (Layer 1)");
plt3 = contourf(q, r, Vap₁, title="v₁ Approximate (Layer 1)");
plt4 = contourf(q, r, Ve₁, title="v₁ Exact (Layer 1)");
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));

plt5 = contourf(q, r, Uap₂, title="u₁ Approximate (Layer 2)");
plt6 = contourf(q, r, Ue₂, title="u₁ Exact (Layer 2)");
plt7 = contourf(q, r, Vap₂, title="v₁ Approximate (Layer 2)");
plt8 = contourf(q, r, Ve₂, title="v₁ Exact (Layer 2)");
plt56 = plot(plt5, plt6, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));
plt78 = plot(plt7, plt8, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));

plt9 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
plot!(plt9, h, h.^4, label="O(h⁴)", lw=1);
plt10_1 = scatter(Tuple.(𝐱𝐲₁ |> vec), size=(700,800), markersize=0.5, xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Physical Domain")
plt10_2 = scatter(Tuple.(𝐱𝐲₂ |> vec), size=(700,800), markersize=0.5, markercolor="red", xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Physical Domain")
plt10_12 = plot(plt10_1, plt10_2, layout=(2,1))
plt10_3 = scatter(Tuple.(𝐪𝐫 |> vec), xlabel="q", ylabel="r", label="Reference Domain", markersize=0.5);
plt10 = plot(plt10_12, plt10_3, layout=(1,2));
plt910 = plot(plt9, plt10, layout=(2,1), size=(700,800));

plt11_1 = contourf(q, r, abs.(Uap₁ - Ue₁))
plt11_2 = contourf(q, r, abs.(Uap₂ - Ue₂))
plot(plt11_1, plt11_2)
