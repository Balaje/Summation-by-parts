# include("2d_elasticity_problem.jl")

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = [q, 0.0]; # Bottom boundary. Also the interface
c₂¹(r) = [1.0, r]; # Right boundary
c₃¹(q) = [q, 1.0 + 0.0*sin(2π*q)]; # Top boundary
# Layer 2 (q,r) ∈ [0,1] × [-1,0]
c₀²(r) = [0.0, r-1]; # Left boundary
c₁²(q) = [q, -1.0]; # Bottom boundary. 
c₂²(r) = [1.0, r-1]; # Right boundary
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
μ(x) = E/(2*(1+ν)) + 0.0*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;
λ(x) = E*ν/((1+ν)*(1-2ν)) + 0.0*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;

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
    S'*𝒫(x)*S
end

# Extract the property matrices
Aₜ¹(qr) = t𝒫(Ω₁,qr)[1:2, 1:2];
Bₜ¹(qr) = t𝒫(Ω₁,qr)[3:4, 3:4];
Cₜ¹(qr) = t𝒫(Ω₁,qr)[1:2, 3:4];
Aₜ²(qr) = t𝒫(Ω₂,qr)[1:2, 1:2];
Bₜ²(qr) = t𝒫(Ω₂,qr)[3:4, 3:4];
Cₜ²(qr) = t𝒫(Ω₂,qr)[1:2, 3:4];

M = 21
𝐪𝐫 = generate_2d_grid((M,M))
function 𝐊2(𝐪𝐫)
    # Property coefficients on the first layer
    Aₜ¹¹₁(x) = Aₜ¹(x)[1,1]
    Aₜ¹²₁(x) = Aₜ¹(x)[1,2]
    Aₜ²¹₁(x) = Aₜ¹(x)[2,1]
    Aₜ²²₁(x) = Aₜ¹(x)[2,2]

    Bₜ¹¹₁(x) = Bₜ¹(x)[1,1]
    Bₜ¹²₁(x) = Bₜ¹(x)[1,2]
    Bₜ²¹₁(x) = Bₜ¹(x)[2,1]
    Bₜ²²₁(x) = Bₜ¹(x)[2,2]

    Cₜ¹¹₁(x) = Cₜ¹(x)[1,1]
    Cₜ¹²₁(x) = Cₜ¹(x)[1,2]
    Cₜ²¹₁(x) = Cₜ¹(x)[2,1]
    Cₜ²²₁(x) = Cₜ¹(x)[2,2]

    # Property coefficients on the second layer
    Aₜ¹¹₂(x) = Aₜ²(x)[1,1]
    Aₜ¹²₂(x) = Aₜ²(x)[1,2]
    Aₜ²¹₂(x) = Aₜ²(x)[2,1]
    Aₜ²²₂(x) = Aₜ²(x)[2,2]

    Bₜ¹¹₂(x) = Bₜ²(x)[1,1]
    Bₜ¹²₂(x) = Bₜ²(x)[1,2]
    Bₜ²¹₂(x) = Bₜ²(x)[2,1]
    Bₜ²²₂(x) = Bₜ²(x)[2,2]

    Cₜ¹¹₂(x) = Cₜ²(x)[1,1]
    Cₜ¹²₂(x) = Cₜ²(x)[1,2]
    Cₜ²¹₂(x) = Cₜ²(x)[2,1]
    Cₜ²²₂(x) = Cₜ²(x)[2,2]

    detJ₁(x) = (det∘J)(x,Ω₁)
    detJ₂(x) = (det∘J)(x,Ω₂)

    # Get the norm matrices (Same for both layers)
    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
    𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm

    # Bulk matrices for the first layer
    DqqA₁ = [Dqq(detJ₁.(𝐪𝐫).*Aₜ¹¹₁.(𝐪𝐫)) Dqq(detJ₁.(𝐪𝐫).*Aₜ¹²₁.(𝐪𝐫));
             Dqq(detJ₁.(𝐪𝐫).*Aₜ²¹₁.(𝐪𝐫)) Dqq(detJ₁.(𝐪𝐫).*Aₜ²²₁.(𝐪𝐫))]
    DrrB₁ = [Drr(detJ₁.(𝐪𝐫).*Bₜ¹¹₁.(𝐪𝐫)) Drr(detJ₁.(𝐪𝐫).*Bₜ¹²₁.(𝐪𝐫));
             Drr(detJ₁.(𝐪𝐫).*Bₜ²¹₁.(𝐪𝐫)) Drr(detJ₁.(𝐪𝐫).*Bₜ²²₁.(𝐪𝐫))]
    DqrC₁ = [Dqr(detJ₁.(𝐪𝐫).*Cₜ¹¹₁.(𝐪𝐫)) Dqr(detJ₁.(𝐪𝐫).*Cₜ¹²₁.(𝐪𝐫));
             Dqr(detJ₁.(𝐪𝐫).*Cₜ²¹₁.(𝐪𝐫)) Dqr(detJ₁.(𝐪𝐫).*Cₜ²²₁.(𝐪𝐫))]
    DrqCᵀ₁ = [Drq(detJ₁.(𝐪𝐫).*Cₜ¹¹₁.(𝐪𝐫)) Drq(detJ₁.(𝐪𝐫).*Cₜ²¹₁.(𝐪𝐫));
              Drq(detJ₁.(𝐪𝐫).*Cₜ¹²₁.(𝐪𝐫)) Drq(detJ₁.(𝐪𝐫).*Cₜ²²₁.(𝐪𝐫))]    
    𝐏₁ = DqqA₁ + DrrB₁ + DqrC₁ + DrqCᵀ₁

    # Bulk matrices for the second layer
    DqqA₂ = [Dqq(detJ₂.(𝐪𝐫).*Aₜ¹¹₂.(𝐪𝐫)) Dqq(detJ₂.(𝐪𝐫).*Aₜ¹²₂.(𝐪𝐫));
             Dqq(detJ₂.(𝐪𝐫).*Aₜ²¹₂.(𝐪𝐫)) Dqq(detJ₂.(𝐪𝐫).*Aₜ²²₂.(𝐪𝐫))]
    DrrB₂ = [Drr(detJ₂.(𝐪𝐫).*Bₜ¹¹₂.(𝐪𝐫)) Drr(detJ₁.(𝐪𝐫).*Bₜ¹²₂.(𝐪𝐫));
             Drr(detJ₂.(𝐪𝐫).*Bₜ²¹₂.(𝐪𝐫)) Drr(detJ₁.(𝐪𝐫).*Bₜ²²₂.(𝐪𝐫))]
    DqrC₂ = [Dqr(detJ₂.(𝐪𝐫).*Cₜ¹¹₂.(𝐪𝐫)) Dqr(detJ₁.(𝐪𝐫).*Cₜ¹²₂.(𝐪𝐫));
             Dqr(detJ₂.(𝐪𝐫).*Cₜ²¹₂.(𝐪𝐫)) Dqr(detJ₁.(𝐪𝐫).*Cₜ²²₂.(𝐪𝐫))]
    DrqCᵀ₂ = [Drq(detJ₂.(𝐪𝐫).*Cₜ¹¹₂.(𝐪𝐫)) Drq(detJ₁.(𝐪𝐫).*Cₜ²¹₂.(𝐪𝐫));
              Drq(detJ₂.(𝐪𝐫).*Cₜ¹²₂.(𝐪𝐫)) Drq(detJ₁.(𝐪𝐫).*Cₜ²²₂.(𝐪𝐫))]    
    𝐏₂ = DqqA₂ + DrrB₂ + DqrC₂ + DrqCᵀ₂

    # Traction matrices for the first layer
    TqAC₁ = [Tq(Aₜ¹¹₁.(𝐪𝐫), Cₜ¹¹₁.(𝐪𝐫)) Tq(Aₜ¹²₁.(𝐪𝐫), Cₜ¹²₁.(𝐪𝐫));
             Tq(Aₜ²¹₁.(𝐪𝐫), Cₜ²¹₁.(𝐪𝐫)) Tq(Aₜ²²₁.(𝐪𝐫), Cₜ²²₁.(𝐪𝐫))]
    TrCB₁ = [Tr(Cₜ¹¹₁.(𝐪𝐫), Bₜ¹¹₁.(𝐪𝐫)) Tr(Cₜ²¹₁.(𝐪𝐫), Bₜ¹²₁.(𝐪𝐫));
             Tr(Cₜ¹²₁.(𝐪𝐫), Bₜ²¹₁.(𝐪𝐫)) Tr(Cₜ²²₁.(𝐪𝐫), Bₜ²²₁.(𝐪𝐫))]

    # Traction matrices for the second layer
    TqAC₂ = [Tq(Aₜ¹¹₂.(𝐪𝐫), Cₜ¹¹₂.(𝐪𝐫)) Tq(Aₜ¹²₂.(𝐪𝐫), Cₜ¹²₂.(𝐪𝐫));
             Tq(Aₜ²¹₂.(𝐪𝐫), Cₜ²¹₂.(𝐪𝐫)) Tq(Aₜ²²₂.(𝐪𝐫), Cₜ²²₂.(𝐪𝐫))]
    TrCB₂ = [Tr(Cₜ¹¹₂.(𝐪𝐫), Bₜ¹¹₂.(𝐪𝐫)) Tr(Cₜ²¹₂.(𝐪𝐫), Bₜ¹²₂.(𝐪𝐫));
             Tr(Cₜ¹²₂.(𝐪𝐫), Bₜ²¹₂.(𝐪𝐫)) Tr(Cₜ²²₂.(𝐪𝐫), Bₜ²²₂.(𝐪𝐫))]

    detJ1₁ = [1,1] ⊗ vec(detJ₁.(𝐪𝐫))
    detJ1₂ = [1,1] ⊗ vec(detJ₂.(𝐪𝐫))

    𝐏 = blockdiag(spdiagm(detJ1₁.^-1)*𝐏₁, spdiagm(detJ1₂.^-1)*𝐏₂)

    𝐓q₁ = TqAC₁
    𝐓r₁ = TrCB₁
    𝐓q₂ = TqAC₂
    𝐓r₂ = TrCB₂

    𝐓 = blockdiag(-(I(2) ⊗ 𝐇q₀)*(𝐓q₁) + (I(2) ⊗ 𝐇qₙ)*(𝐓q₁) + (I(2) ⊗ 𝐇rₙ)*(𝐓r₁),
                  -(I(2) ⊗ 𝐇q₀)*(𝐓q₂) + (I(2) ⊗ 𝐇qₙ)*(𝐓q₂) + -(I(2) ⊗ 𝐇r₀)*(𝐓r₂))

    # Traction on the interface
    Id₁ = spdiagm(ones(m^2+n^2))
    B̃ = [Id₁ -Id₁; -Id₁ Id₁]
    B̂ = [Id₁ Id₁; -Id₁ -Id₁]
    𝐇ᵢ = blockdiag((I(2) ⊗ 𝐇r₀), (I(2) ⊗ 𝐇rₙ))
    𝐓r = blockdiag(𝐓r₁, 𝐓r₂)
    ζ₀ = 20*(m-1)^3
    𝐓ᵢ = 𝐇ᵢ*(-0.5*B̂*𝐓r + 0.5*𝐓r'*B̂' + ζ₀*B̃)

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
N = [21,31,41,51]
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
