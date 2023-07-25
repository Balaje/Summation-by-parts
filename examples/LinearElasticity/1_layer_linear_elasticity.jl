# include("2d_elasticity_problem.jl");

using SBP
using StaticArrays
using LinearAlgebra

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
C₁₂(x) = λ(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫(x) = @SMatrix [c₁₁(x) 0 0 c₁₂(x); 0 c₃₃(x) c₃₃(x) 0; 0 c₃₃(x) c₃₃(x) 0; c₁₂(x) 0 0 c₂₂(x)];


## Transform the material properties to the reference grid
function t𝒫(𝒮, qr)
    x = 𝒮(qr)
    invJ = J⁻¹(qr, 𝒮)
    S = invJ ⊗ I(2)
    S'*𝒫(x)*S
end

# Extract the property matrices
Aₜ(qr) = t𝒫(Ω,qr)[1:2, 1:2];
Bₜ(qr) = t𝒫(Ω,qr)[3:4, 3:4];
Cₜ(qr) = t𝒫(Ω,qr)[1:2, 3:4];

# Coefficients

M = 21
𝐪𝐫 = generate_2d_grid((M,M))
function 𝐊(𝐪𝐫)
    Aₜ¹¹(x) = Aₜ(x)[1,1]
    Aₜ¹²(x) = Aₜ(x)[1,2]
    Aₜ²¹(x) = Aₜ(x)[2,1]
    Aₜ²²(x) = Aₜ(x)[2,2]

    Bₜ¹¹(x) = Bₜ(x)[1,1]
    Bₜ¹²(x) = Bₜ(x)[1,2]
    Bₜ²¹(x) = Bₜ(x)[2,1]
    Bₜ²²(x) = Bₜ(x)[2,2]

    Cₜ¹¹(x) = Cₜ(x)[1,1]
    Cₜ¹²(x) = Cₜ(x)[1,2]
    Cₜ²¹(x) = Cₜ(x)[2,1]
    Cₜ²²(x) = Cₜ(x)[2,2]

    detJ(x) = (det∘J)(x,Ω)

    DqqA = [Dqq(detJ.(𝐪𝐫).*Aₜ¹¹.(𝐪𝐫)) Dqq(detJ.(𝐪𝐫).*Aₜ¹².(𝐪𝐫));
            Dqq(detJ.(𝐪𝐫).*Aₜ²¹.(𝐪𝐫)) Dqq(detJ.(𝐪𝐫).*Aₜ²².(𝐪𝐫))]
    DrrB = [Drr(detJ.(𝐪𝐫).*Bₜ¹¹.(𝐪𝐫)) Drr(detJ.(𝐪𝐫).*Bₜ¹².(𝐪𝐫));
            Drr(detJ.(𝐪𝐫).*Bₜ²¹.(𝐪𝐫)) Drr(detJ.(𝐪𝐫).*Bₜ²².(𝐪𝐫))]
    DqrC = [Dqr(detJ.(𝐪𝐫).*Cₜ¹¹.(𝐪𝐫)) Dqr(detJ.(𝐪𝐫).*Cₜ¹².(𝐪𝐫));
            Dqr(detJ.(𝐪𝐫).*Cₜ²¹.(𝐪𝐫)) Dqr(detJ.(𝐪𝐫).*Cₜ²².(𝐪𝐫))]
    DrqCᵀ = [Drq(detJ.(𝐪𝐫).*Cₜ¹¹.(𝐪𝐫)) Drq(detJ.(𝐪𝐫).*Cₜ²¹.(𝐪𝐫));
             Drq(detJ.(𝐪𝐫).*Cₜ¹².(𝐪𝐫)) Drq(detJ.(𝐪𝐫).*Cₜ²².(𝐪𝐫))]
    
    𝐏 = DqqA + DrrB + DqrC + DrqCᵀ

    TqAC = [Tq(Aₜ¹¹.(𝐪𝐫), Cₜ¹¹.(𝐪𝐫)) Tq(Aₜ¹².(𝐪𝐫), Cₜ¹².(𝐪𝐫));
            Tq(Aₜ²¹.(𝐪𝐫), Cₜ²¹.(𝐪𝐫)) Tq(Aₜ²².(𝐪𝐫), Cₜ²².(𝐪𝐫))]
    TrCB = [Tr(Cₜ¹¹.(𝐪𝐫), Bₜ¹¹.(𝐪𝐫)) Tr(Cₜ²¹.(𝐪𝐫), Bₜ¹².(𝐪𝐫));
            Tr(Cₜ¹².(𝐪𝐫), Bₜ²¹.(𝐪𝐫)) Tr(Cₜ²².(𝐪𝐫), Bₜ²².(𝐪𝐫))]

    m, n = size(𝐪𝐫)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
    
    𝐇q₀, 𝐇qₙ, 𝐇r₀, 𝐇rₙ = sbp_2d.norm

    𝐓q = TqAC
    𝐓r = TrCB

    detJ1 = [1,1] ⊗ vec(detJ.(𝐪𝐫))
    spdiagm(detJ1.^-1)*𝐏 - (-(I(2) ⊗ 𝐇q₀)*(𝐓q) + (I(2) ⊗ 𝐇qₙ)*(𝐓q)
                            -(I(2) ⊗ 𝐇r₀)*(𝐓r) + (I(2) ⊗ 𝐇rₙ)*(𝐓r))
end

stima = 𝐊(𝐪𝐫)
