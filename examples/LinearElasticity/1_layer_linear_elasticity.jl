# include("2d_elasticity_problem.jl");

using SBP
using StaticArrays

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
c₁₂(x) = λ(x)

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
    invJ = J⁻¹(𝒮, qr)
    S = invJ ⊗ I(2)
    S'*𝒫(x)*S
end

# Extract the property matrices
Aₜ(qr) = t𝒫(Ω,qr)[1:2, 1:2];
Bₜ(qr) = t𝒫(Ω,qr)[3:4, 3:4];
Cₜ(qr) = t𝒫(Ω,qr)[1:2, 3:4];

# Coefficients

M = 21
qr = generate_2d_grid((M,M))
function 𝐊(qr)
    Aₜ¹¹(qr) = Aₜ(qr)[1,1]
    Aₜ¹²(qr) = Aₜ(qr)[1,2]
    Aₜ²¹(qr) = Aₜ(qr)[2,1]
    Aₜ²²(qr) = Aₜ(qr)[2,2]
    Bₜ¹¹(qr) = Bₜ(qr)[1,1]
    Bₜ¹²(qr) = Bₜ(qr)[1,2]
    Bₜ²¹(qr) = Bₜ(qr)[2,1]
    Bₜ²²(qr) = Bₜ(qr)[2,2]
    Cₜ¹¹(qr) = Cₜ(qr)[1,1]
    Cₜ¹²(qr) = Cₜ(qr)[1,2]
    Cₜ²¹(qr) = Cₜ(qr)[2,1]
    Cₜ²²(qr) = Cₜ(qr)[2,2]

    𝐃𝐪𝐪𝐀 = Dqq.([[Aₜ¹¹.(qr)]; [Aₜ¹².(qr)]; [Aₜ²¹.(qr)]; [Aₜ²².(qr)]]);
    𝐃𝐫𝐫𝐁 = Drr.([[Bₜ¹¹.(qr)]; [Bₜ¹².(qr)]; [Bₜ²¹.(qr)]; [Bₜ²².(qr)]]);
    𝐃𝐪𝐫𝐂 = Dqr.([[Cₜ¹¹.(qr)]; [Cₜ¹².(qr)]; [Cₜ²¹.(qr)]; [Cₜ²².(qr)]]);
    
    [𝐃𝐪𝐪𝐀[1] 𝐃𝐪𝐪𝐀[2]; 𝐃𝐪𝐪𝐀[3] 𝐃𝐪𝐪𝐀[4]] + [𝐃𝐫𝐫𝐁[1] 𝐃𝐫𝐫𝐁[2]; 𝐃𝐫𝐫𝐁[3] 𝐃𝐫𝐫𝐁[4]] +
        [𝐃𝐪𝐫𝐂[1] 𝐃𝐪𝐫𝐂[2]; 𝐃𝐪𝐫𝐂[3] 𝐃𝐪𝐫𝐂[4]] + [𝐃𝐪𝐫𝐂[1] 𝐃𝐪𝐫𝐂[3]; 𝐃𝐪𝐫𝐂[2] 𝐃𝐪𝐫𝐂[4]]
end

stima = 𝐊(qr)
