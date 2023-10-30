###################################################################################
# Program to solve the linear elasticity equations with a Perfectly Matched Layer
# 1) The computational domain Ω = [0,4.4π] × [0, 4π]
# -------------- CORRECTION WORK IN PROGRESS.... -----------------
###################################################################################

include("2d_elasticity_problem.jl");

using SplitApplyCombine
using LoopVectorization

# Define the domain
c₀(r) = @SVector [0.0, 1.1*r]
c₁(q) = @SVector [1.1*q, 1.1*(0.0 + 0.0*sin(π*q))]
c₂(r) = @SVector [1.1, 1.1*r]
c₃(q) = @SVector [1.1*q, 1.1]
domain = domain_2d(c₀, c₁, c₂, c₃)

"""
The Lamé parameters μ, λ
"""
λ(x) = 2.0
μ(x) = 1.0

"""
Material properties coefficients of an anisotropic material
"""
c₁₁(x) = 2*μ(x)+λ(x)
c₂₂(x) = 2*μ(x)+λ(x)
c₃₃(x) = μ(x)
c₁₂(x) = λ(x)

"""
The PML damping
"""
const Lᵥ = 1.0
const Lₕ = 1.0
const δ = 0.1*Lᵥ
const σ₀ᵛ = 4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ Lᵥ) || x[1] > Lᵥ)
    return σ₀ᵛ*((x[1] - Lᵥ)/δ)^3  
  else
    return 0.0
  end
end

function σₕ(x)
  if((x[2] ≈ Lₕ) || x[2] > Lₕ)
    return σ₀ʰ*((x[2] - Lₕ)/δ)^3  
  else
    return 0.0
  end  
end

"""
The material property tensor in the physical coordinates
𝒫(x) = [A(x) C(x); 
        C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫(x) = @SMatrix [c₁₁(x) 0 0 c₁₂(x); 0 c₃₃(x) c₃₃(x) 0; 0 c₃₃(x) c₃₃(x) 0; c₁₂(x) 0 0 c₂₂(x)];

"""
The material property tensor with the PML is given as follows:
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁(x) + σₕ(x)*c₁₁(x) 0 0 0; 0 -σᵥ(x)*c₃₃(x) + σₕ(x)*c₃₃(x) 0 0; 0 0 σᵥ(x)*c₃₃(x) - σₕ(x)*c₃₃(x)  0; 0 0 0 σᵥ(x)*c₂₂(x) - σₕ(x)*c₂₂(x)];

"""
Density function 
"""
ρ(x) = 1.0

"""
Material velocity tensors
"""
Z₁(x) = @SMatrix [√(c₁₁(x)/ρ(x))  0;  0 √(c₃₃(x)/ρ(x))]
Z₂(x) = @SMatrix [√(c₃₃(x)/ρ(x))  0;  0 √(c₂₂(x)/ρ(x))]


m = 21;
𝛀 = DiscreteDomain(domain, (m,m));
Ω(qr) = S(qr, 𝛀.domain);
𝐪𝐫 = generate_2d_grid((m,m));


"""
Function to obtain the PML stiffness matrix
"""
Pqr = P2R.(𝒫,Ω,𝐪𝐫);
Pᴾᴹᴸqr = P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ, Ω, 𝐪𝐫);
𝐏 = Pᴱ(Pqr).A;
𝐏ᴾᴹᴸ = Pᴾᴹᴸ(Pᴾᴹᴸqr).A;

# Get the PML characteristic boundary conditions
𝐙₁ = 𝐙(Z₁, Ω, 𝐪𝐫);  𝐙₂ = 𝐙(Z₂, Ω, 𝐪𝐫);
σᵥ𝐙₁ = 𝐙(x -> σᵥ(x)*Z₁(x), Ω, 𝐪𝐫);  σᵥ𝐙₂ = 𝐙(x -> σₕ(x)*Z₂(x), Ω, 𝐪𝐫);
PQRᵪ = Pqr, Pᴾᴹᴸqr, 𝐙₁, 𝐙₂, σᵥ𝐙₁, σᵥ𝐙₂;

χq₀, χr₀, χqₙ, χrₙ = χᴾᴹᴸ(PQRᵪ, 𝛀, [-1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,-1]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,1]).A;

SJr₀, SJq₀, SJrₙ, SJqₙ = Js(𝛀, [0,-1];  X=I(2)), Js(𝛀, [-1,0];  X=I(2)), Js(𝛀, [0,1];  X=I(2)), Js(𝛀, [1,0];  X=I(2))

m, n = size(𝐪𝐫)
sbp_q = SBP_1_2_CONSTANT_0_1(m)
sbp_r = SBP_1_2_CONSTANT_0_1(n)
sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)

𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
# Bulk Jacobian
𝐉 = Jb(𝛀, 𝐪𝐫)

# The SBP-SAT Formulation
