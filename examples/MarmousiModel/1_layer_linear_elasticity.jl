# include("2d_elasticity_problem.jl");
using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Plots

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');


## Define the physical domain
c₀(r) = @SVector [0.0 + 0.0*sin(π*r), r] # Left boundary 
c₁(q) = @SVector [q, 0.0 + 0.0*sin(2π*q)] # Bottom boundary
c₂(r) = @SVector [1.0 + 0.0*sin(π*r), r] # Right boundary
c₃(q) = @SVector [q, 1.0 + 0.0*sin(2π*q)]
domain = domain_2d(c₀, c₁, c₂, c₃)

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
function 𝐊!(𝒫, 𝛀::DiscreteDomain, 𝐪𝐫)
  Ω(qr) = S(qr, 𝛀.domain)
  detJ(x) = (det∘J)(x,Ω)    

  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(n)
  sbp_r = SBP_1_2_CONSTANT_0_1(m)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  
  # Get the material property matrix evaluated at grid points    
  Pqr = P2R.(𝒫,Ω,𝐪𝐫) 

  # Elasticity bulk differential operator  
  𝐏 = Pᴱ(Pqr).A 

  # Elasticity Traction Operators
  𝐓q₀, 𝐓r₀, 𝐓qₙ, 𝐓rₙ = Tᴱ(Pqr, 𝛀, [-1,0]).A, Tᴱ(Pqr, 𝛀, [0,-1]).A, Tᴱ(Pqr, 𝛀, [1,0]).A, Tᴱ(Pqr, 𝛀, [0,1]).A   

  # The surface Jacobians on the boundary
  SJr₀, SJq₀, SJrₙ, SJqₙ = Js(𝛀, [0,-1];  X=I(2)), Js(𝛀, [-1,0];  X=I(2)), Js(𝛀, [0,1];  X=I(2)), Js(𝛀, [1,0];  X=I(2))   
  
  # The norm-inverse on the boundary
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
  # Bulk Jacobian
  𝐉 = Jb(𝛀, 𝐪𝐫)

  SAT = (-(I(2) ⊗ 𝐇q₀⁻¹)*SJq₀*(𝐓q₀) + (I(2) ⊗ 𝐇qₙ⁻¹)*SJqₙ*(𝐓qₙ) -(I(2) ⊗ 𝐇r₀⁻¹)*SJr₀*(𝐓r₀) + (I(2) ⊗ 𝐇rₙ⁻¹)*SJrₙ*(𝐓rₙ))

  # The SBP-SAT Formulation    
  𝐉\(𝐏 - SAT)
end

m = 31; n = 21;
𝐪𝐫 = generate_2d_grid((m,n));
𝛀 = DiscreteDomain(domain, (m,n));
Ω(qr) = S(qr, 𝛀.domain);
stima = 𝐊!(𝒫, 𝛀, 𝐪𝐫);
