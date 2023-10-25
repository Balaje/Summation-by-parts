include("2d_elasticity_problem.jl")

using SplitApplyCombine

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
# Define the parametrization for interface
f(q) = 1 + 0.0*sin(2π*q)
cᵢ¹(q) = [q, f(q)];
cᵢ²(r) = [f(r), r];
# Define the rest of the boundary
c₀¹(r) = [0.0 , 1+r]; # Left boundary
c₁¹(q) = cᵢ(q) # Bottom boundary. (Interface 1)
c₂¹(r) = [1.0, 1+r]; # Right boundary
c₃¹(q) = [q, 2.0 + 0.0*sin(2π*q)]; # Top boundary
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
# Layer 2 (q,r) ∈ [0,1] × [0,1]
c₀²(r) = [0.0, r]; # Left boundary
c₁²(q) = [q, 0.0]; # Bottom boundary. 
c₂²(r) = cᵢ²(r); # Right boundary (Interface 2)
c₃²(q) = c₁¹(q); # Top boundary. (Interface 1)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
Ω₂(qr) = S(qr, domain₂)
c₀³(r) = cᵢ²(r) # Left boundary (Interface 2)
c₁³(q) = [1.0 + q, 0.0] # Bottom boundary
c₂³(r) = [2.0, r] # Right boundary
c₃³(q) = [1.0 + q, 1.0] # Top boundary
domain₃ = domain_2d(c₀³, c₁³, c₂³, c₃³)

## Define the material properties on the physical grid
"""
The Lamé parameters μ, λ
"""
λ¹(x) = 2.0
μ¹(x) = 1.0
λ²(x) = 2.0
μ²(x) = 1.0
λ³(x) = 2.0
μ³(x) = 1.0

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

c₁₁³(x) = 2*μ³(x)+λ³(x)
c₂₂³(x) = 2*μ³(x)+λ³(x)
c₃₃³(x) = μ³(x)
c₁₂³(x) = λ³(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫¹(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫²(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];
𝒫³(x) = @SMatrix [c₁₁³(x) 0 0 c₁₂³(x); 0 c₃₃³(x) c₃₃³(x) 0; 0 c₃₃³(x) c₃₃³(x) 0; c₁₂³(x) 0 0 c₂₂³(x)];

"""
Cauchy Stress tensor using the displacement field.
"""
σ¹(∇u,x) = 𝒫¹(x)*∇u
σ²(∇u,x) = 𝒫²(x)*∇u
σ³(∇u,x) = 𝒫³(x)*∇u

"""
Density function 
"""
ρ¹(x) = 1.0
ρ²(x) = 0.5
ρ³(x) = 0.25

"""
Stiffness matrix function
"""
function 𝐊3!(𝒫, 𝛀::Tuple{DiscreteDomain, DiscreteDomain, DiscreteDomain},  𝐪𝐫)
  𝒫¹, 𝒫², 𝒫³ = 𝒫
  𝛀₁, 𝛀₂, 𝛀₃ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  Ω₃(qr) = S(qr, 𝛀₃.domain)
  @assert 𝛀₁.mn == 𝛀₂.mn == 𝛀₃.mn "Grid size need to be equal"
  (size(𝐪𝐫) != 𝛀₁.mn) && begin
    @warn "Grid not same size. Using the grid size in DiscreteDomain and overwriting the reference grid.."
    𝐪𝐫 = generate_2d_grid(𝛀.mn)
  end
  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Pqr₁) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹, 𝐓r₀¹, 𝐓qₙ¹, 𝐓rₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [-1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,-1]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [1,0]; X=I(2)).A, Tᴱ(Pqr₁, 𝛀₁, [0,1]; X=I(2)).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Pqr₂) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀², 𝐓r₀², 𝐓qₙ², 𝐓rₙ² = Tᴱ(Pqr₂, 𝛀₂, [-1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,-1]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [1,0]; X=I(2)).A, Tᴱ(Pqr₂, 𝛀₂, [0,1]; X=I(2)).A 

  # Get the bulk and the traction operator for the 3rd layer
  detJ₃(x) = (det∘J)(x, Ω₃)    
  Pqr₃ = P2R.(𝒫³, Ω₃, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₃ = Pᴱ(Pqr₃) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀³, 𝐓r₀³, 𝐓qₙ³, 𝐓rₙ³ = Tᴱ(Pqr₃, 𝛀₃, [-1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,-1]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [1,0]; X=I(2)).A, Tᴱ(Pqr₃, 𝛀₃, [0,1]; X=I(2)).A 
  
  # Get the norm matrices (Same for all layers)
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
  # Determinants of the transformation
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫)
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫) 
  𝐉₃ = Jb(𝛀₃, 𝐪𝐫) 
  𝐉 = blockdiag(𝐉₁, 𝐉₂, 𝐉₃)   
  
  # Surface Jacobians of the outer boundaries
  # - Layer 1  
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ = Js(𝛀₁, [0,-1]; X=I(2)), Js(𝛀₁, [-1,0]; X=I(2)), Js(𝛀₁, [0,1]; X=I(2)), Js(𝛀₁, [1,0]; X=I(2))   
  # - Layer 2
  SJr₀², SJq₀², _, _ = Js(𝛀₂, [0,-1]; X=I(2)), Js(𝛀₂, [-1,0]; X=I(2)), Js(𝛀₂, [0,1]; X=I(2)), Js(𝛀₂, [1,0]; X=I(2))   
  # - Layer 3
  SJr₀³, _, SJrₙ³, SJqₙ³ = Js(𝛀₃, [0,-1]; X=I(2)), Js(𝛀₃, [-1,0]; X=I(2)), Js(𝛀₃, [0,1]; X=I(2)), Js(𝛀₃, [1,0]; X=I(2))   

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁.A, 𝐏₂.A, 𝐏₃.A)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀⁻¹)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ⁻¹)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ⁻¹)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀⁻¹)*SJq₀²*(𝐓q₀²) + -(I(2)⊗𝐇r₀⁻¹)*SJr₀²*(𝐓r₀²), 
                 (I(2)⊗𝐇qₙ⁻¹)*SJqₙ³*(𝐓qₙ³) + -(I(2)⊗𝐇r₀⁻¹)*SJr₀³*(𝐓r₀³) + (I(2)⊗𝐇rₙ⁻¹)*SJrₙ³*(𝐓rₙ³))
  𝐓rᵢ¹ = blockdiag(𝐓r₀¹, 𝐓rₙ²)            
  𝐓qᵢ² = blockdiag(𝐓qₙ², 𝐓q₀³)            
  
  # Get the Interface SAT for Conforming Interface
  B̂₁, B̃₁, 𝐇⁻¹₁ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=I(2))
  B̂₂, B̃₂, 𝐇⁻¹₂ = SATᵢᴱ(𝛀₂, 𝛀₃, [1; 0], [-1; 0], ConformingInterface(); X=I(2))
  
  h = 1/(m-1)
  ζ₀ = 40/h
  𝐓ᵢ¹ = blockdiag((I(2)⊗𝐇⁻¹₁)*(0.5*B̂₁*𝐓rᵢ¹ - 0.5*𝐓rᵢ¹'*B̂₁ - ζ₀*B̃₁), zero(𝐏₃.A))
  𝐓ᵢ² = blockdiag(zero(𝐏₁.A), (I(2)⊗𝐇⁻¹₂)*(-0.5*B̂₂*𝐓qᵢ² + 0.5*𝐓qᵢ²'*B̂₂ - ζ₀*B̃₂))
    
  𝐉\(𝐏 - 𝐓 - 𝐓ᵢ¹ - 𝐓ᵢ²)
end
  
m = 21;
𝐪𝐫 = generate_2d_grid((m,m))
𝛀₁ = DiscreteDomain(domain₁, (m,m))
𝛀₂ = DiscreteDomain(domain₂, (m,m))
𝛀₃ = DiscreteDomain(domain₃, (m,m))
Ω₁(qr) = S(qr, 𝛀₁.domain)
Ω₂(qr) = S(qr, 𝛀₂.domain)
Ω₃(qr) = S(qr, 𝛀₃.domain)
stima3 = 𝐊3!((𝒫¹, 𝒫², 𝒫³), (𝛀₁, 𝛀₂, 𝛀₃), 𝐪𝐫)