using SBP
using LoopVectorization
using SplitApplyCombine
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Plots

"""
Density function 
"""
ρ₁(x) = 1.5
ρ₂(x) = 3.0

"""
The Lamé parameters μ₁, λ₁ on Layer 1
"""
μ₁(x) = 1.8^2*ρ₁(x)
λ₁(x) = 3.118^2*ρ₁(x) - 2μ₁(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 2
"""
μ₂(x) = 3^2*ρ₂(x)
λ₂(x) = 5.196^2*ρ₂(x) - 2μ₂(x)


"""
Material properties coefficients of an anisotropic material
"""
c₁₁¹(x) = 2*μ₁(x)+λ₁(x)
c₂₂¹(x) = 2*μ₁(x)+λ₁(x)
c₃₃¹(x) = μ₁(x)
c₁₂¹(x) = λ₁(x)

c₁₁²(x) = 2*μ₂(x)+λ₂(x)
c₂₂²(x) = 2*μ₂(x)+λ₂(x)
c₃₃²(x) = μ₂(x)
c₁₂²(x) = λ₂(x)


"""
The PML damping
"""
const Lᵥ = 3.6π
const Lₕ = 3.6π
const δ = 0.1*4π  
const σ₀ᵛ = 4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
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

"""
Horizontal PML strip
"""
function σₕ(x)
  if((x[2] ≈ Lₕ) || (x[2] > Lₕ))
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
𝒫₁(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫₂(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];

"""
The material property tensor with the PML is given as follows:
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫₁ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁¹(x) + σₕ(x)*c₁₁¹(x) 0 0 0; 0 -σᵥ(x)*c₃₃¹(x) + σₕ(x)*c₃₃¹(x) 0 0; 0 0 σᵥ(x)*c₃₃¹(x) - σₕ(x)*c₃₃¹(x)  0; 0 0 0 σᵥ(x)*c₂₂¹(x) - σₕ(x)*c₂₂¹(x)];
𝒫₂ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁²(x) + σₕ(x)*c₁₁²(x) 0 0 0; 0 -σᵥ(x)*c₃₃²(x) + σₕ(x)*c₃₃²(x) 0 0; 0 0 σᵥ(x)*c₃₃²(x) - σₕ(x)*c₃₃²(x)  0; 0 0 0 σᵥ(x)*c₂₂²(x) - σₕ(x)*c₂₂²(x)];

"""
Impedance matrices
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

"""
Function to obtain the stiffness matrix corresponding to the 2-layer linear elasticity
"""
function 𝐊2!(𝒫, 𝛀::Tuple{DiscreteDomain, DiscreteDomain},  𝐪𝐫)
  𝒫¹, 𝒫² = 𝒫
  𝛀₁, 𝛀₂ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  @assert 𝛀₁.mn == 𝛀₂.mn "Grid size need to be equal"
  (size(𝐪𝐫) != 𝛀₁.mn) && begin
    @warn "Grid not same size. Using the grid size in DiscreteDomain and overwriting the reference grid.."
    𝐪𝐫 = generate_2d_grid(𝛀.mn)
  end
  # Get the bulk and the traction operator for the 1st layer
  detJ₁(x) = (det∘J)(x, Ω₁)
  Pqr₁ = P2R.(𝒫¹, Ω₁, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₁ = Pᴱ(Pqr₁) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀¹, 𝐓r₀¹, 𝐓qₙ¹, 𝐓rₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [-1,0]).A, Tᴱ(Pqr₁, 𝛀₁, [0,-1]).A, Tᴱ(Pqr₁, 𝛀₁, [1,0]).A, Tᴱ(Pqr₁, 𝛀₁, [0,1]).A 
  
  # Get the bulk and the traction operator for the 2nd layer
  detJ₂(x) = (det∘J)(x, Ω₂)    
  Pqr₂ = P2R.(𝒫², Ω₂, 𝐪𝐫) # Property matrix evaluated at grid points
  𝐏₂ = Pᴱ(Pqr₂) # Elasticity bulk differential operator
  # Elasticity traction operators
  𝐓q₀², 𝐓r₀², 𝐓qₙ², 𝐓rₙ² = Tᴱ(Pqr₂, 𝛀₂, [-1,0]).A, Tᴱ(Pqr₂, 𝛀₂, [0,-1]).A, Tᴱ(Pqr₂, 𝛀₂, [1,0]).A, Tᴱ(Pqr₂, 𝛀₂, [0,1]).A 
  
  # Get the norm matrices (Same for both layers)
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  
  # Determinants of the transformation
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫)
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫) 
  𝐉 = blockdiag(𝐉₁, 𝐉₂)
  𝐉⁻¹ = sparse(𝐉\I(size(𝐉,1)))
  
  # Surface Jacobians of the outer boundaries
  # - Layer 1  
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ = Js(𝛀₁, [0,-1]; X=I(2)), Js(𝛀₁, [-1,0]; X=I(2)), Js(𝛀₁, [0,1]; X=I(2)), Js(𝛀₁, [1,0]; X=I(2))   
  # - Layer 2
  SJr₀², SJq₀², _, SJqₙ² = Js(𝛀₂, [0,-1]; X=I(2)), Js(𝛀₂, [-1,0]; X=I(2)), Js(𝛀₂, [0,1]; X=I(2)), Js(𝛀₂, [1,0]; X=I(2))   

  # Combine the operators    
  𝐏 = blockdiag(𝐏₁.A, 𝐏₂.A)
  𝐓 = blockdiag(-(I(2)⊗𝐇q₀⁻¹)*SJq₀¹*(𝐓q₀¹) + (I(2)⊗𝐇qₙ⁻¹)*SJqₙ¹*(𝐓qₙ¹) + (I(2)⊗𝐇rₙ⁻¹)*SJrₙ¹*(𝐓rₙ¹),
                -(I(2)⊗𝐇q₀⁻¹)*SJq₀²*(𝐓q₀²) + (I(2)⊗𝐇qₙ⁻¹)*SJqₙ²*(𝐓qₙ²) + -(I(2)⊗𝐇r₀⁻¹)*SJr₀²*(𝐓r₀²))
  𝐓rᵢ = blockdiag(𝐓r₀¹, 𝐓rₙ²)            
  
  # Get the Interface SAT for Conforming Interface
  B̂, B̃, 𝐇⁻¹ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=I(2))
  
  h = 1/(m-1)
  ζ₀ = 40/h
  𝐓ᵢ = (I(2)⊗I(2)⊗𝐇⁻¹)*(0.5*B̂*𝐓rᵢ - 0.5*𝐓rᵢ'*B̂ - ζ₀*B̃)
  
  𝐉⁻¹*(𝐏 - 𝐓 - 𝐓ᵢ)
end

"""
Function to obtain the PML stiffness matrix corresponding to the 2-layer linear elasticity
"""
function 𝐊2ₚₘₗ(𝒫, 𝒫ᴾᴹᴸ, Z₁₂, 𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫)
  # Extract domains
  𝛀₁, 𝛀₂ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);

  # Extract the material property functions
  # (Z₁¹, Z₂¹), (Z₁², Z₂²) = Z₁₂
  Z¹₁₂, Z²₁₂ = Z₁₂
  Z₁¹, Z₂¹ = Z¹₁₂
  Z₁², Z₂² = Z²₁₂

  𝒫₁, 𝒫₂ = 𝒫
  𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ = 𝒫ᴾᴹᴸ

  # Get the bulk terms for layer 1
  Pqr₁ = P2R.(𝒫₁,Ω₁,𝐪𝐫);
  Pᴾᴹᴸqr₁ = P2Rᴾᴹᴸ.(𝒫₁ᴾᴹᴸ, Ω₁, 𝐪𝐫);  
  𝐏₁ = Pᴱ(Pqr₁).A;
  𝐏₁ᴾᴹᴸ₁, 𝐏₁ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₁).A;

  # Get the bulk terms for layer 2
  Pqr₂ = P2R.(𝒫₂,Ω₂,𝐪𝐫);
  Pᴾᴹᴸqr₂ = P2Rᴾᴹᴸ.(𝒫₂ᴾᴹᴸ, Ω₂, 𝐪𝐫);  
  𝐏₂ = Pᴱ(Pqr₂).A;
  𝐏₂ᴾᴹᴸ₁, 𝐏₂ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₂).A;

  # Get the 2d SBP operators on the reference grid
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  Dq, Dr = sbp_2d.D1
  Dqr = [I(2)⊗Dq, I(2)⊗Dr]

  # Obtain some quantities on the grid points on Layer 1
  # Bulk Jacobian
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫)
  𝐉₁⁻¹ = 𝐉₁\(I(size(𝐉₁,1))) 
  # Impedance matrices
  𝐙₁₂¹ = 𝐙((Z₁¹,Z₂¹), Ω₁, 𝐪𝐫);
  𝛔₁₂¹ = 𝐙((x->σₕ(x)*Z₁¹(x), x->σᵥ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫)
  𝛕₁₂¹ = 𝐙((x->σₕ(x)*σᵥ(x)*Z₁¹(x), x->σₕ(x)*σᵥ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫)
  𝛔ᵥ¹ = I(2) ⊗ spdiagm(σᵥ.(Ω₁.(vec(𝐪𝐫))));  𝛔ₕ¹ = I(2) ⊗ spdiagm(σₕ.(Ω₁.(vec(𝐪𝐫))));
  𝛒₁ = I(2) ⊗ spdiagm(ρ₁.(Ω₁.(vec(𝐪𝐫))))
  # Get the transformed gradient
  Jqr₁ = J⁻¹.(𝐪𝐫, Ω₁);
  J_vec₁ = get_property_matrix_on_grid(Jqr₁, 2);
  J_vec_diag₁ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₁];
  Dx₁, Dy₁ = J_vec_diag₁*Dqr; 

  # Obtain some quantities on the grid points on Layer 1
  # Bulk Jacobian
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫)
  𝐉₂⁻¹ = 𝐉₂\(I(size(𝐉₂,1))) 
  # Impedance matrices
  𝐙₁₂² = 𝐙((Z₁²,Z₂²), Ω₂, 𝐪𝐫);
  𝛔₁₂² = 𝐙((x->σₕ(x)*Z₁²(x), x->σᵥ(x)*Z₂²(x)), Ω₂, 𝐪𝐫)
  𝛕₁₂² = 𝐙((x->σᵥ(x)*σₕ(x)*Z₁²(x), x->σᵥ(x)*σₕ(x)*Z₂²(x)), Ω₂, 𝐪𝐫)  
  𝛔ᵥ² = I(2) ⊗ spdiagm(σᵥ.(Ω₂.(vec(𝐪𝐫))));  𝛔ₕ² = I(2) ⊗ spdiagm(σₕ.(Ω₂.(vec(𝐪𝐫))));
  𝛒₂ = I(2) ⊗ spdiagm(ρ₂.(Ω₂.(vec(𝐪𝐫))))
  # Get the transformed gradient
  Jqr₂ = J⁻¹.(𝐪𝐫, Ω₂);
  J_vec₂ = get_property_matrix_on_grid(Jqr₂, 2);
  J_vec_diag₂ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₂];
  Dx₂, Dy₂ = J_vec_diag₂*Dqr;

  # Surface Jacobian Matrices on Layer 1
  SJr₀¹, SJq₀¹, SJrₙ¹, SJqₙ¹ =  𝐉₁⁻¹*Js(𝛀₁, [0,-1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [-1,0];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [0,1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 2
  SJr₀², SJq₀², SJrₙ², SJqₙ² =  𝐉₂⁻¹*Js(𝛀₂, [0,-1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [-1,0];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [0,1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [1,0];  X=I(2))

  # We build the governing equations on both layer simultaneously
  # Equation 1: ∂u/∂t = p
  EQ1₁ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m)⊗I(m))
  EQ1₂ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m)⊗I(m))

  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [E1(2,i,(6,6)) for i=1:6];
  eq2s₁ = [(𝐉₁⁻¹*𝐏₁)+α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)-𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, -𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹), 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₁, 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₂, -α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)+𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, 𝛒₁*𝛔ᵥ¹*𝛔ₕ¹];
  eq2s₂ = [(𝐉₂⁻¹*𝐏₂)+α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)-𝛒₂*𝛔ᵥ²*𝛔ₕ², -𝛒₂*(𝛔ᵥ²+𝛔ₕ²), 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₁, 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₂, -α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)+𝛒₂*𝛔ᵥ²*𝛔ₕ², 𝛒₂*𝛔ᵥ²*𝛔ₕ²];
  EQ2₁ = sum(es .⊗ eq2s₁);  
  EQ2₂ = sum(es .⊗ eq2s₂);

  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [E1(3,i,(6,6)) for i=[1,3]];
  eq3s₁ = [Dx₁, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ᵥ¹)];
  eq3s₂ = [Dx₂, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ᵥ²)];
  EQ3₁ = sum(es .⊗ eq3s₁);
  EQ3₂ = sum(es .⊗ eq3s₂);

  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [E1(4,i,(6,6)) for i=[1,4]]
  eq4s₁ = [Dy₁, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ₕ¹)]
  eq4s₂ = [Dy₂, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ₕ²)]
  EQ4₁ = sum(es .⊗ eq4s₁)
  EQ4₂ = sum(es .⊗ eq4s₂)

  # Equation 5 ∂q/∂t = α(u-q)
  es = [E1(5,i,(6,6)) for i=[1,5]]
  eq5s₁ = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  eq5s₂ = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ5₁ = sum(es .⊗ eq5s₁)#=  =#
  EQ5₂ = sum(es .⊗ eq5s₂)

  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [E1(6,i,(6,6)) for i=[1,5,6]]
  eq6s₁ = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  eq6s₂ = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ6₁ = sum(es .⊗ eq6s₁)
  EQ6₂ = sum(es .⊗ eq6s₂)

  # PML characteristic boundary conditions
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ¹ = Pqr₁, Pᴾᴹᴸqr₁, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, 𝐉₁;
  χq₀¹, χr₀¹, χqₙ¹, χrₙ¹ = χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [-1,0]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [0,-1]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [1,0]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₁ = (fill(SJq₀¹,6).*fill((I(2)⊗𝐇q₀⁻¹),6));
  SJ_𝐇qₙ⁻¹₁ = (fill(SJqₙ¹,6).*fill((I(2)⊗𝐇qₙ⁻¹),6));
  SJ_𝐇r₀⁻¹₁ = (fill(SJr₀¹,6).*fill((I(2)⊗𝐇r₀⁻¹),6));
  SJ_𝐇rₙ⁻¹₁ = (fill(SJrₙ¹,6).*fill((I(2)⊗𝐇rₙ⁻¹),6));
  SAT₁ = sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹)) + sum(es.⊗(SJ_𝐇rₙ⁻¹₁.*χrₙ¹));
  
  PQRᵪ² = Pqr₂, Pᴾᴹᴸqr₂, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², 𝐉₂;
  χq₀², χr₀², χqₙ², χrₙ² = χᴾᴹᴸ(PQRᵪ², 𝛀₂, [-1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [0,-1]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₂ = (fill(SJq₀²,6).*fill((I(2)⊗𝐇q₀⁻¹),6));
  SJ_𝐇qₙ⁻¹₂ = (fill(SJqₙ²,6).*fill((I(2)⊗𝐇qₙ⁻¹),6));
  SJ_𝐇r₀⁻¹₂ = (fill(SJr₀²,6).*fill((I(2)⊗𝐇r₀⁻¹),6));
  SJ_𝐇rₙ⁻¹₂ = (fill(SJrₙ²,6).*fill((I(2)⊗𝐇rₙ⁻¹),6));
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²)) + sum(es.⊗(SJ_𝐇r₀⁻¹₂.*χr₀²));

  # The interface part
  Eᵢ¹ = E1(2,1,(6,6)) ⊗ I(2)
  Eᵢ² = E1(1,1,(6,6)) ⊗ I(2)
  # Get the jump matrices
  B̂,  B̃, _ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂ᵀ, _, 𝐇⁻¹ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)
  # Traction on interface From Layer 1
  Tr₀¹ = Tᴱ(Pqr₁, 𝛀₁, [0;-1]).A
  Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁ = Tᴾᴹᴸ(Pᴾᴹᴸqr₁, 𝛀₁, [0;-1]).A  
  # Traction on interface From Layer 2
  Trₙ² = Tᴱ(Pqr₂, 𝛀₂, [0;1]).A
  Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂ = Tᴾᴹᴸ(Pᴾᴹᴸqr₂, 𝛀₂, [0;1]).A
  # Assemble the traction on the two layers
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓r₀¹ = sum(es .⊗ [Tr₀¹, Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁])
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓rₙ² = sum(es .⊗ [Trₙ², Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂])
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀ₀¹ = sum(es .⊗ [(Tr₀¹)', (Tr₀ᴾᴹᴸ₁₁)', (Tr₀ᴾᴹᴸ₂₁)'])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀₙ² = sum(es .⊗ [(Trₙ²)', (Trₙᴾᴹᴸ₁₂)', (Trₙᴾᴹᴸ₂₂)'])
  𝐓rᵢ = blockdiag(𝐓r₀¹, 𝐓rₙ²)      
  𝐓rᵢᵀ = blockdiag(𝐓rᵀ₀¹, 𝐓rᵀₙ²)   
  h = 4π/(m-1)
  ζ₀ = 300/h  
  # Assemble the interface SAT
  𝐉 = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₁⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₂⁻¹)
  SATᵢ = (I(2)⊗I(12)⊗𝐇⁻¹)*𝐉*(0.5*B̂*𝐓rᵢ - 0.5*𝐓rᵢᵀ*B̂ᵀ - ζ₀*B̃)

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂));  
  SATₙ = blockdiag(SAT₁, SAT₂)
  bulk - SATᵢ - SATₙ;
end

"""
Inverse of the mass matrix for the PML case
"""
function 𝐌2⁻¹ₚₘₗ(𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫, ρ)
  ρ₁, ρ₂ = ρ
  𝛀₁, 𝛀₂ = 𝛀
  m, n = size(𝐪𝐫)
  Id = sparse(I(2)⊗I(m)⊗I(n))
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁.(Ω₁.(𝐪𝐫))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂.(Ω₂.(𝐪𝐫))))
  blockdiag(blockdiag(Id, ρᵥ¹, Id, Id, Id, Id), blockdiag(Id, ρᵥ², Id, Id, Id, Id))
end 

"""
Inverse of the mass matrix without the PML
"""
function 𝐌2⁻¹(𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫, ρ)
  ρ₁, ρ₂ = ρ
  𝛀₁, 𝛀₂ = 𝛀
  m, n = size(𝐪𝐫)
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁.(Ω₁.(𝐪𝐫))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂.(Ω₂.(𝐪𝐫))))
  blockdiag(ρᵥ¹, ρᵥ²)
end 

"""
A non-allocating implementation of the RK4 scheme
"""
function RK4_1!(M, sol)  
  X₀, k₁, k₂, k₃, k₄ = sol
  # k1 step  
  mul!(k₁, M, X₀);
  # k2 step
  mul!(k₂, M, k₁, 0.5*Δt, 0.0); mul!(k₂, M, X₀, 1, 1);
  # k3 step
  mul!(k₃, M, k₂, 0.5*Δt, 0.0); mul!(k₃, M, X₀, 1, 1);
  # k4 step
  mul!(k₄, M, k₃, Δt, 0.0); mul!(k₄, M, X₀, 1, 1);
  # Final step
  @turbo for i=1:lastindex(X₀)
    X₀[i] = X₀[i] + (Δt/6)*(k₁[i] + k₂[i] + k₃[i] + k₄[i])
  end
  X₀
end

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Function to split the solution into the corresponding variables
"""
function split_solution(X, N, M)  
  splitdimsview(reshape(X, (N^2, M)))
end


##########################
# Define the two domains #
##########################
# Define the domain for PML computation
cᵢ_pml(q) = @SVector [4π*q, 0.0π*exp(-0.5*(4π*q-2π)^2)]
c₀¹_pml(r) = @SVector [0.0, 4π*r]
c₁¹_pml(q) = cᵢ_pml(q)
c₂¹_pml(r) = @SVector [4π, 4π*r]
c₃¹_pml(q) = @SVector [4π*q, 4π]
domain₁_pml = domain_2d(c₀¹_pml, c₁¹_pml, c₂¹_pml, c₃¹_pml)
c₀²_pml(r) = @SVector [0.0, 4π*r-4π]
c₁²_pml(q) = @SVector [4π*q, -4π]
c₂²_pml(r) = @SVector [4π, 4π*r-4π]
c₃²_pml(q) = cᵢ_pml(q)
domain₂_pml = domain_2d(c₀²_pml, c₁²_pml, c₂²_pml, c₃²_pml)
# Define the domain for full elasticity computation
cᵢ(q) = @SVector [8π*q, 0.0π*exp(-0.5*(8π*q-2π)^2)]
c₀¹(r) = @SVector [0.0, 8π*r]
c₁¹(q) = cᵢ(q)
c₂¹(r) = @SVector [8π, 8π*r]
c₃¹(q) = @SVector [8π*q, 8π]
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
c₀²(r) = @SVector [0.0, 8π*r-8π]
c₁²(q) = @SVector [8π*q, -8π]
c₂²(r) = @SVector [8π, 8π*r-8π]
c₃²(q) = cᵢ(q)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)


const Δt = 1e-3
tf = 3.0
ntime = ceil(Int, tf/Δt)

#######################################
# Time stepping for the Full elasticity
#######################################
N₁ = 161;
𝛀₁ = DiscreteDomain(domain₁, (N₁,N₁));
𝛀₂ = DiscreteDomain(domain₂, (N₁,N₁));
U₀(x) = @SVector [exp(-4*((x[1]-3.6π)^2 + (x[2]-π)^2)), -exp(-4*((x[1]-3.6π)^2 + (x[2]-π)^2))]
V₀(x) = @SVector [0.0,0.0]
Ω₁(qr) = S(qr, 𝛀₁.domain);
Ω₂(qr) = S(qr, 𝛀₂.domain);
𝐪𝐫 = generate_2d_grid((N₁,N₁))
xy₁ = Ω₁.(𝐪𝐫); xy₂ = Ω₂.(𝐪𝐫);
stima2 = 𝐊2!((𝒫₁, 𝒫₂), (𝛀₁, 𝛀₂), 𝐪𝐫);
massma2 =  𝐌2⁻¹((𝛀₁, 𝛀₂), 𝐪𝐫, (ρ₁, ρ₂));
let
  t = 0.0
  X₀ = vcat(eltocols(vec(U₀.(xy₁))), eltocols(vec(U₀.(xy₂))));
  Y₀ = vcat(eltocols(vec(V₀.(xy₁))), eltocols(vec(V₀.(xy₂))));
  global Z₀ = vcat(X₀, Y₀)  
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma2*stima2
  K = [zero(M) I(size(M,1)); M zero(M)]
  # @gif for i=1:ntime
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(K, sol)    
    t += Δt        
    (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))
  end
  # end  every 100 
end  
N = 𝛀₁.mn[1];
Uref = Z₀[1:4N^2];
u1ref₁,u2ref₁ = Tuple(split_solution(Uref, N, 4)[1:2]);
u1ref₂,u2ref₂ = Tuple(split_solution(Uref, N, 4)[3:4]);
plt4 = scatter(Tuple.(vec(xy₁)), zcolor=vec(u1ref₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt4, Tuple.(vec(xy₂)), zcolor=vec(u1ref₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
scatter!(plt4, Tuple.([cᵢ(q) for q in LinRange(0,1,N)]), ms=2, msw=0.1, label="", mc=:red)
scatter!(plt4, Tuple.([[Lᵥ,q] for q in LinRange(-4π,4π,N)]), label="x ≥ "*string(round(Lᵥ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
scatter!(plt4, Tuple.([[q,Lₕ] for q in LinRange(0,4π,N)]), label="y ≥ "*string(round(Lₕ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
scatter!(plt4, Tuple.([[q,-Lₕ] for q in LinRange(0,4π,N)]), label="y ≤ "*string(round(-Lₕ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
scatter!(plt4, Tuple.([[q,-Lₕ-δ] for q in LinRange(0,4π,N)]), label="", markercolor=:black, markersize=2, msw=0.1);    
scatter!(plt4, Tuple.([[q,Lₕ+δ] for q in LinRange(0,4π,N)]), label="", markercolor=:black, markersize=2, msw=0.1);    
scatter!(plt4, Tuple.([[Lᵥ+δ,q] for q in LinRange(-4π,4π,N)]), label="", markercolor=:black, markersize=2, msw=0.1);    
title!(plt4, "Time t="*string(tf))
u1ref₁ᶠ = copy(u1ref₁)
u2ref₁ᶠ = copy(u2ref₁)


#######################################
# Time stepping for the PML elasticity
#######################################
𝐔(x) = @SVector [exp(-4*((x[1]-3.6π)^2 + (x[2]-π)^2)), -exp(-4*((x[1]-3.6π)^2 + (x[2]-π)^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

Ns = [21,41,81]
max_err = zeros(Float64, length(Ns))
for (Ni,i) = zip(Ns, 1:length(Ns))
  N₂ = Ni;
  𝛀₁ᴾᴹᴸ = DiscreteDomain(domain₁_pml, (N₂,N₂));
  𝛀₂ᴾᴹᴸ = DiscreteDomain(domain₂_pml, (N₂,N₂));
  𝐪𝐫ᴾᴹᴸ = generate_2d_grid((N₂,N₂))
  Ω₁ᴾᴹᴸ(qr) = S(qr, 𝛀₁ᴾᴹᴸ.domain);
  Ω₂ᴾᴹᴸ(qr) = S(qr, 𝛀₂ᴾᴹᴸ.domain);
  xy₁ᴾᴹᴸ = Ω₁ᴾᴹᴸ.(𝐪𝐫ᴾᴹᴸ); xy₂ᴾᴹᴸ = Ω₂ᴾᴹᴸ.(𝐪𝐫ᴾᴹᴸ);
  stima2_pml =  𝐊2ₚₘₗ((𝒫₁, 𝒫₂), (𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ), ((Z₁¹, Z₂¹), (Z₁², Z₂²)), (𝛀₁ᴾᴹᴸ, 𝛀₂ᴾᴹᴸ), 𝐪𝐫ᴾᴹᴸ);
  massma2_pml =  𝐌2⁻¹ₚₘₗ((𝛀₁, 𝛀₂), 𝐪𝐫ᴾᴹᴸ, (ρ₁, ρ₂));
  # Begin time loop
  let
    t = 0.0
    X₀¹ = vcat(eltocols(vec(𝐔.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐏.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐕.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐖.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐐.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐑.(xy₁ᴾᴹᴸ))));
    X₀² = vcat(eltocols(vec(𝐔.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐏.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐕.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐖.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐐.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐑.(xy₂ᴾᴹᴸ))));
    X₀ = vcat(X₀¹, X₀²)
    k₁ = zeros(Float64, length(X₀))
    k₂ = zeros(Float64, length(X₀))
    k₃ = zeros(Float64, length(X₀))
    k₄ = zeros(Float64, length(X₀)) 
    M = massma2_pml*stima2_pml
    # @gif for i=1:ntime
    for i=1:ntime
      sol = X₀, k₁, k₂, k₃, k₄
      X₀ = RK4_1!(M, sol)    
      t += Δt    
      (i%100==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))
    end
    # end  every 10  
    global Xref = X₀
  end  
  local N = 𝛀₁ᴾᴹᴸ.mn[1];
  local u1ref₁, u2ref₁ = Tuple(split_solution(Xref[1:12N^2], N, 12)[1:2]);
  local u1ref₂, u2ref₂ = Tuple(split_solution(Xref[12N^2+1:24N^2], N, 12)[1:2]);
  global plt3 = scatter(Tuple.(vec(xy₁ᴾᴹᴸ)), zcolor=vec(u1ref₁), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
  scatter!(plt3, Tuple.(vec(xy₂ᴾᴹᴸ)), zcolor=vec(u1ref₂), colormap=:turbo, ylabel="y(=r)", markersize=4, msw=0.01, label="");
  scatter!(plt3, Tuple.([[Lᵥ,q] for q in LinRange(Ω₂ᴾᴹᴸ([0.0,0.0])[2],Ω₁ᴾᴹᴸ([1.0,1.0])[2],N)]), label="x ≥ "*string(round(Lᵥ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
  scatter!(plt3, Tuple.([[q,Lₕ] for q in LinRange(Ω₁ᴾᴹᴸ([0.0,1.0])[1],Ω₁ᴾᴹᴸ([1.0,1.0])[1],N)]), label="y ≥ "*string(round(Lₕ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
  scatter!(plt3, Tuple.([[q,-Lₕ] for q in LinRange(Ω₂ᴾᴹᴸ([0.0,0.0])[1],Ω₂ᴾᴹᴸ([1.0,0.0])[1],N)]), label="y ≤ "*string(round(-Lₕ,digits=3))*" (PML)", markercolor=:black, markersize=2, msw=0.1);    
  scatter!(plt3, Tuple.([cᵢ_pml(q) for q in LinRange(0,1,N)]), ms=2, msw=0.1, label="", mc=:red)
  title!(plt3, "Time t="*string(tf))
  u1ref₁ᴾ = copy(u1ref₁)
  u2ref₁ᴾ = copy(u2ref₁)

  aspect_ratio = Int64((N₁-1)/((N₂-1)*2));
  comput_domain = Int64(((N₂)^2 - length(findnz(sparse(σᵥ.(xy₁ᴾᴹᴸ) .< 1e-8))[3]))/N₂)
  indices = 1:aspect_ratio:Int64((N₁-1)/2)+1;
  U_PML₁ = reshape(u1ref₁ᴾ, (N₂,N₂))[1:end-comput_domain, 1:end-comput_domain];
  U_FULL₁ = reshape(u1ref₁ᶠ, (N₁,N₁))[indices,indices][1:end-comput_domain, 1:end-comput_domain];
  DU_FULL_PML₁ = abs.(U_PML₁-U_FULL₁);
  global plt5 = scatter(Tuple.(xy₁ᴾᴹᴸ[1:end-comput_domain, 1:end-comput_domain] |> vec), zcolor=vec(DU_FULL_PML₁), msw=0.0, ms=4)

  max_err[i] = maximum(DU_FULL_PML₁)
end 


plot(plot(plt3, xlims=(0,8π), ylims=(-8π,8π)), plot(plt4, xlims=(0,8π), ylims=(-8π,8π)),size=(1600,800))