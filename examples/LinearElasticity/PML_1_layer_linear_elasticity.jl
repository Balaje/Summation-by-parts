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
c₁(q) = @SVector [1.1*q, 0.0 + 0.11*sin(π*q)]
c₂(r) = @SVector [1.1, 1.1*r]
c₃(q) = @SVector [1.1*q, 1.1 - 0.11*sin(π*q)]
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
const σ₀ᵛ = 0.4*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0.0*(√(4*1))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
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
function 𝐊ᴾᴹᴸ(𝒫, 𝒫ᴾᴹᴸ, Z₁₂, 𝛀::DiscreteDomain, 𝐪𝐫)
  Ω(qr) = S(qr, 𝛀.domain);
  Z₁, Z₂ = Z₁₂

  Pqr = P2R.(𝒫,Ω,𝐪𝐫);
  Pᴾᴹᴸqr = P2Rᴾᴹᴸ.(𝒫ᴾᴹᴸ, Ω, 𝐪𝐫);
  𝐏 = Pᴱ(Pqr).A;
  𝐏ᴾᴹᴸ₁, 𝐏ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr).A;

  # Obtain some quantities on the grid points
  𝐙₁ = 𝐙(Z₁, Ω, 𝐪𝐫);  𝐙₂ = 𝐙(Z₂, Ω, 𝐪𝐫);
  𝛔ᵥ = I(2) ⊗ spdiagm(σᵥ.(Ω.(vec(𝐪𝐫))));  𝛔ₕ = I(2) ⊗ spdiagm(σₕ.(Ω.(vec(𝐪𝐫))));
  𝛒 = I(2) ⊗ spdiagm(ρ.(Ω.(vec(𝐪𝐫))))
  # Get the transformed gradient
  Jqr = J⁻¹.(𝐪𝐫, Ω);
  J_vec = get_property_matrix_on_grid(Jqr, 2);
  J_vec_diag = [I(2)⊗spdiagm(vec(p)) for p in J_vec];
  # Get the 2d SBP operators on the reference grid
  m, n = size(𝐪𝐫)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = sbp_2d.norm
  Dq, Dr = sbp_2d.D1
  Dqr = [I(2)⊗Dq, I(2)⊗Dr]
  Dx, Dy = J_vec_diag*Dqr;
  # Bulk Jacobian
  𝐉 = Jb(𝛀, 𝐪𝐫)
  𝐉⁻¹ = 𝐉\(I(size(𝐉,1))) 

  # Surface Jacobian Matrices
  SJr₀, SJq₀, SJrₙ, SJqₙ =  𝐉⁻¹*Js(𝛀, [0,-1];  X=I(2)), 𝐉⁻¹*Js(𝛀, [-1,0];  X=I(2)), 𝐉⁻¹*Js(𝛀, [0,1];  X=I(2)), 𝐉⁻¹*Js(𝛀, [1,0];  X=I(2))

  # Equation 1: ∂u/∂t = p
  EQ1 = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m)⊗I(m))

  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [E1(2,i,(6,6)) for i=1:6];
  eq2s = [(𝐉⁻¹*𝐏)+α*𝛒*(𝛔ᵥ+𝛔ₕ)-𝛒*𝛔ᵥ*𝛔ₕ, -𝛒*(𝛔ᵥ+𝛔ₕ), 𝐉⁻¹*𝐏ᴾᴹᴸ₁, 𝐉⁻¹*𝐏ᴾᴹᴸ₂, -α*𝛒*(𝛔ᵥ+𝛔ₕ)+𝛒*𝛔ᵥ*𝛔ₕ, 𝛒*𝛔ᵥ*𝛔ₕ];
  EQ2 = sum(es .⊗ eq2s);

  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [E1(3,i,(6,6)) for i=[1,3]];
  eq3s = [Dx, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ᵥ)];
  EQ3 = sum(es .⊗ eq3s);

  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [E1(4,i,(6,6)) for i=[1,4]]
  eq4s = [Dy, -(α*(I(2)⊗I(m)⊗I(n)) + 𝛔ₕ)]
  EQ4 = sum(es .⊗ eq4s)

  # Equation 5 ∂q/∂t = α(u-q)
  es = [E1(5,i,(6,6)) for i=[1,5]]
  eq5s = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ5 = sum(es .⊗ eq5s)

  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [E1(6,i,(6,6)) for i=[1,5,6]]
  eq6s = [α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n)), -α*(I(2)⊗I(m)⊗I(n))]
  EQ6 = sum(es .⊗ eq6s)

  # PML characteristic boundary conditions
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ = Pqr, Pᴾᴹᴸqr, 𝐙₁, 𝐙₂, 𝛔ᵥ, 𝛔ₕ;
  χq₀, χr₀, χqₙ, χrₙ = χᴾᴹᴸ(PQRᵪ, 𝛀, [-1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,-1]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [1,0]).A, χᴾᴹᴸ(PQRᵪ, 𝛀, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹ = (fill(SJq₀,6).*fill((I(2)⊗𝐇q₀⁻¹),6));
  SJ_𝐇qₙ⁻¹ = (fill(SJqₙ,6).*fill((I(2)⊗𝐇qₙ⁻¹),6));
  SJ_𝐇r₀⁻¹ = (fill(SJr₀,6).*fill((I(2)⊗𝐇r₀⁻¹),6));
  SJ_𝐇rₙ⁻¹ = (fill(SJrₙ,6).*fill((I(2)⊗𝐇rₙ⁻¹),6));
  SAT = sum(es.⊗(SJ_𝐇q₀⁻¹.*χq₀)) + sum(es.⊗(SJ_𝐇qₙ⁻¹.*χqₙ)) + sum(es.⊗(SJ_𝐇r₀⁻¹.*χr₀)) + sum(es.⊗(SJ_𝐇rₙ⁻¹.*χrₙ));

  # The SBP-SAT Formulation
  bulk = (EQ1 + EQ2 + EQ3 + EQ4 + EQ5 + EQ6);
  bulk - SAT;
end

stima = 𝐊ᴾᴹᴸ(𝒫, 𝒫ᴾᴹᴸ, (Z₁, Z₂), 𝛀, 𝐪𝐫);