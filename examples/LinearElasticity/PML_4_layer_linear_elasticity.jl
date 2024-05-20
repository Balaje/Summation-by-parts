# include("2d_elasticity_problem.jl");
using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff

# Needs pyplot() for this to work ...
using PyPlot
using Plots
pyplot()
using LaTeXStrings
using ColorSchemes
PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

using SplitApplyCombine

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');
eltocols(v::Vector{MVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Get the x-and-y coordinates from coordinates
"""
getX(C) = C[1]; getY(C) = C[2];

# Define the domain
interface₁(q) = @SVector [-4 + 48*q, -10.0]
interface₂(q) = @SVector [-4 + 48*q, -20.0]
interface₃(q) = @SVector [-4 + 48*q, -30.0]

c₀¹(r) = @SVector [-4.0, 10*(r-1)] # Left
c₁¹(q) = interface₁(q) # Bottom
c₂¹(r) = @SVector [44.0, 10*(r-1)] # Right
c₃¹(q) = @SVector [-4 + 48*q, 0.0] # Top
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)

c₀²(r) = @SVector [-4.0, 10*r-20] # Left
c₁²(q) = interface₂(q) # Bottom
c₂²(r) = @SVector [44.0, 10*r-20] # Right
c₃²(q) = interface₁(q) # Top
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)

c₀³(r) = @SVector [-4.0, 10*r-30] # Left
c₁³(q) = interface₃(q) # Bottom
c₂³(r) = @SVector [44.0, 10*r-30] # Right
c₃³(q) = interface₂(q) # Top
domain₃ = domain_2d(c₀³, c₁³, c₂³, c₃³)

c₀⁴(r) = @SVector [-4.0, -44 + 14*r] # Left
c₁⁴(q) = @SVector [-4 + 48*q, -44.0] # Bottom
c₂⁴(r) = @SVector [44.0, -44 + 14*r] # Right
c₃⁴(q) = interface₃(q) # Top
domain₄ = domain_2d(c₀⁴, c₁⁴, c₂⁴, c₃⁴)


##### ##### ##### ##### ##### ##### 
# EXAMPLE OF AN ANISOTROPIC DOMAIN
##### ##### ##### ##### ##### ##### 
# """
# Material properties coefficients of an anisotropic material
# """
# c₁₁¹(x) = 4.0
# c₂₂¹(x) = 20.0
# c₃₃¹(x) = 2.0
# c₁₂¹(x) = 3.8

# c₁₁²(x) = 4*c₁₁¹(x)
# c₂₂²(x) = 4*c₂₂¹(x)
# c₃₃²(x) = 4*c₃₃¹(x)
# c₁₂²(x) = 4*c₁₂¹(x)

# ρ₁(x) = 1.0
# ρ₂(x) = 0.25

##### ##### ##### ##### ##### ##### 
# EXAMPLE OF AN ISOTROPIC DOMAIN
##### ##### ##### ##### ##### ##### 
"""
Density functions
"""
ρ₁(x) = 1.5
ρ₂(x) = 1.9
ρ₃(x) = 2.1
ρ₄(x) = 3.0

"""
The Lamé parameters μ₁, λ₁ on Layer 1
"""
μ₁(x) = 1.8^2*ρ₁(x)
λ₁(x) = 3.118^2*ρ₁(x) - 2μ₁(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 2
"""
μ₂(x) = 2.3^2*ρ₂(x)
λ₂(x) = 3.984^2*ρ₂(x) - 2μ₂(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 3
"""
μ₃(x) = 2.7^2*ρ₃(x)
λ₃(x) = 4.667^2*ρ₃(x) - 2μ₃(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 4
"""
μ₄(x) = 3^2*ρ₄(x)
λ₄(x) = 5.196^2*ρ₄(x) - 2μ₄(x)


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

c₁₁³(x) = 2*μ₃(x)+λ₃(x)
c₂₂³(x) = 2*μ₃(x)+λ₃(x)
c₃₃³(x) = μ₃(x)
c₁₂³(x) = λ₃(x)

c₁₁⁴(x) = 2*μ₄(x)+λ₄(x)
c₂₂⁴(x) = 2*μ₄(x)+λ₄(x)
c₃₃⁴(x) = μ₄(x)
c₁₂⁴(x) = λ₄(x)

"""
The p- and s- wave speeds
"""
cpx₁ = √(c₁₁¹(1.0)/ρ₁(1.0))
cpy₁ = √(c₂₂¹(1.0)/ρ₁(1.0))
csx₁ = √(c₃₃¹(1.0)/ρ₁(1.0))
csy₁ = √(c₃₃¹(1.0)/ρ₁(1.0))
cp₁ = max(cpx₁, cpy₁)
cs₁ = max(csx₁, csy₁)

cpx₂ = √(c₁₁²(1.0)/ρ₂(1.0))
cpy₂ = √(c₂₂²(1.0)/ρ₂(1.0))
csx₂ = √(c₃₃²(1.0)/ρ₂(1.0))
csy₂ = √(c₃₃²(1.0)/ρ₂(1.0))
cp₂ = max(cpx₂, cpy₂)
cs₂ = max(csx₂, csy₂)

cpx₃ = √(c₁₁³(1.0)/ρ₃(1.0))
cpy₃ = √(c₂₂³(1.0)/ρ₃(1.0))
csx₃ = √(c₃₃³(1.0)/ρ₃(1.0))
csy₃ = √(c₃₃³(1.0)/ρ₃(1.0))
cp₃ = max(cpx₃, cpy₃)
cs₃ = max(csx₃, csy₃)

cpx₄ = √(c₁₁⁴(1.0)/ρ₄(1.0))
cpy₄ = √(c₂₂⁴(1.0)/ρ₄(1.0))
csx₄ = √(c₃₃⁴(1.0)/ρ₄(1.0))
csy₄ = √(c₃₃⁴(1.0)/ρ₄(1.0))
cp₄ = max(cpx₄, cpy₄)
cs₄ = max(csx₄, csy₄)


"""
The PML damping
"""
const L = 40
const δ = 0.1*L
const σ₀ = 4*((max(cp₁, cp₂, cp₃, cp₄)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σ(x)
  if((x[1] ≈ L) || x[1] > L)
    return σ₀*((x[1] - L)/δ)^3  
  elseif((x[1] ≈ 0.0) || x[1] < 0.0)
    return σ₀*((0.0 - x[1])/δ)^3
  else
    return 0.0
  end
end

"""
Horizontal PML strip
"""
function τ(x)
  if((x[2] ≈ -L) || x[2] < -L)
    return σ₀*(((-L) - x[2])/δ)^3
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
𝒫₃(x) = @SMatrix [c₁₁³(x) 0 0 c₁₂³(x); 0 c₃₃³(x) c₃₃³(x) 0; 0 c₃₃³(x) c₃₃³(x) 0; c₁₂³(x) 0 0 c₂₂³(x)];
𝒫₄(x) = @SMatrix [c₁₁⁴(x) 0 0 c₁₂⁴(x); 0 c₃₃⁴(x) c₃₃⁴(x) 0; 0 c₃₃⁴(x) c₃₃⁴(x) 0; c₁₂⁴(x) 0 0 c₂₂⁴(x)];


"""
The material property tensor with the PML is given as follows:
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫₁ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁¹(x) 0 0 0; 0 -σ(x)*c₃₃¹(x) 0 0; 0 0 σ(x)*c₃₃¹(x)  0; 0 0 0 σ(x)*c₂₂¹(x)];
𝒫₂ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁²(x) 0 0 0; 0 -σ(x)*c₃₃²(x) 0 0; 0 0 σ(x)*c₃₃²(x)  0; 0 0 0 σ(x)*c₂₂²(x)];
𝒫₃ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁³(x) 0 0 0; 0 -σ(x)*c₃₃³(x) 0 0; 0 0 σ(x)*c₃₃³(x)  0; 0 0 0 σ(x)*c₂₂³(x)];
𝒫₄ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁⁴(x) 0 0 0; 0 -σ(x)*c₃₃⁴(x) 0 0; 0 0 σ(x)*c₃₃⁴(x)  0; 0 0 0 σ(x)*c₂₂⁴(x)];


"""
Material velocity tensors
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

Z₁³(x) = @SMatrix [√(c₁₁³(x)*ρ₃(x))  0;  0 √(c₃₃³(x)*ρ₃(x))]
Z₂³(x) = @SMatrix [√(c₃₃³(x)*ρ₃(x))  0;  0 √(c₂₂³(x)*ρ₃(x))]

Z₁⁴(x) = @SMatrix [√(c₁₁⁴(x)*ρ₄(x))  0;  0 √(c₃₃⁴(x)*ρ₄(x))]
Z₂⁴(x) = @SMatrix [√(c₃₃⁴(x)*ρ₄(x))  0;  0 √(c₂₂⁴(x)*ρ₄(x))]

"""
Function to obtain the PML stiffness matrix
"""
function 𝐊4ₚₘₗ(𝒫, 𝒫ᴾᴹᴸ, Z₁₂, 𝛀::NTuple{4,DiscreteDomain}, 𝐪𝐫)
  # Extract domains
  𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);
  Ω₃(qr) = S(qr, 𝛀₃.domain);
  Ω₄(qr) = S(qr, 𝛀₄.domain);
  𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃, 𝐪𝐫₄ = 𝐪𝐫

  # Extract the material property functions
  # (Z₁¹, Z₂¹), (Z₁², Z₂²) = Z₁₂
  Z¹₁₂, Z²₁₂, Z³₁₂, Z⁴₁₂ = Z₁₂
  Z₁¹, Z₂¹ = Z¹₁₂
  Z₁², Z₂² = Z²₁₂
  Z₁³, Z₂³ = Z³₁₂
  Z₁⁴, Z₂⁴ = Z⁴₁₂

  𝒫₁, 𝒫₂, 𝒫₃, 𝒫₄ = 𝒫
  𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ, 𝒫₃ᴾᴹᴸ, 𝒫₄ᴾᴹᴸ  = 𝒫ᴾᴹᴸ

  # Get the bulk terms for layer 1
  Pqr₁ = P2R.(𝒫₁,Ω₁,𝐪𝐫₁);
  Pᴾᴹᴸqr₁ = P2Rᴾᴹᴸ.(𝒫₁ᴾᴹᴸ, Ω₁, 𝐪𝐫₁);  
  𝐏₁ = Pᴱ(Pqr₁).A;
  𝐏₁ᴾᴹᴸ₁, 𝐏₁ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₁).A;

  # Get the bulk terms for layer 2
  Pqr₂ = P2R.(𝒫₂,Ω₂,𝐪𝐫₂);
  Pᴾᴹᴸqr₂ = P2Rᴾᴹᴸ.(𝒫₂ᴾᴹᴸ, Ω₂, 𝐪𝐫₂);  
  𝐏₂ = Pᴱ(Pqr₂).A;
  𝐏₂ᴾᴹᴸ₁, 𝐏₂ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₂).A;

  # Get the bulk terms for layer 3
  Pqr₃ = P2R.(𝒫₃,Ω₃,𝐪𝐫₃);
  Pᴾᴹᴸqr₃ = P2Rᴾᴹᴸ.(𝒫₃ᴾᴹᴸ, Ω₃, 𝐪𝐫₃);  
  𝐏₃ = Pᴱ(Pqr₃).A;
  𝐏₃ᴾᴹᴸ₁, 𝐏₃ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₃).A;

  # Get the bulk terms for layer 4
  Pqr₄ = P2R.(𝒫₄,Ω₄,𝐪𝐫₄);
  Pᴾᴹᴸqr₄ = P2Rᴾᴹᴸ.(𝒫₄ᴾᴹᴸ, Ω₄, 𝐪𝐫₄);  
  𝐏₄ = Pᴱ(Pqr₄).A;
  𝐏₄ᴾᴹᴸ₁, 𝐏₄ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₄).A;

  # Get the 2d SBP operators on the reference grid
  n₁, m₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀⁻¹₁, 𝐇qₙ⁻¹₁, _, 𝐇rₙ⁻¹₁ = sbp_2d₁.norm
  Dq₁, Dr₁ = sbp_2d₁.D1
  Dqr₁ = [I(2)⊗Dq₁, I(2)⊗Dr₁]
  n₂, m₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀⁻¹₂, 𝐇qₙ⁻¹₂, _, _ = sbp_2d₂.norm
  Dq₂, Dr₂ = sbp_2d₂.D1
  Dqr₂ = [I(2)⊗Dq₂, I(2)⊗Dr₂]
  n₃, m₃ = size(𝐪𝐫₃)
  sbp_q₃ = SBP_1_2_CONSTANT_0_1(m₃)
  sbp_r₃ = SBP_1_2_CONSTANT_0_1(n₃)
  sbp_2d₃ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₃, sbp_r₃)
  𝐇q₀⁻¹₃, 𝐇qₙ⁻¹₃, _, _ = sbp_2d₃.norm
  Dq₃, Dr₃ = sbp_2d₃.D1
  Dqr₃ = [I(2)⊗Dq₃, I(2)⊗Dr₃]
  n₄, m₄ = size(𝐪𝐫₄)
  sbp_q₄ = SBP_1_2_CONSTANT_0_1(m₄)
  sbp_r₄ = SBP_1_2_CONSTANT_0_1(n₄)
  sbp_2d₄ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₄, sbp_r₄)
  𝐇q₀⁻¹₄, 𝐇qₙ⁻¹₄, 𝐇r₀⁻¹₄, _ = sbp_2d₄.norm
  Dq₄, Dr₄ = sbp_2d₄.D1
  Dqr₄ = [I(2)⊗Dq₄, I(2)⊗Dr₄]

  # Obtain some quantities on the grid points on Layer 1
  # Bulk Jacobian
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫₁)
  𝐉₁⁻¹ = 𝐉₁\(I(size(𝐉₁,1))) 
  # Impedance matrices
  𝐙₁₂¹ = 𝐙((Z₁¹,Z₂¹), Ω₁, 𝐪𝐫₁);
  𝛔₁₂¹ = 𝐙((x->τ(x)*Z₁¹(x), x->σ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫₁)
  𝛕₁₂¹ = 𝐙((x->τ(x)*σ(x)*Z₁¹(x), x->τ(x)*σ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫₁)
  𝛔ᵥ¹ = I(2) ⊗ spdiagm(σ.(Ω₁.(vec(𝐪𝐫₁))));  𝛔ₕ¹ = spzeros(size(𝛔ᵥ¹))
  𝛒₁ = I(2) ⊗ spdiagm(ρ₁.(Ω₁.(vec(𝐪𝐫₁))))
  # Get the transformed gradient
  Jqr₁ = J⁻¹.(𝐪𝐫₁, Ω₁);
  J_vec₁ = get_property_matrix_on_grid(Jqr₁, 2);
  J_vec_diag₁ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₁];
  Dx₁, Dy₁ = J_vec_diag₁*Dqr₁; 

  # Obtain some quantities on the grid points on Layer 2
  # Bulk Jacobian
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫₂)
  𝐉₂⁻¹ = 𝐉₂\(I(size(𝐉₂,1))) 
  # Impedance matrices
  𝐙₁₂² = 𝐙((Z₁²,Z₂²), Ω₂, 𝐪𝐫₂);
  𝛔₁₂² = 𝐙((x->τ(x)*Z₁²(x), x->σ(x)*Z₂²(x)), Ω₂, 𝐪𝐫₂)
  𝛕₁₂² = 𝐙((x->τ(x)*σ(x)*Z₁²(x), x->τ(x)*σ(x)*Z₂²(x)), Ω₂, 𝐪𝐫₂)  
  𝛔ᵥ² = I(2) ⊗ spdiagm(σ.(Ω₂.(vec(𝐪𝐫₂))));  𝛔ₕ² = spzeros(size(𝛔ᵥ²))
  𝛒₂ = I(2) ⊗ spdiagm(ρ₂.(Ω₂.(vec(𝐪𝐫₂))))
  # Get the transformed gradient
  Jqr₂ = J⁻¹.(𝐪𝐫₂, Ω₂);
  J_vec₂ = get_property_matrix_on_grid(Jqr₂, 2);
  J_vec_diag₂ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₂];
  Dx₂, Dy₂ = J_vec_diag₂*Dqr₂;

  # Obtain some quantities on the grid points on Layer 3
  # Bulk Jacobian
  𝐉₃ = Jb(𝛀₃, 𝐪𝐫₃)
  𝐉₃⁻¹ = 𝐉₃\(I(size(𝐉₃,1))) 
  # Impedance matrices
  𝐙₁₂³ = 𝐙((Z₁³,Z₂³), Ω₃, 𝐪𝐫₃);
  𝛔₁₂³ = 𝐙((x->τ(x)*Z₁³(x), x->σ(x)*Z₂³(x)), Ω₃, 𝐪𝐫₃)
  𝛕₁₂³ = 𝐙((x->τ(x)*σ(x)*Z₁³(x), x->τ(x)*σ(x)*Z₂³(x)), Ω₃, 𝐪𝐫₃)  
  𝛔ᵥ³ = I(2) ⊗ spdiagm(σ.(Ω₃.(vec(𝐪𝐫₃))));  𝛔ₕ³ = spzeros(size(𝛔ᵥ³))
  𝛒₃ = I(2) ⊗ spdiagm(ρ₃.(Ω₃.(vec(𝐪𝐫₃))))
  # Get the transformed gradient
  Jqr₃ = J⁻¹.(𝐪𝐫₂, Ω₂);
  J_vec₃ = get_property_matrix_on_grid(Jqr₃, 2);
  J_vec_diag₃ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₃];
  Dx₃, Dy₃ = J_vec_diag₃*Dqr₃;

  # Obtain some quantities on the grid points on Layer 4
  # Bulk Jacobian
  𝐉₄ = Jb(𝛀₄, 𝐪𝐫₄)
  𝐉₄⁻¹ = 𝐉₄\(I(size(𝐉₄,1))) 
  # Impedance matrices
  𝐙₁₂⁴ = 𝐙((Z₁⁴,Z₂⁴), Ω₄, 𝐪𝐫₄);
  𝛔₁₂⁴ = 𝐙((x->τ(x)*Z₁⁴(x), x->σ(x)*Z₂⁴(x)), Ω₄, 𝐪𝐫₄)
  𝛕₁₂⁴ = 𝐙((x->σ(x)*τ(x)*Z₁⁴(x), x->σ(x)*τ(x)*Z₂⁴(x)), Ω₄, 𝐪𝐫₄)  
  𝛔ᵥ⁴ = I(2) ⊗ spdiagm(σ.(Ω₄.(vec(𝐪𝐫₄))));  𝛔ₕ⁴ = spzeros(size(𝛔ᵥ⁴))
  𝛒₄ = I(2) ⊗ spdiagm(ρ₄.(Ω₄.(vec(𝐪𝐫₄))))
  # Get the transformed gradient
  Jqr₄ = J⁻¹.(𝐪𝐫₄, Ω₄);
  J_vec₄ = get_property_matrix_on_grid(Jqr₄, 2);
  J_vec_diag₄ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₄];
  Dx₄, Dy₄ = J_vec_diag₄*Dqr₄;

  # Surface Jacobian Matrices on Layer 1
  SJq₀¹, SJrₙ¹, SJqₙ¹ = 𝐉₁⁻¹*Js(𝛀₁, [-1,0];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [0,1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 2
  SJq₀², SJqₙ² = 𝐉₂⁻¹*Js(𝛀₂, [-1,0];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 3
  SJq₀³, SJqₙ³ =  𝐉₃⁻¹*Js(𝛀₃, [-1,0];  X=I(2)), 𝐉₃⁻¹*Js(𝛀₃, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 4
  SJr₀⁴, SJq₀⁴, SJqₙ⁴ =  𝐉₄⁻¹*Js(𝛀₄, [0,-1];  X=I(2)), 𝐉₄⁻¹*Js(𝛀₄, [-1,0];  X=I(2)), 𝐉₄⁻¹*Js(𝛀₄, [1,0];  X=I(2))

  # We build the governing equations on both layer simultaneously
  # Equation 1: ∂u/∂t = p
  EQ1₁ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₁)⊗I(n₁))
  EQ1₂ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₂)⊗I(n₂))
  EQ1₃ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₃)⊗I(n₃))
  EQ1₄ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₄)⊗I(n₄))

  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [E1(2,i,(6,6)) for i=1:6];
  eq2s₁ = [(𝐉₁⁻¹*𝐏₁)+α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)-𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, -𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹), 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₁, 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₂, -α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)+𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, 𝛒₁*𝛔ᵥ¹*𝛔ₕ¹];
  eq2s₂ = [(𝐉₂⁻¹*𝐏₂)+α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)-𝛒₂*𝛔ᵥ²*𝛔ₕ², -𝛒₂*(𝛔ᵥ²+𝛔ₕ²), 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₁, 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₂, -α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)+𝛒₂*𝛔ᵥ²*𝛔ₕ², 𝛒₂*𝛔ᵥ²*𝛔ₕ²];
  eq2s₃ = [(𝐉₃⁻¹*𝐏₃)+α*𝛒₃*(𝛔ᵥ³+𝛔ₕ³)-𝛒₃*𝛔ᵥ³*𝛔ₕ³, -𝛒₃*(𝛔ᵥ³+𝛔ₕ³), 𝐉₃⁻¹*𝐏₃ᴾᴹᴸ₁, 𝐉₃⁻¹*𝐏₃ᴾᴹᴸ₂, -α*𝛒₃*(𝛔ᵥ³+𝛔ₕ³)+𝛒₃*𝛔ᵥ³*𝛔ₕ³, 𝛒₃*𝛔ᵥ³*𝛔ₕ³];
  eq2s₄ = [(𝐉₄⁻¹*𝐏₄)+α*𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴)-𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴, -𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴), 𝐉₄⁻¹*𝐏₄ᴾᴹᴸ₁, 𝐉₄⁻¹*𝐏₄ᴾᴹᴸ₂, -α*𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴)+𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴, 𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴];
  EQ2₁ = sum(es .⊗ eq2s₁);  
  EQ2₂ = sum(es .⊗ eq2s₂);
  EQ2₃ = sum(es .⊗ eq2s₃);
  EQ2₄ = sum(es .⊗ eq2s₄);

  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [E1(3,i,(6,6)) for i=[1,3]];
  eq3s₁ = [Dx₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ᵥ¹)];
  eq3s₂ = [Dx₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ᵥ²)];
  eq3s₃ = [Dx₃, -(α*(I(2)⊗I(m₃)⊗I(n₃)) + 𝛔ᵥ³)];
  eq3s₄ = [Dx₄, -(α*(I(2)⊗I(m₄)⊗I(n₄)) + 𝛔ᵥ⁴)];
  EQ3₁ = sum(es .⊗ eq3s₁);
  EQ3₂ = sum(es .⊗ eq3s₂);
  EQ3₃ = sum(es .⊗ eq3s₃);
  EQ3₄ = sum(es .⊗ eq3s₄);

  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [E1(4,i,(6,6)) for i=[1,4]]
  eq4s₁ = [Dy₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ₕ¹)]
  eq4s₂ = [Dy₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ₕ²)]
  eq4s₃ = [Dy₃, -(α*(I(2)⊗I(m₃)⊗I(n₃)) + 𝛔ₕ³)]
  eq4s₄ = [Dy₄, -(α*(I(2)⊗I(m₄)⊗I(n₄)) + 𝛔ₕ⁴)]
  EQ4₁ = sum(es .⊗ eq4s₁)
  EQ4₂ = sum(es .⊗ eq4s₂)
  EQ4₃ = sum(es .⊗ eq4s₃)
  EQ4₄ = sum(es .⊗ eq4s₄)

  # Equation 5 ∂q/∂t = α(u-q)
  es = [E1(5,i,(6,6)) for i=[1,5]]
  eq5s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  eq5s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]
  eq5s₃ = [α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃))]
  eq5s₄ = [α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄))]
  EQ5₁ = sum(es .⊗ eq5s₁)
  EQ5₂ = sum(es .⊗ eq5s₂)
  EQ5₃ = sum(es .⊗ eq5s₃)
  EQ5₄ = sum(es .⊗ eq5s₄)

  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [E1(6,i,(6,6)) for i=[1,5,6]]
  eq6s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  eq6s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]
  eq6s₃ = [α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃))]
  eq6s₄ = [α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄))]
  EQ6₁ = sum(es .⊗ eq6s₁)
  EQ6₂ = sum(es .⊗ eq6s₂)
  EQ6₃ = sum(es .⊗ eq6s₃)
  EQ6₄ = sum(es .⊗ eq6s₄)

  # Traction free boundary condition on Top
  Trₙ¹ = Tᴱ(Pqr₁, 𝛀₁, [0;1]).A
  Trₙᴾᴹᴸ₁₁, Trₙᴾᴹᴸ₂₁ = Tᴾᴹᴸ(Pᴾᴹᴸqr₁, 𝛀₁, [0;1]).A 
  es = [E1(2,i,(6,6)) for i=[1,3,4]];
  𝐓rₙ¹ = [Trₙ¹, Trₙᴾᴹᴸ₁₁, Trₙᴾᴹᴸ₂₁]
  # The SAT Terms on the boundary 
  SJ_𝐇rₙ⁻¹₁ = (fill(SJrₙ¹,3).*fill((I(2)⊗𝐇rₙ⁻¹₁),3));
  SAT₁ = sum(es.⊗(SJ_𝐇rₙ⁻¹₁.*𝐓rₙ¹));  

  # PML characteristic boundary conditions on Left and Right
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ¹ = Pqr₁, Pᴾᴹᴸqr₁, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, 𝐉₁;
  χq₀¹, χqₙ¹ = χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [-1,0]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [1,0]).A
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₁ = (fill(SJq₀¹,6).*fill((I(2)⊗𝐇q₀⁻¹₁),6));
  SJ_𝐇qₙ⁻¹₁ = (fill(SJqₙ¹,6).*fill((I(2)⊗𝐇qₙ⁻¹₁),6));  
  SAT₁ += sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹))
  
  # PML characteristic boundary conditions
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ² = Pqr₂, Pᴾᴹᴸqr₂, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², 𝐉₂;
  χq₀², χqₙ² = χᴾᴹᴸ(PQRᵪ², 𝛀₂, [-1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [1,0]).A
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₂ = (fill(SJq₀²,6).*fill((I(2)⊗𝐇q₀⁻¹₂),6));
  SJ_𝐇qₙ⁻¹₂ = (fill(SJqₙ²,6).*fill((I(2)⊗𝐇qₙ⁻¹₂),6));
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²));

  PQRᵪ³ = Pqr₃, Pᴾᴹᴸqr₃, 𝐙₁₂³, 𝛔₁₂³, 𝛕₁₂³, 𝐉₃;
  χq₀³, χqₙ³ = χᴾᴹᴸ(PQRᵪ³, 𝛀₃, [-1,0]).A, χᴾᴹᴸ(PQRᵪ³, 𝛀₃, [1,0]).A
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₃ = (fill(SJq₀³,6).*fill((I(2)⊗𝐇q₀⁻¹₃),6));
  SJ_𝐇qₙ⁻¹₃ = (fill(SJqₙ³,6).*fill((I(2)⊗𝐇qₙ⁻¹₃),6));
  SAT₃ = sum(es.⊗(SJ_𝐇q₀⁻¹₃.*χq₀³)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₃.*χqₙ³));

  PQRᵪ⁴ = Pqr₄, Pᴾᴹᴸqr₄, 𝐙₁₂⁴, 𝛔₁₂⁴, 𝛕₁₂⁴, 𝐉₄;
  χq₀⁴, χr₀⁴, χqₙ⁴ = χᴾᴹᴸ(PQRᵪ⁴, 𝛀₄, [-1,0]).A, χᴾᴹᴸ(PQRᵪ⁴, 𝛀₄, [0,-1]).A, χᴾᴹᴸ(PQRᵪ⁴, 𝛀₄, [1,0]).A
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₄ = (fill(SJq₀⁴,6).*fill((I(2)⊗𝐇q₀⁻¹₄),6));
  SJ_𝐇qₙ⁻¹₄ = (fill(SJqₙ⁴,6).*fill((I(2)⊗𝐇qₙ⁻¹₄),6));
  SJ_𝐇r₀⁻¹₄ = (fill(SJr₀⁴,6).*fill((I(2)⊗𝐇r₀⁻¹₄),6));
  SAT₄ = sum(es.⊗(SJ_𝐇q₀⁻¹₄.*χq₀⁴)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₄.*χqₙ⁴)) + sum(es.⊗(SJ_𝐇r₀⁻¹₄.*χr₀⁴));

  # The interface part
  Eᵢ¹ = E1(2,1,(6,6)) ⊗ I(2)
  Eᵢ² = E1(1,1,(6,6)) ⊗ I(2)
  # Get the jump matrices on the three interfaces
  # Layers 1-2
  B̂₁,  B̃₁,  _ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂ᵀ₁, _, 𝐇₁⁻¹₁, 𝐇₂⁻¹₁ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)
  # Layers 2-3
  B̂₂,  B̃₂,  _ = SATᵢᴱ(𝛀₂, 𝛀₃, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂ᵀ₂, _, 𝐇₁⁻¹₂, 𝐇₂⁻¹₂ = SATᵢᴱ(𝛀₂, 𝛀₃, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)
  # Layers 3-4
  B̂₃,  B̃₃,  _ = SATᵢᴱ(𝛀₃, 𝛀₄, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂ᵀ₃, _, 𝐇₁⁻¹₃, 𝐇₂⁻¹₃ = SATᵢᴱ(𝛀₃, 𝛀₄, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)
  # Traction on interface From Layer 1
  Tr₀¹ = Tᴱ(Pqr₁, 𝛀₁, [0;-1]).A
  Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁ = Tᴾᴹᴸ(Pᴾᴹᴸqr₁, 𝛀₁, [0;-1]).A  
  # Traction on interfaces From Layer 2
  Tr₀² = Tᴱ(Pqr₂, 𝛀₂, [0;-1]).A
  Tr₀ᴾᴹᴸ₁₂, Tr₀ᴾᴹᴸ₂₂ = Tᴾᴹᴸ(Pᴾᴹᴸqr₂, 𝛀₂, [0;-1]).A
  Trₙ² = Tᴱ(Pqr₂, 𝛀₂, [0;1]).A
  Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂ = Tᴾᴹᴸ(Pᴾᴹᴸqr₂, 𝛀₂, [0;1]).A
  # Traction on interface From Layer 3
  Tr₀³ = Tᴱ(Pqr₃, 𝛀₃, [0;-1]).A
  Tr₀ᴾᴹᴸ₁₃, Tr₀ᴾᴹᴸ₂₃ = Tᴾᴹᴸ(Pᴾᴹᴸqr₃, 𝛀₃, [0;-1]).A
  Trₙ³ = Tᴱ(Pqr₃, 𝛀₃, [0;1]).A
  Trₙᴾᴹᴸ₁₃, Trₙᴾᴹᴸ₂₃ = Tᴾᴹᴸ(Pᴾᴹᴸqr₃, 𝛀₃, [0;1]).A  
  # Traction on interface From Layer 4
  Trₙ⁴ = Tᴱ(Pqr₄, 𝛀₄, [0;1]).A
  Trₙᴾᴹᴸ₁₄, Trₙᴾᴹᴸ₂₄ = Tᴾᴹᴸ(Pᴾᴹᴸqr₄, 𝛀₄, [0;1]).A
  # Assemble the traction on the two layers
  # Layer 1
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓r₀¹ = sum(es .⊗ [Tr₀¹, Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁])
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀ₀¹ = sum(es .⊗ [(Tr₀¹)', (Tr₀ᴾᴹᴸ₁₁)', (Tr₀ᴾᴹᴸ₂₁)'])  
  # Layer 2
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓rₙ² = sum(es .⊗ [Trₙ², Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂])  
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓r₀² = sum(es .⊗ [Tr₀², Tr₀ᴾᴹᴸ₁₂, Tr₀ᴾᴹᴸ₂₂])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀₙ² = sum(es .⊗ [(Trₙ²)', (Trₙᴾᴹᴸ₁₂)', (Trₙᴾᴹᴸ₂₂)'])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀ₀² = sum(es .⊗ [(Tr₀²)', (Tr₀ᴾᴹᴸ₁₂)', (Tr₀ᴾᴹᴸ₂₂)'])  
  # Layer 3
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓rₙ³ = sum(es .⊗ [Trₙ³, Trₙᴾᴹᴸ₁₃, Trₙᴾᴹᴸ₂₃])  
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓r₀³ = sum(es .⊗ [Tr₀³, Tr₀ᴾᴹᴸ₁₃, Tr₀ᴾᴹᴸ₂₃])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀₙ³ = sum(es .⊗ [(Trₙ³)', (Trₙᴾᴹᴸ₁₃)', (Trₙᴾᴹᴸ₂₃)'])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀ₀³ = sum(es .⊗ [(Tr₀³)', (Tr₀ᴾᴹᴸ₁₃)', (Tr₀ᴾᴹᴸ₂₃)'])  
  # Layer 4   
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓rₙ⁴ = sum(es .⊗ [Trₙ⁴, Trₙᴾᴹᴸ₁₄, Trₙᴾᴹᴸ₂₄])
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀₙ⁴ = sum(es .⊗ [(Trₙ⁴)', (Trₙᴾᴹᴸ₁₄)', (Trₙᴾᴹᴸ₂₄)'])  

  𝐓rᵢ¹ = blockdiag(𝐓r₀¹, 𝐓rₙ²)      
  𝐓rᵢ² = blockdiag(𝐓r₀², 𝐓rₙ³)      
  𝐓rᵢ³ = blockdiag(𝐓r₀³, 𝐓rₙ⁴)      
  𝐓rᵢ¹ᵀ = blockdiag(𝐓rᵀ₀¹, 𝐓rᵀₙ²)   
  𝐓rᵢ²ᵀ = blockdiag(𝐓rᵀ₀², 𝐓rᵀₙ³)   
  𝐓rᵢ³ᵀ = blockdiag(𝐓rᵀ₀³, 𝐓rᵀₙ⁴)   
  h = norm(Ω₁(𝐪𝐫₁[1,1]) - Ω₁(𝐪𝐫₁[1,2]))
  ζ₀ = 30*5.196/h  
  # Assemble the interface SAT
  𝐉₁₂ = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₁⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₂⁻¹)
  𝐉₂₃ = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₂⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₃⁻¹)
  𝐉₃₄ = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₃⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₄⁻¹)
  𝐓ᵢ¹ = blockdiag(I(12)⊗𝐇₁⁻¹₁, I(12)⊗𝐇₂⁻¹₁)*𝐉₁₂*(0.5*B̂₁*𝐓rᵢ¹ - 0.5*𝐓rᵢ¹ᵀ*B̂ᵀ₁ - ζ₀*B̃₁)
  𝐓ᵢ² = blockdiag(I(12)⊗𝐇₁⁻¹₂, I(12)⊗𝐇₂⁻¹₂)*𝐉₂₃*(0.5*B̂₂*𝐓rᵢ² - 0.5*𝐓rᵢ²ᵀ*B̂ᵀ₂ - ζ₀*B̃₂)
  𝐓ᵢ³ = blockdiag(I(12)⊗𝐇₁⁻¹₃, I(12)⊗𝐇₂⁻¹₃)*𝐉₃₄*(0.5*B̂₃*𝐓rᵢ³ - 0.5*𝐓rᵢ³ᵀ*B̂ᵀ₃ - ζ₀*B̃₃)

  SATᵢ¹ = blockdiag(𝐓ᵢ¹, zero(EQ1₃), zero(EQ1₄))
  SATᵢ² = blockdiag(zero(EQ1₁), 𝐓ᵢ², zero(EQ1₄))
  SATᵢ³ = blockdiag(zero(EQ1₁), zero(EQ1₂), 𝐓ᵢ³)

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), 
                   (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂),
                   (EQ1₃ + EQ2₃ + EQ3₃ + EQ4₃ + EQ5₃ + EQ6₃),
                   (EQ1₄ + EQ2₄ + EQ3₄ + EQ4₄ + EQ5₄ + EQ6₄));  
  SATₙ = blockdiag(SAT₁, SAT₂, SAT₃, SAT₄)
  bulk - SATᵢ¹ - SATᵢ² - SATᵢ³ - SATₙ
  # (SAT₁, SAT₂, SAT₃, SAT₄), (SATᵢ¹, SATᵢ², SATᵢ³), 
  # ((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂),
  # (EQ1₃ + EQ2₃ + EQ3₃ + EQ4₃ + EQ5₃ + EQ6₃), (EQ1₄ + EQ2₄ + EQ3₄ + EQ4₄ + EQ5₄ + EQ6₄)),
  # blockdiag(I(12)⊗𝐇₁⁻¹₁, I(12)⊗𝐇₂⁻¹₁)*𝐉₁₂*(0.5*B̂₁*𝐓rᵢ¹ - 0.5*𝐓rᵢ¹ᵀ*B̂ᵀ₁ - ζ₀*B̃₁)
end

"""
Inverse of the mass matrix
"""
function 𝐌4⁻¹ₚₘₗ(𝛀::NTuple{4,DiscreteDomain}, 𝐪𝐫, ρ)
  ρ₁, ρ₂, ρ₃, ρ₄ = ρ
  𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄ = 𝛀
  𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃, 𝐪𝐫₄ = 𝐪𝐫
  m₁, n₁ = size(𝐪𝐫₁)
  m₂, n₂ = size(𝐪𝐫₂)
  m₃, n₃ = size(𝐪𝐫₃)
  m₄, n₄ = size(𝐪𝐫₄)
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Id₃ = sparse(I(2)⊗I(m₃)⊗I(n₃))
  Id₄ = sparse(I(2)⊗I(m₄)⊗I(n₄))
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);
  Ω₃(qr) = S(qr, 𝛀₃.domain);
  Ω₄(qr) = S(qr, 𝛀₄.domain);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁.(Ω₁.(𝐪𝐫₁))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂.(Ω₂.(𝐪𝐫₂))))
  ρᵥ³ = I(2)⊗spdiagm(vec(1 ./ρ₃.(Ω₃.(𝐪𝐫₃))))
  ρᵥ⁴ = I(2)⊗spdiagm(vec(1 ./ρ₄.(Ω₄.(𝐪𝐫₄))))
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁, Id₁), 
            blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂, Id₂),
            blockdiag(Id₃, ρᵥ³, Id₃, Id₃, Id₃, Id₃),
            blockdiag(Id₄, ρᵥ⁴, Id₄, Id₄, Id₄, Id₄))
end 

"""
A non-allocating implementation of the RK4 scheme
"""
function RK4_1!(M, sol, Δt)  
  X₀, k₁, k₂, k₃, k₄ = sol  
  k₁ .= M*(X₀)
  k₂ .= M*(X₀+0.5*Δt*k₁)
  k₃ .= M*(X₀+0.5*Δt*k₂)
  k₄ .= M*(X₀+Δt*k₃)
  X₀ .+= (Δt/6)*(k₁ + 2*k₂ + 2*k₃ + k₄)
end

"""
Right hand side function
"""
function f(t::Float64, x::SVector{2,Float64}, params)
  s₁, s₂, M₀ = params
  @SVector[-1/(2π*√(s₁*s₂))*exp(-(x[1]-20)^2/(2s₁) - (x[2]+15)^2/(2s₂))*(x[1]-20)/s₁*exp(-(t-0.215)^2/0.15)*M₀,
           -1/(2π*√(s₁*s₂))*exp(-(x[1]-20)^2/(2s₁) - (x[2]+15)^2/(2s₂))*(x[2]+15)/s₂*exp(-(t-0.215)^2/0.15)*M₀]
end

"""
A non-allocating implementation of the RK4 scheme with forcing
"""
function RK4_1!(MK, sol, Δt, F, M)  
  X₀, k₁, k₂, k₃, k₄ = sol
  F₁, F₂, F₄ = F
  k₁ .= MK*(X₀) + M*F₁
  k₂ .= MK*(X₀+0.5*Δt*k₁) + M*F₂
  k₃ .= MK*(X₀+0.5*Δt*k₂) + M*F₂
  k₄ .= MK*(X₀+Δt*k₃) + M*F₄
  X₀ .+= (Δt/6)*(k₁ + 2*k₂ + 2*k₃ + k₄)
end

"""
Function to split the solution into the corresponding variables
"""
function split_solution(X, MN, P)    
  res = splitdimsview(reshape(X, (prod(MN), P)))
  u1, u2 = res[1:2]
  (u1,u2)
end

"""
Initial conditions
"""
# 𝐔(x) = @SVector [exp(-5*((x[1]-20)^2 + (x[2]+15)^2)), exp(-5*((x[1]-20)^2 + (x[2]+15)^2))]
𝐔(x) = @SVector [0.0, 0.0]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

h = 0.1;
Nx = ceil(Int64, 48/h) + 1;
Ny = ceil(Int64, 10/h) + 1;
Ny1 = ceil(Int64, 14/h) + 1;
𝛀₁ = DiscreteDomain(domain₁, (Nx, Ny));
𝛀₂ = DiscreteDomain(domain₂, (Nx, Ny));
𝛀₃ = DiscreteDomain(domain₃, (Nx, Ny));
𝛀₄ = DiscreteDomain(domain₄, (Nx, Ny1));
Ω₁(qr) = S(qr, 𝛀₁.domain);
Ω₂(qr) = S(qr, 𝛀₂.domain);
Ω₃(qr) = S(qr, 𝛀₃.domain);
Ω₄(qr) = S(qr, 𝛀₄.domain);
𝐪𝐫₁ = generate_2d_grid((Nx, Ny));
𝐪𝐫₂ = generate_2d_grid((Nx, Ny));
𝐪𝐫₃ = generate_2d_grid((Nx, Ny));
𝐪𝐫₄ = generate_2d_grid((Nx, Ny1));
xy₁ = Ω₁.(𝐪𝐫₁);
xy₂ = Ω₂.(𝐪𝐫₂);
xy₃ = Ω₃.(𝐪𝐫₃);
xy₄ = Ω₄.(𝐪𝐫₄);
stima = 𝐊4ₚₘₗ((𝒫₁, 𝒫₂, 𝒫₃, 𝒫₄), (𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ, 𝒫₃ᴾᴹᴸ, 𝒫₄ᴾᴹᴸ), ((Z₁¹, Z₂¹), (Z₁², Z₂²), (Z₁³, Z₂³), (Z₁⁴, Z₂⁴)), (𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄), (𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃, 𝐪𝐫₄));
massma = 𝐌4⁻¹ₚₘₗ((𝛀₁, 𝛀₂, 𝛀₃, 𝛀₄), (𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃, 𝐪𝐫₄), (ρ₁, ρ₂, ρ₃, ρ₄));
# Define the time stepping
Δt = 0.2*h/sqrt(max((cp₁^2+cs₁^2), (cp₂^2+cs₂^2), (cp₃^2+cs₃^2), (cp₄^2+cs₄^2)));
tf = 1000.0
ntime = ceil(Int, tf/Δt)
Δt = tf/ntime;
maxvals = zeros(Float64, ntime);

const param = (0.5*h, 0.5*h, 1000)

plt3 = Vector{Plots.Plot}(undef,3+ceil(Int64, tf/10));

# Begin time loop
let
  t = 0.0
  X₀¹ = vcat(eltocols(vec(𝐔.(xy₁))), eltocols(vec(𝐏.(xy₁))), eltocols(vec(𝐕.(xy₁))), eltocols(vec(𝐖.(xy₁))), eltocols(vec(𝐐.(xy₁))), eltocols(vec(𝐑.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔.(xy₂))), eltocols(vec(𝐏.(xy₂))), eltocols(vec(𝐕.(xy₂))), eltocols(vec(𝐖.(xy₂))), eltocols(vec(𝐐.(xy₂))), eltocols(vec(𝐑.(xy₂))));
  X₀³ = vcat(eltocols(vec(𝐔.(xy₃))), eltocols(vec(𝐏.(xy₃))), eltocols(vec(𝐕.(xy₃))), eltocols(vec(𝐖.(xy₃))), eltocols(vec(𝐐.(xy₃))), eltocols(vec(𝐑.(xy₃))));
  X₀⁴ = vcat(eltocols(vec(𝐔.(xy₄))), eltocols(vec(𝐏.(xy₄))), eltocols(vec(𝐕.(xy₄))), eltocols(vec(𝐖.(xy₄))), eltocols(vec(𝐐.(xy₄))), eltocols(vec(𝐑.(xy₄))));

  X₀ = vcat(X₀¹, X₀², X₀³, X₀⁴)
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  M = massma*stima
  count = 1;
  # @gif for i=1:ntime
  Hq = SBP_1_2_CONSTANT_0_1(Nx).norm;
  Hr = SBP_1_2_CONSTANT_0_1(Ny).norm;
  Hr1 = SBP_1_2_CONSTANT_0_1(Ny1).norm;
  Hqr = Hq ⊗ Hr
  Hqr1 = Hq ⊗ Hr1
  function 𝐅(t, xy, Z2) 
    Z, Z1 = Z2
    xy₁, xy₂, xy₃, xy₄ = xy    
    [Z; eltocols(f.(Ref(t), vec(xy₁), Ref(param))); Z; Z; Z; Z;
     Z; eltocols(f.(Ref(t), vec(xy₂), Ref(param))); Z; Z; Z; Z;
     Z; eltocols(f.(Ref(t), vec(xy₃), Ref(param))); Z; Z; Z; Z;
     Z1; eltocols(f.(Ref(t), vec(xy₄), Ref(param))); Z1; Z1; Z1; Z1]
  end
  xys =  xy₁, xy₂, xy₃, xy₄
  Z = zeros(2*length(xy₁))
  Z1 = zeros(2*length(xy₄))
  for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    # # This block is for the moment-source function
    Fs = (𝐅((i-1)*Δt, xys, (Z,Z1)), 𝐅((i-0.5)Δt, xys, (Z,Z1)), 𝐅(i*Δt, xys, (Z,Z1)))
    X₀ = RK4_1!(M, sol, Δt, Fs, massma)
    # X₀ = RK4_1!(M, sol, Δt)    
    t += Δt    
    (i%ceil(Int64,ntime/20)==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))

    u1ref₁,u2ref₁ = split_solution(X₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
    u1ref₂,u2ref₂ = split_solution(X₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
    u1ref₃,u2ref₃ = split_solution(X₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
    u1ref₄,u2ref₄ = split_solution(X₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))+12*(prod(𝛀₄.mn))], 𝛀₄.mn, 12);
    
    U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
    U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)
    U3 = sqrt.(u1ref₃.^2 + u2ref₃.^2)
    U4 = sqrt.(u1ref₄.^2 + u2ref₄.^2)
    
    if((i==ceil(Int64, 3/Δt)) || (i == ceil(Int64, 5/Δt)) || (i == ceil(Int64, 9/Δt)) || ((i*Δt)%10 ≈ 0.0))
      plt3[count] = Plots.contourf(getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₂), getY.(xy₂), reshape(U2,size(xy₂)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₃), getY.(xy₃), reshape(U3,size(xy₃)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₄), getY.(xy₄), reshape(U4,size(xy₄)...), colormap=:jet)
      Plots.vline!(plt3[count], [L], label="\$ x \\ge "*string(round(L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.vline!(plt3[count], [0], label="\$ x \\ge "*string(round(0, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.hline!(plt3[count], [-L], label="\$ y \\ge "*string(round(-L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.plot!(plt3[count], getX.(interface₁.(LinRange(0,1,100))), getY.(interface₁.(LinRange(0,1,100))), label="Interface 1", lc=:red, lw=2, legend=:none)
      Plots.plot!(plt3[count], getX.(interface₂.(LinRange(0,1,100))), getY.(interface₂.(LinRange(0,1,100))), label="Interface 2", lc=:red, lw=2, legend=:none)
      Plots.plot!(plt3[count], getX.(interface₃.(LinRange(0,1,100))), getY.(interface₃.(LinRange(0,1,100))), label="Interface 3", lc=:red, lw=2,  aspect_ratio=1.09, legend=:none)
      xlims!(plt3[count], (0-δ,L+δ))
      ylims!(plt3[count], (-L-δ,0))
      xlabel!(plt3[count], "\$x\$")
      ylabel!(plt3[count], "\$y\$")
      count += 1
    end

    maxvals[i] = sqrt(u1ref₁'*Hqr*u1ref₁ + u2ref₁'*Hqr*u2ref₁ +
                      u1ref₂'*Hqr*u1ref₂ + u2ref₂'*Hqr*u2ref₂ + 
                      u1ref₃'*Hqr*u1ref₃ + u2ref₃'*Hqr*u2ref₃ + 
                      u1ref₄'*Hqr1*u1ref₄ + u2ref₄'*Hqr1*u2ref₄)
  end
  # end  every 10  
  global Xref = X₀
end;

u1ref₁,u2ref₁ = split_solution(Xref[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
u1ref₂,u2ref₂ = split_solution(Xref[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
u1ref₃,u2ref₃ = split_solution(Xref[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
u1ref₄,u2ref₄ = split_solution(Xref[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))+12*(prod(𝛀₄.mn))], 𝛀₄.mn, 12);

U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)*sqrt(0.5)
U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)*sqrt(0.5)
U3 = sqrt.(u1ref₃.^2 + u2ref₃.^2)*sqrt(0.5)
U4 = sqrt.(u1ref₄.^2 + u2ref₄.^2)*sqrt(0.5)

plt3_1 = Plots.plot();
Plots.contourf!(plt3_1, getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₂), getY.(xy₂), reshape(U2, size(xy₂)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₃), getY.(xy₃), reshape(U3,size(xy₃)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₄), getY.(xy₄), reshape(U4,size(xy₄)...), colormap=:jet)
Plots.vline!(plt3_1, [L], label="\$ x \\ge "*string(round(L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.vline!(plt3_1, [0], label="\$ x \\ge "*string(round(0, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.hline!(plt3_1, [-L], label="\$ y \\ge "*string(round(-L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.plot!(plt3_1, getX.(interface₁.(LinRange(0,1,100))), getY.(interface₁.(LinRange(0,1,100))), label="Interface 1", lc=:red, lw=2, legend=:none)
Plots.plot!(plt3_1, getX.(interface₂.(LinRange(0,1,100))), getY.(interface₂.(LinRange(0,1,100))), label="Interface 2", lc=:red, lw=2, legend=:none)
Plots.plot!(plt3_1, getX.(interface₃.(LinRange(0,1,100))), getY.(interface₃.(LinRange(0,1,100))), label="Interface 3", lc=:red, lw=2, legend=:none, aspect_ratio=1.09)
xlims!(plt3_1, (0-δ,L+δ))
ylims!(plt3_1, (-L-δ,0.0))
xlabel!(plt3_1, "\$x\$")
ylabel!(plt3_1, "\$y\$")
# c_ticks = (LinRange(2.5e-6,1.0e-5,5), string.(round.(LinRange(1.01,7.01,5), digits=4)).*"\$ \\times 10^{-7}\$");
# Plots.plot!(plt3_1, colorbar_ticks=c_ticks)

plt5 = Plots.plot(LinRange(0,tf,ntime), maxvals, label="", lw=1, yaxis=:log10)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| \\bf{u} \\|_{H} \$")
# Plots.xlims!(plt5, (0,1000))