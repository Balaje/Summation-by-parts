# include("2d_elasticity_problem.jl");
using SummationByPartsPML
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
# using LoopVectorization

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Get the x-and-y coordinates from coordinates
"""
getX(C) = C[1]; getY(C) = C[2];

# Define the domain
cᵢ(q) = @SVector [4.4π*q, 0.8π*exp(-40π*(q-0.5)^2)]
c₀¹(r) = @SVector [0.0, 4π*r]
c₁¹(q) = cᵢ(q)
c₂¹(r) = @SVector [4.4π, 4π*r]
c₃¹(q) = @SVector [4.4π*q, 4π]
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
c₀²(r) = @SVector [0.0, 4π*r - 4π]
c₁²(q) = @SVector [4.4π*q, -4π]
c₂²(r) = @SVector [4.4π, 4π*r-4π]
c₃²(q) = cᵢ(q)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)


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

"""
The PML damping
"""
const Lᵥ = 4π
const Lₕ = 4π
const δ = 0.1*Lᵥ
const σ₀ᵛ = 4*((max(cp₁, cp₂)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*((max(cs₁, cs₂)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
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
  elseif( (x[2] ≈ -Lₕ) || (x[2] < -Lₕ) )
    return σ₀ʰ*abs((x[2] + Lₕ)/δ)^3  
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
Material velocity tensors
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

"""
Function to transform the material properties to the reference domain
"""
function transform_material_properties_to_reference_domain(props, domain, reference_coords)
  𝒫, 𝒫ᴾᴹᴸ = props  
  reference_grid_material_properties = transform_material_properties.(𝒫, domain, reference_coords)
  reference_grid_material_properties_pml = transform_material_properties_pml.(𝒫ᴾᴹᴸ, domain, reference_coords)
  reference_grid_material_properties, reference_grid_material_properties_pml
end

"""
Function to compute the bulk elasticity operator on the reference grid from the material property functions
"""
function compute_bulk_elasticity_operators(props, domain, reference_coords)
  reference_grid_material_properties, reference_grid_material_properties_pml = transform_material_properties_to_reference_domain(props, domain, reference_coords)
  bulk_elasticity_operator = elasticity_operator(reference_grid_material_properties).A
  bulk_elasticity_pml_operator = elasticity_pml_operator(reference_grid_material_properties_pml).A
  bulk_elasticity_operator, bulk_elasticity_pml_operator
end
function compute_bulk_elasticity_operators(material_properties::NTuple{2, Matrix{SMatrix{4, 4, Float64, 16}}})
  reference_grid_material_properties, reference_grid_material_properties_pml = material_properties
  bulk_elasticity_operator = elasticity_operator(reference_grid_material_properties).A
  bulk_elasticity_pml_operator = elasticity_pml_operator(reference_grid_material_properties_pml).A
  bulk_elasticity_operator, bulk_elasticity_pml_operator
end

"""
Function to compute the gradient operators
"""

function compute_gradient_operators_on_physical_domain(domain, reference_grid)
  I₂ = Ref(I(2))
  n, m = size(reference_grid)
  sbp_q = SBP4_1D(m)
  sbp_r = SBP4_1D(n)
  sbp_2d = SBP4_2D(sbp_q, sbp_r)  
  Dq, Dr = sbp_2d.D1  
  Dqr = kron.(I₂, [Dq, Dr]) # Each displacement has two fields
  Jqr = inverse_transfinite_interpolation_jacobian.(reference_grid, domain);
  J_vec = get_property_matrix_on_grid(Jqr, 2);
  J_vec_diag = kron.(I₂, spdiagm.(vec.(J_vec)));
  J_vec_diag*Dqr;
end


"""
Function to calculate the norm inverse on the four boundaries
"""
function get_sbp_norm_2d(sbp_2d::SBP4_2D)
  sbp_2d.norm
end

"""
Function to calculate the SBP operators on the reference grid
"""
function get_sbp_operators_on_reference_grid(reference_grid)
  n, m = size(reference_grid)
  sbp_q = SBP4_1D(m)
  sbp_r = SBP4_1D(n)
  sbp_2d = SBP4_2D(sbp_q, sbp_r) 
  sbp_2d
end

"""
Function to compute the coefficients of the RHS of the PML modified elastic wave equation
"""
function get_pml_elastic_wave_coefficients(material_properties, domain, reference_grid)
  Z₁₂, σₕσᵥ, ρ = material_properties
  # Extract the material property functions
  Z₁, Z₂ = Z₁₂  
  # Extract the PML damping functions
  σₕ, σᵥ = σₕσᵥ
  # Extract the density of the materials  
  𝐙₁₂ = compute_impedance_function((Z₁, Z₂), domain, reference_grid)
  𝛔₁₂ = compute_impedance_function((x->σₕ(x)*Z₁(x), x->σᵥ(x)*Z₂(x)), domain, reference_grid)
  𝛕₁₂ = compute_impedance_function((x->σₕ(x)*σᵥ(x)*Z₁(x), x->σₕ(x)*σᵥ(x)*Z₂(x)), domain, reference_grid)
  𝛔ᵥ = I(2) ⊗ spdiagm(σᵥ.(domain.(vec(reference_grid))))  
  𝛔ₕ = I(2) ⊗ spdiagm(σₕ.(domain.(vec(reference_grid))))
  𝛒  = I(2) ⊗ spdiagm(ρ.(domain.(vec(reference_grid))))
  𝐙₁₂, 𝛔₁₂, 𝛕₁₂, (𝛔ᵥ, 𝛔ₕ), 𝛒
end

"""
Function to compute the surface Jacobian
"""
function compute_surface_jacobian_matrices_on_domain(domain, reference_coords, J⁻¹)
  (J⁻¹*surface_jacobian(domain, reference_coords, [-1,0];  X=I(2)), J⁻¹*surface_jacobian(domain, reference_coords, [1,0];  X=I(2)), 
   J⁻¹*surface_jacobian(domain, reference_coords, [0,-1];  X=I(2)), J⁻¹*surface_jacobian(domain, reference_coords, [0,1];  X=I(2)))
end

"""
Function to compute the absorbing boundary conditions on the domain
"""
function compute_absorbing_boundary_conditions_on_domain(domain, reference_coords, coeffs)
  (elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [-1,0]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [1,0]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [0,-1]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [0,1]).A)
end

"""
Function to compute the surface integration operator for the SAT terms
"""
function compute_surface_integration_operators(sbp_2d::SBP4_2D, surface_jacobian_matrices::NTuple{4, AbstractMatrix{Float64}})
  SJq₀, SJqₙ, SJr₀, SJrₙ = surface_jacobian_matrices
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = get_sbp_norm_2d(sbp_2d) 
  I₂ = I(2)
  (fill(SJq₀*(I₂⊗𝐇q₀⁻¹), 6), fill(SJqₙ*(I₂⊗𝐇qₙ⁻¹), 6),
   fill(SJr₀*(I₂⊗𝐇r₀⁻¹), 6), fill(SJrₙ*(I₂⊗𝐇rₙ⁻¹), 6))
end

"""
Function to obtain the PML stiffness matrix
"""
function two_layer_elasticity_pml_stiffness_matrix(domains::NTuple{2, domain_2d}, reference_grids::NTuple{2, AbstractMatrix{SVector{2,Float64}}}, material_properties)
  # Extract domain
  domain₁, domain₂ = domains
  Ω₁(qr) = transfinite_interpolation(qr, domain₁)
  Ω₂(qr) = transfinite_interpolation(qr, domain₂)
  qr₁, qr₂ = reference_grids  
  𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α = material_properties
  # Extract the material property functions
  # (Z₁¹, Z₂¹), (Z₁², Z₂²) = Z₁₂
  Z¹₁₂, Z²₁₂ = Z₁₂
  Z₁¹, Z₂¹ = Z¹₁₂
  Z₁², Z₂² = Z²₁₂
  # Extract the elastic material tensors
  𝒫₁, 𝒫₂ = 𝒫
  𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ = 𝒫ᴾᴹᴸ
  # Extract the PML damping functions
  # σₕ, σᵥ = σₕσᵥ
  # Extract the density of the materials
  ρ₁, ρ₂ = ρ
  # Get the discretization 
  n₁, m₁ = size(qr₁)
  n₂, m₂ = size(qr₂)
  # Transform the material properties to the reference grid 
  reference_grid_material_properties₁, reference_grid_material_properties_pml₁ = transform_material_properties_to_reference_domain((𝒫₁,𝒫₁ᴾᴹᴸ), Ω₁, qr₁) # Layer 1  
  reference_grid_material_properties₂, reference_grid_material_properties_pml₂ = transform_material_properties_to_reference_domain((𝒫₂,𝒫₂ᴾᴹᴸ), Ω₂, qr₂) # Layer 2  
  # Compute the bulk terms on the two layers
  bulk_elasticity_operator₁, bulk_elasticity_pml_operator₁ = compute_bulk_elasticity_operators((reference_grid_material_properties₁, reference_grid_material_properties_pml₁)) # Layer 1  
  bulk_elasticity_operator₂, bulk_elasticity_pml_operator₂ = compute_bulk_elasticity_operators((reference_grid_material_properties₂, reference_grid_material_properties_pml₂)) # Layer 2
  # Get the 2d SBP operators and the surface norms on the reference grid on the two domains
  sbp_2d₁ = get_sbp_operators_on_reference_grid(qr₁) # Layer 1  
  sbp_2d₂ = get_sbp_operators_on_reference_grid(qr₂) # Layer 2  
  # The determinant of the Jacobian of transformation
  J₁ = bulk_jacobian(Ω₁, qr₁);  J₁⁻¹ = J₁\(I(size(J₁,1))) # Layer 1
  J₂ = bulk_jacobian(Ω₂, qr₂);  J₂⁻¹ = J₂\(I(size(J₂,1))) # Layer 2
  # Impedance matrices
  𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, (𝛔ᵥ¹, 𝛔ₕ¹), 𝛒₁ = get_pml_elastic_wave_coefficients((Z¹₁₂, σₕσᵥ, ρ₁), Ω₁, qr₁)  # Layer 1
  𝐙₁₂², 𝛔₁₂², 𝛕₁₂², (𝛔ᵥ², 𝛔ₕ²), 𝛒₂ = get_pml_elastic_wave_coefficients((Z²₁₂, σₕσᵥ, ρ₂), Ω₂, qr₂)  # Layer 2
  # Gradient Operators in the physical domain
  Dx₁, Dy₁ = compute_gradient_operators_on_physical_domain(Ω₁, qr₁) # Layer 1
  Dx₂, Dy₂ = compute_gradient_operators_on_physical_domain(Ω₂, qr₂) # Layer 2
  # Surface Jacobian Matrices 
  SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹ =  compute_surface_jacobian_matrices_on_domain(Ω₁, qr₁, J₁⁻¹) # Layer 1  
  SJq₀², SJqₙ², SJr₀², SJrₙ² =  compute_surface_jacobian_matrices_on_domain(Ω₂, qr₂, J₂⁻¹) # Layer 2
  # We build the governing equations on both layer using Kronecker products
  # Equation 1: ∂u/∂t = p
  EQ1₁ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₁)⊗I(n₁))
  EQ1₂ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₂)⊗I(n₂))
  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [δᵢⱼ(2,i,(6,6)) for i=1:6];
  eq2s₁ = [(J₁⁻¹*bulk_elasticity_operator₁)+α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)-𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, 
            -𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹), 
            J₁⁻¹*bulk_elasticity_pml_operator₁[1], 
            J₁⁻¹*bulk_elasticity_pml_operator₁[2], 
            -α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)+𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, 
            𝛒₁*𝛔ᵥ¹*𝛔ₕ¹];
  EQ2₁ = sum(es .⊗ eq2s₁);  
  eq2s₂ = [(J₂⁻¹*bulk_elasticity_operator₂)+α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)-𝛒₂*𝛔ᵥ²*𝛔ₕ², 
            -𝛒₂*(𝛔ᵥ²+𝛔ₕ²), 
            J₂⁻¹*bulk_elasticity_pml_operator₂[1], 
            J₂⁻¹*bulk_elasticity_pml_operator₂[2], 
            -α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)+𝛒₂*𝛔ᵥ²*𝛔ₕ², 
            𝛒₂*𝛔ᵥ²*𝛔ₕ²];  
  EQ2₂ = sum(es .⊗ eq2s₂);
  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [δᵢⱼ(3,i,(6,6)) for i=[1,3]];
  eq3s₁ = [Dx₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ᵥ¹)];
  EQ3₁ = sum(es .⊗ eq3s₁);
  eq3s₂ = [Dx₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ᵥ²)];  
  EQ3₂ = sum(es .⊗ eq3s₂);
  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [δᵢⱼ(4,i,(6,6)) for i=[1,4]]
  eq4s₁ = [Dy₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ₕ¹)]
  eq4s₂ = [Dy₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ₕ²)]
  EQ4₁ = sum(es .⊗ eq4s₁)
  EQ4₂ = sum(es .⊗ eq4s₂)
  # Equation 5 ∂q/∂t = α(u-q)
  es = [δᵢⱼ(5,i,(6,6)) for i=[1,5]]
  eq5s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  EQ5₁ = sum(es .⊗ eq5s₁)
  eq5s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ5₂ = sum(es .⊗ eq5s₂)
  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [δᵢⱼ(6,i,(6,6)) for i=[1,5,6]]
  eq6s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  EQ6₁ = sum(es .⊗ eq6s₁)
  eq6s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ6₂ = sum(es .⊗ eq6s₂)
  #######
  # PML characteristic boundary conditions on the outer boundaries of the two layers
  #######
  # On Layer 1:
  es = [δᵢⱼ(2,i,(6,6)) for i=1:6];
  abc_coeffs₁ = 𝒫₁, 𝒫₁ᴾᴹᴸ, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, J₁;
  χq₀¹, χqₙ¹, _, χrₙ¹ = compute_absorbing_boundary_conditions_on_domain(Ω₁, qr₁, abc_coeffs₁)
  SJ_𝐇q₀⁻¹₁, SJ_𝐇qₙ⁻¹₁, _, SJ_𝐇rₙ⁻¹₁ = compute_surface_integration_operators(sbp_2d₁, (SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹))
  # -- The SAT Terms on the boundary of Layer 1: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₁ = sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹)) + sum(es.⊗(SJ_𝐇rₙ⁻¹₁.*χrₙ¹));
  # On Layer 2:
  abc_coeffs₂ = 𝒫₂, 𝒫₂ᴾᴹᴸ, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², J₂;
  SJ_𝐇q₀⁻¹₂, SJ_𝐇qₙ⁻¹₂, SJ_𝐇r₀⁻¹₂, _ = compute_surface_integration_operators(sbp_2d₂, (SJq₀², SJqₙ², SJr₀², SJrₙ²))
  χq₀², χqₙ², χr₀², _ = compute_absorbing_boundary_conditions_on_domain(Ω₂, qr₂, abc_coeffs₂)
  # -- The SAT Terms on the boundary of Layer 2: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²)) + sum(es.⊗(SJ_𝐇r₀⁻¹₂.*χr₀²));
  #######
  # Imposing the interface continuity condition
  #######
  # Get the jump matrices
  jump₁, jump₂, _ = interface_SAT_operator((Ω₁,qr₁), (Ω₂,qr₂), [0;-1], [0;1]; X = (δᵢⱼ(2,1,(6,6))⊗I(2)))
  jump₁ᵀ, _, 𝐇₁⁻¹, 𝐇₂⁻¹ = interface_SAT_operator((Ω₁,qr₁), (Ω₂,qr₂), [0;-1], [0;1]; X = (δᵢⱼ(1,1,(6,6))⊗I(2)))  
  # Traction on interface From Layer 1
  traction_on_layer_1 = elasticity_traction_operator(𝒫₁, Ω₁, qr₁, [0;-1]).A
  pml_traction_on_layer_1 = elasticity_traction_pml_operator(𝒫₁ᴾᴹᴸ, Ω₁, qr₁, [0;-1]).A   
  # Traction on interface From Layer 2
  traction_on_layer_2 = elasticity_traction_operator(𝒫₂, Ω₂, qr₂, [0;1]).A
  pml_traction_on_layer_2 = elasticity_traction_pml_operator(𝒫₂ᴾᴹᴸ, Ω₂, qr₂, [0;1]).A   
  # Assemble the traction on the two layers
  es = [δᵢⱼ(1,i,(6,6)) for i=[1,3,4]]; 
  total_traction_on_layer_1 = sum(es .⊗ [traction_on_layer_1, pml_traction_on_layer_1[1], pml_traction_on_layer_1[2]])
  total_traction_on_layer_2 = sum(es .⊗ [traction_on_layer_2, pml_traction_on_layer_2[1], pml_traction_on_layer_2[2]])
  es = [δᵢⱼ(2,i,(6,6)) for i=[1,3,4]]; 
  total_traction_on_layer_1ᵀ = sum(es .⊗ [(traction_on_layer_1)', (pml_traction_on_layer_1[1])', (pml_traction_on_layer_1[2])'])  
  total_traction_on_layer_2ᵀ = sum(es .⊗ [(traction_on_layer_2)', (pml_traction_on_layer_2[1])', (pml_traction_on_layer_2[2])'])
  interface_traction = blockdiag(total_traction_on_layer_1, total_traction_on_layer_2)      
  interface_tractionᵀ = blockdiag(total_traction_on_layer_1ᵀ, total_traction_on_layer_2ᵀ)   
  h = norm(Ω₁(qr₁[1,2]) - Ω₂(qr₁[1,1]))
  ζ₀ = 400/h  
  # Assemble the interface SAT
  inverse_jacobian = blockdiag(δᵢⱼ(2,2,(6,6))⊗J₁⁻¹, δᵢⱼ(2,2,(6,6))⊗J₂⁻¹)
  interface_jump_terms = (0.5*jump₁*interface_traction - 0.5*interface_tractionᵀ*jump₁ᵀ - ζ₀*jump₂)
  SATᵢ = blockdiag(I(12)⊗𝐇₁⁻¹, I(12)⊗𝐇₂⁻¹)*inverse_jacobian*interface_jump_terms # Interface SAT

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂));  # All the bulk equations
  SATₙ = blockdiag(SAT₁, SAT₂); # Neumann boundary SAT
  bulk - SATᵢ - SATₙ
end

"""
Inverse of the mass matrix
"""
function two_layer_elasticity_pml_mass_matrix(domains::NTuple{2, domain_2d}, reference_grids::NTuple{2, AbstractMatrix{SVector{2,Float64}}}, ρ)
  ρ₁, ρ₂ = ρ
  domain₁, domain₂ = domains
  qr₁, qr₂ = reference_grids
  n₁, m₁ = size(qr₁)
  n₂, m₂ = size(qr₂)
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Ω₁(qr) = transfinite_interpolation(qr, domain₁);
  Ω₂(qr) = transfinite_interpolation(qr, domain₂);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁.(Ω₁.(qr₁))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂.(Ω₂.(qr₂))))
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁, Id₁), blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂, Id₂))
end 

"""
The RK4 scheme
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
𝐔(x) = @SVector [exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2)), exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

N = 21;
Ω₁(qr) = transfinite_interpolation(qr, domain₁);
Ω₂(qr) = transfinite_interpolation(qr, domain₂);
qr₁ = reference_grid_2d((round(Int64, 1.1*N - 0.1),N));
qr₂ = reference_grid_2d((round(Int64, 1.1*N - 0.1),N));
xy₁ = Ω₁.(qr₁);
xy₂ = Ω₂.(qr₂);

𝒫 = 𝒫₁, 𝒫₂
𝒫ᴾᴹᴸ = 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ
Z₁₂ = (Z₁¹, Z₂¹), (Z₁², Z₂²)
σₕσᵥ = σₕ, σᵥ
ρ = ρ₁, ρ₂
stima = two_layer_elasticity_pml_stiffness_matrix((domain₁,domain₂), (qr₁,qr₂), (𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α));
massma = two_layer_elasticity_pml_mass_matrix((domain₁,domain₂), (qr₁,qr₂), (ρ₁, ρ₂));

# Define the time stepping
const Δt = 0.2*norm(xy₁[1,1] - xy₁[1,2])/sqrt(max(cp₁, cp₂)^2 + max(cs₁,cs₂)^2)
tf = 1.0
ntime = ceil(Int, tf/Δt)
maxvals = zeros(Float64, ntime)

plt3 = Vector{Plots.Plot}(undef,3);

# Begin time loop
let
  t = 0.0
  X₀¹ = vcat(eltocols(vec(𝐔.(xy₁))), eltocols(vec(𝐏.(xy₁))), eltocols(vec(𝐕.(xy₁))), eltocols(vec(𝐖.(xy₁))), eltocols(vec(𝐐.(xy₁))), eltocols(vec(𝐑.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔.(xy₂))), eltocols(vec(𝐏.(xy₂))), eltocols(vec(𝐕.(xy₂))), eltocols(vec(𝐖.(xy₂))), eltocols(vec(𝐐.(xy₂))), eltocols(vec(𝐑.(xy₂))));
  X₀ = vcat(X₀¹, X₀²)
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  M = massma*stima
  count = 1;
  # @gif for i=1:ntime
  Hq = SBP4_1D(round(Int64,1.1*N - 0.1)).norm;
  Hr = SBP4_1D(N).norm;
  Hqr = Hq ⊗ Hr
  # @gif for i=1:ntime
  for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    X₀ = RK4_1!(M, sol, Δt)    
    t += Δt    
    (i%30==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))

    u1ref₁,u2ref₁ = split_solution(X₀[1:12*(prod(size(qr₁)))], size(qr₁), 12);
    u1ref₂,u2ref₂ = split_solution(X₀[12*(prod(size(qr₁)))+1:12*(prod(size(qr₁)))+12*(prod(size(qr₁)))], size(qr₁), 12);
    U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
    U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)
    
    if((i==ceil(Int64, 1/Δt)) || (i == ceil(Int64, 2/Δt)) || (i == ceil(Int64, 5/Δt)))
      plt3[count] = Plots.contourf(getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₂), getY.(xy₂), reshape(U2,size(xy₂)...), colormap=:jet)
      Plots.vline!(plt3[count], [Lᵥ], label="\$ x \\ge "*string(round(Lᵥ, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.plot!(plt3[count], getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="Interface", lc=:red, lw=2, size=(400,500), legend=:none)
      xlims!(plt3[count], (0,Lᵥ+δ))
      ylims!(plt3[count], (-Lₕ,Lₕ))
      xlabel!(plt3[count], "\$x\$")
      ylabel!(plt3[count], "\$y\$")
      count += 1
    end

    plt3_gif = Plots.contourf(getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
    Plots.contourf!(plt3_gif, getX.(xy₂), getY.(xy₂), reshape(U2,size(xy₂)...), colormap=:jet)
    Plots.vline!(plt3_gif, [Lᵥ], label="\$ x \\ge "*string(round(Lᵥ, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
    Plots.plot!(plt3_gif, getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="Interface", lc=:red, lw=2, size=(400,500), legend=:none)
    xlims!(plt3_gif, (0,Lᵥ+δ))
    ylims!(plt3_gif, (-Lₕ,Lₕ))
    xlabel!(plt3_gif, "\$x\$")
    ylabel!(plt3_gif, "\$y\$")

    maxvals[i] = sqrt(u1ref₁'*Hqr*u1ref₁ + u2ref₁'*Hqr*u2ref₁ + u1ref₂'*Hqr*u1ref₂ + u2ref₂'*Hqr*u2ref₂)
  end
  # end every 15
  global Xref = X₀
end  

u1ref₁,u2ref₁ = split_solution(Xref[1:12*(prod(size(qr₁)))], size(qr₁), 12);
u1ref₂,u2ref₂ = split_solution(Xref[12*(prod(size(qr₁)))+1:12*(prod(size(qr₁)))+12*(prod(size(qr₁)))], size(qr₁), 12);
U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)

plt3_1 = Plots.plot();
Plots.contourf!(plt3_1, getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₂), getY.(xy₂), reshape(U2, size(xy₂)...), colormap=:jet)
Plots.vline!(plt3_1, [Lᵥ], label="\$ x \\ge "*string(round(Lᵥ, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.plot!(plt3_1, getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="Interface", lc=:red, lw=2, size=(400,500), legend=:none)
xlims!(plt3_1, (0,Lᵥ+δ))
ylims!(plt3_1, (-Lₕ,Lₕ))
xlabel!(plt3_1, "\$x\$")
ylabel!(plt3_1, "\$y\$")
# c_ticks = (LinRange(2.5e-6,1.0e-5,5), string.(round.(LinRange(1.01,7.01,5), digits=4)).*"\$ \\times 10^{-7}\$");
# Plots.plot!(plt3_1, colorbar_ticks=c_ticks)

plt4 = Plots.scatter(vec(Tuple.(xy₁)), mc=:red, msw=0.01, ms=4, label="")
Plots.scatter!(vec(Tuple.(xy₂)), mc=:blue, msw=0.01, ms=4, label="", size=(400,500))
Plots.plot!(getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="", lc=:green, lw=1, size=(400,500))
xlims!(plt4, (0-0.4π, 4π+0.8π))
ylims!(plt4, (-4π-0.8π, 4π+0.8π))
xlabel!(plt4, "\$ x \$")
ylabel!(plt4, "\$ y \$")

plt5 = Plots.plot(LinRange(0,tf,ntime), maxvals, label="", lw=2, yaxis=:log10)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| \\bf{u} \\|_{H} \$")
Plots.xlims!(plt5, (0,tf))


plt6 = Plots.plot([0, 0], [Lᵥ, -Lᵥ], lw=2, lc=:black, label="")
Plots.plot!(plt6, [0, Lₕ+δ], [Lᵥ, Lᵥ], lw=2, lc=:black, label="")
Plots.plot!(plt6, [Lₕ, Lₕ], [Lᵥ, -Lᵥ], lw=1, lc=:black, label="", ls=:dash)
Plots.plot!(plt6, [Lₕ+δ, Lₕ+δ], [Lᵥ, -Lᵥ], lw=2, lc=:black, label="")
Plots.plot!(plt6, [0, Lₕ+δ], [-Lᵥ, -Lᵥ], lw=2, lc=:black, label="")
Plots.plot!(plt6, getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), lc=:red, ls=:dash, lw=1, label="Curvilinear Interface", size=(400,500))
xlims!(plt6, (0-0.4π, 4π+0.8π))
ylims!(plt6, (-4π-0.8π, 4π+2.2π))
Plots.annotate!(plt6, 2.2π, 2π, ("\$ \\Omega_1 \$", 15, :black))
Plots.annotate!(plt6, 2.2π, -2π, ("\$ \\Omega_2 \$", 15, :black))
xlabel!(plt6, "\$ x \$")
ylabel!(plt6, "\$ y \$")

plt7 = Plots.contourf(getX.(xy₁), getY.(xy₁), σᵥ.(xy₁) .+ 1e-16, colormap=:jet)
Plots.contourf!(plt7, getX.(xy₂), getY.(xy₂), σᵥ.(xy₂) .+ 1e-16, colormap=:jet)
Plots.plot!(plt7, [Lₕ, Lₕ], [Lᵥ, -Lᵥ], lw=2, lc=:black, label="")
Plots.plot!(plt7, getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), lc=:red, ls=:dash, lw=1, label="Curvilinear Interface", size=(400,500))
xlims!(plt7, (0-0.4π, 4π+0.8π))
ylims!(plt7, (-4π-0.8π, 4π+2.2π))
Plots.annotate!(plt7, 2.2π, 2π, ("\$ \\Omega_1 \$", 15, :white))
Plots.annotate!(plt7, 2.2π, -2π, ("\$ \\Omega_2 \$", 15, :white))
xlabel!(plt7, "\$ x \$")
ylabel!(plt7, "\$ y \$")