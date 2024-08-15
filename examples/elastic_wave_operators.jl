"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');
eltocols(v::Vector{MVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Function to transform the material properties to the reference domain.
"""
function transform_material_properties_to_reference_domain(props, domain, reference_coords)
  𝒫, 𝒫ᴾᴹᴸ = props  
  reference_grid_material_properties = transform_material_properties.(𝒫, domain, reference_coords)
  reference_grid_material_properties_pml = transform_material_properties_pml.(𝒫ᴾᴹᴸ, domain, reference_coords)
  reference_grid_material_properties, reference_grid_material_properties_pml
end

"""
Function to compute the bulk elasticity operator on the reference grid from the material property functions.
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
Function to compute the gradient operators on the physical domain.
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
Function to calculate the 2d SBP operators on the reference grid from the 1d SBP operators
"""
function get_sbp_operators_on_reference_grid(reference_grid)
  n, m = size(reference_grid)
  sbp_q = SBP4_1D(m)
  sbp_r = SBP4_1D(n)
  sbp_2d = SBP4_2D(sbp_q, sbp_r) 
  sbp_2d
end

"""
Function to obtain the norm inverse on the four boundaries after building the 2d SBP operators
"""
function get_sbp_norm_2d(sbp_2d::SBP4_2D)
  sbp_2d.norm
end

"""
Function to compute the coefficients of the LHS of the PML-modified elastic wave equation.
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
Function to compute the surface Jacobian on the boundaries.
"""
function compute_surface_jacobian_matrices_on_domain(domain, reference_coords, J⁻¹)
  (J⁻¹*surface_jacobian(domain, reference_coords, [-1,0];  X=I(2)), J⁻¹*surface_jacobian(domain, reference_coords, [1,0];  X=I(2)), 
   J⁻¹*surface_jacobian(domain, reference_coords, [0,-1];  X=I(2)), J⁻¹*surface_jacobian(domain, reference_coords, [0,1];  X=I(2)))
end

"""
Function to compute the absorbing boundary conditions on the domain.
"""
function compute_absorbing_boundary_conditions_on_domain(domain, reference_coords, coeffs)
  (elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [-1,0]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [1,0]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [0,-1]).A, 
   elasticity_absorbing_boundary_pml_operator(coeffs, domain, reference_coords, [0,1]).A)
end

"""
Function to compute the surface integration operator for the SAT terms.
"""
function compute_surface_integration_operators(sbp_2d::SBP4_2D, surface_jacobian_matrices::NTuple{4, AbstractMatrix{Float64}})
  SJq₀, SJqₙ, SJr₀, SJrₙ = surface_jacobian_matrices
  𝐇q₀⁻¹, 𝐇qₙ⁻¹, 𝐇r₀⁻¹, 𝐇rₙ⁻¹ = get_sbp_norm_2d(sbp_2d) 
  I₂ = I(2)
  (fill(SJq₀*(I₂⊗𝐇q₀⁻¹), 6), fill(SJqₙ*(I₂⊗𝐇qₙ⁻¹), 6),
   fill(SJr₀*(I₂⊗𝐇r₀⁻¹), 6), fill(SJrₙ*(I₂⊗𝐇rₙ⁻¹), 6))
end

"""
Function to obtain the displacements from the raw solution vector.
"""
function split_solution(X, MN, P)    
  res = splitdimsview(reshape(X, (prod(MN), P)))
  u1, u2 = res[1:2]
  (u1,u2)
end