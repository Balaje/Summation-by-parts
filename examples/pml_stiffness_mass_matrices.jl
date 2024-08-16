"""
Function to obtain the PML stiffness matrix for two-layered medium.
- We impose the absorbing boundary conditions on all the outer boundaries.
- At the interface between the layers, we impose continuity of traction and displacements.
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

  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # Compute and transform the PDE to the reference domain
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

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

  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # We build the governing equations on both layers using Kronecker products
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
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

  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # PML characteristic boundary conditions on the outer boundaries of the two layers
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # On Layer 1:
  es = [δᵢⱼ(2,i,(6,6)) for i=1:6];
  abc_coeffs₁ = 𝒫₁, 𝒫₁ᴾᴹᴸ, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, J₁
  χq₀¹, χqₙ¹, _, χrₙ¹ = compute_absorbing_boundary_conditions_on_domain(Ω₁, qr₁, abc_coeffs₁)
  SJ_𝐇q₀⁻¹₁, SJ_𝐇qₙ⁻¹₁, _, SJ_𝐇rₙ⁻¹₁ = compute_surface_integration_operators(sbp_2d₁, (SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹))
  # -- The SAT Terms on the boundary of Layer 1: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₁ = sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹)) + sum(es.⊗(SJ_𝐇rₙ⁻¹₁.*χrₙ¹))
  # On Layer 2:
  abc_coeffs₂ = 𝒫₂, 𝒫₂ᴾᴹᴸ, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², J₂;
  χq₀², χqₙ², χr₀², _ = compute_absorbing_boundary_conditions_on_domain(Ω₂, qr₂, abc_coeffs₂)
  SJ_𝐇q₀⁻¹₂, SJ_𝐇qₙ⁻¹₂, SJ_𝐇r₀⁻¹₂, _ = compute_surface_integration_operators(sbp_2d₂, (SJq₀², SJqₙ², SJr₀², SJrₙ²))
  # -- The SAT Terms on the boundary of Layer 2: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²)) + sum(es.⊗(SJ_𝐇r₀⁻¹₂.*χr₀²))

  ##### ##### ##### ##### ##### ##### ##### ##### 
  # Imposing the interface continuity condition
  ##### ##### ##### ##### ##### ##### ##### ##### 
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
  h = norm(Ω₁(qr₁[1,2]) - Ω₁(qr₁[1,1]))
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
Function to compute the inverse of the mass matrix corresponding to the two-layer problem
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
Function to compute the stifness and mass matrices for the 4-layer case:
1) Here we use the traction-free boundary at the top.
2) And use absorbing boundaries on the left, right and bottom boundaries.
3) At the interfaces, we enforce traction and displacement continuities.
"""
function four_layer_elasticity_pml_stiffness_matrix(domains::NTuple{4, domain_2d}, reference_grids::NTuple{4, AbstractMatrix{SVector{2,Float64}}}, material_properties)
  # Extract domain
  domain₁, domain₂, domain₃, domain₄ = domains
  Ω₁(qr) = transfinite_interpolation(qr, domain₁)
  Ω₂(qr) = transfinite_interpolation(qr, domain₂)
  Ω₃(qr) = transfinite_interpolation(qr, domain₃)
  Ω₄(qr) = transfinite_interpolation(qr, domain₄)
  qr₁, qr₂, qr₃, qr₄ = reference_grids  
  𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α = material_properties
  # Extract the material property functions
  # (Z₁¹, Z₂¹), (Z₁², Z₂²) = Z₁₂
  Z¹₁₂, Z²₁₂, Z³₁₂, Z⁴₁₂ = Z₁₂
  # Extract the elastic material tensors
  𝒫₁, 𝒫₂, 𝒫₃, 𝒫₄ = 𝒫
  𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ, 𝒫₃ᴾᴹᴸ, 𝒫₄ᴾᴹᴸ = 𝒫ᴾᴹᴸ
  # Extract the PML damping functions
  # σₕ, σᵥ = σₕσᵥ
  # Extract the density of the materials
  ρ₁, ρ₂, ρ₃, ρ₄ = ρ
  # Get the discretization 
  n₁, m₁ = size(qr₁)
  n₂, m₂ = size(qr₂)
  n₃, m₃ = size(qr₃)
  n₄, m₄ = size(qr₄)

  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # Compute and transform the PDE to the reference domain
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

  # Transform the material properties to the reference grid 
  reference_grid_material_properties₁, reference_grid_material_properties_pml₁ = transform_material_properties_to_reference_domain((𝒫₁,𝒫₁ᴾᴹᴸ), Ω₁, qr₁) # Layer 1  
  reference_grid_material_properties₂, reference_grid_material_properties_pml₂ = transform_material_properties_to_reference_domain((𝒫₂,𝒫₂ᴾᴹᴸ), Ω₂, qr₂) # Layer 2  
  reference_grid_material_properties₃, reference_grid_material_properties_pml₃ = transform_material_properties_to_reference_domain((𝒫₃,𝒫₃ᴾᴹᴸ), Ω₃, qr₃) # Layer 3  
  reference_grid_material_properties₄, reference_grid_material_properties_pml₄ = transform_material_properties_to_reference_domain((𝒫₄,𝒫₄ᴾᴹᴸ), Ω₄, qr₄) # Layer 4  

  # Compute the bulk terms on the two layers
  bulk_elasticity_operator₁, bulk_elasticity_pml_operator₁ = compute_bulk_elasticity_operators((reference_grid_material_properties₁, reference_grid_material_properties_pml₁)) # Layer 1  
  bulk_elasticity_operator₂, bulk_elasticity_pml_operator₂ = compute_bulk_elasticity_operators((reference_grid_material_properties₂, reference_grid_material_properties_pml₂)) # Layer 2
  bulk_elasticity_operator₃, bulk_elasticity_pml_operator₃ = compute_bulk_elasticity_operators((reference_grid_material_properties₃, reference_grid_material_properties_pml₃)) # Layer 3
  bulk_elasticity_operator₄, bulk_elasticity_pml_operator₄ = compute_bulk_elasticity_operators((reference_grid_material_properties₄, reference_grid_material_properties_pml₄)) # Layer 4

  # Get the 2d SBP operators and the surface norms on the reference grid on the two domains
  sbp_2d₁ = get_sbp_operators_on_reference_grid(qr₁) # Layer 1  
  sbp_2d₂ = get_sbp_operators_on_reference_grid(qr₂) # Layer 2  
  sbp_2d₃ = get_sbp_operators_on_reference_grid(qr₃) # Layer 3  
  sbp_2d₄ = get_sbp_operators_on_reference_grid(qr₄) # Layer 4  
  # The determinant of the Jacobian of transformation
  J₁ = bulk_jacobian(Ω₁, qr₁);  J₁⁻¹ = J₁\(I(size(J₁,1))) # Layer 1
  J₂ = bulk_jacobian(Ω₂, qr₂);  J₂⁻¹ = J₂\(I(size(J₂,1))) # Layer 2
  J₃ = bulk_jacobian(Ω₃, qr₃);  J₃⁻¹ = J₃\(I(size(J₃,1))) # Layer 3
  J₄ = bulk_jacobian(Ω₄, qr₄);  J₄⁻¹ = J₄\(I(size(J₄,1))) # Layer 4
  # Impedance matrices
  𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, (𝛔ᵥ¹, 𝛔ₕ¹), 𝛒₁ = get_pml_elastic_wave_coefficients((Z¹₁₂, σₕσᵥ, ρ₁), Ω₁, qr₁)  # Layer 1
  𝐙₁₂², 𝛔₁₂², 𝛕₁₂², (𝛔ᵥ², 𝛔ₕ²), 𝛒₂ = get_pml_elastic_wave_coefficients((Z²₁₂, σₕσᵥ, ρ₂), Ω₂, qr₂)  # Layer 2
  𝐙₁₂³, 𝛔₁₂³, 𝛕₁₂³, (𝛔ᵥ³, 𝛔ₕ³), 𝛒₃ = get_pml_elastic_wave_coefficients((Z³₁₂, σₕσᵥ, ρ₃), Ω₃, qr₃)  # Layer 3
  𝐙₁₂⁴, 𝛔₁₂⁴, 𝛕₁₂⁴, (𝛔ᵥ⁴, 𝛔ₕ⁴), 𝛒₄ = get_pml_elastic_wave_coefficients((Z⁴₁₂, σₕσᵥ, ρ₄), Ω₄, qr₄)  # Layer 4
  # Gradient Operators in the physical domain
  Dx₁, Dy₁ = compute_gradient_operators_on_physical_domain(Ω₁, qr₁) # Layer 1
  Dx₂, Dy₂ = compute_gradient_operators_on_physical_domain(Ω₂, qr₂) # Layer 2
  Dx₃, Dy₃ = compute_gradient_operators_on_physical_domain(Ω₃, qr₃) # Layer 3
  Dx₄, Dy₄ = compute_gradient_operators_on_physical_domain(Ω₄, qr₄) # Layer 4
  # Surface Jacobian Matrices 
  SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹ =  compute_surface_jacobian_matrices_on_domain(Ω₁, qr₁, J₁⁻¹) # Layer 1  
  SJq₀², SJqₙ², SJr₀², SJrₙ² =  compute_surface_jacobian_matrices_on_domain(Ω₂, qr₂, J₂⁻¹) # Layer 2
  SJq₀³, SJqₙ³, SJr₀³, SJrₙ³ =  compute_surface_jacobian_matrices_on_domain(Ω₃, qr₃, J₃⁻¹) # Layer 3
  SJq₀⁴, SJqₙ⁴, SJr₀⁴, SJrₙ⁴ =  compute_surface_jacobian_matrices_on_domain(Ω₄, qr₄, J₄⁻¹) # Layer 4


  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # We build the governing equations on both layers using Kronecker products
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # Equation 1: ∂u/∂t = p
  EQ1₁ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₁)⊗I(n₁))
  EQ1₂ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₂)⊗I(n₂))
  EQ1₃ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₃)⊗I(n₃))
  EQ1₄ = δᵢⱼ(1,2,(6,6)) ⊗ (I(2)⊗I(m₄)⊗I(n₄))
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
  eq2s₃ = [(J₃⁻¹*bulk_elasticity_operator₃)+α*𝛒₃*(𝛔ᵥ³+𝛔ₕ³)-𝛒₃*𝛔ᵥ³*𝛔ₕ³, 
            -𝛒₃*(𝛔ᵥ³+𝛔ₕ³), 
            J₃⁻¹*bulk_elasticity_pml_operator₃[1], 
            J₃⁻¹*bulk_elasticity_pml_operator₃[2], 
            -α*𝛒₃*(𝛔ᵥ³+𝛔ₕ³)+𝛒₃*𝛔ᵥ³*𝛔ₕ³, 
            𝛒₃*𝛔ᵥ³*𝛔ₕ³];  
  EQ2₃ = sum(es .⊗ eq2s₃);
  eq2s₄ = [(J₄⁻¹*bulk_elasticity_operator₄)+α*𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴)-𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴, 
            -𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴), 
            J₄⁻¹*bulk_elasticity_pml_operator₄[1], 
            J₄⁻¹*bulk_elasticity_pml_operator₄[2], 
            -α*𝛒₄*(𝛔ᵥ⁴+𝛔ₕ⁴)+𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴, 
            𝛒₄*𝛔ᵥ⁴*𝛔ₕ⁴];  
  EQ2₄ = sum(es .⊗ eq2s₄);
  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [δᵢⱼ(3,i,(6,6)) for i=[1,3]];
  eq3s₁ = [Dx₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ᵥ¹)];
  EQ3₁ = sum(es .⊗ eq3s₁);
  eq3s₂ = [Dx₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ᵥ²)];  
  EQ3₂ = sum(es .⊗ eq3s₂);
  eq3s₃ = [Dx₃, -(α*(I(2)⊗I(m₃)⊗I(n₃)) + 𝛔ᵥ³)];  
  EQ3₃ = sum(es .⊗ eq3s₃);
  eq3s₄ = [Dx₄, -(α*(I(2)⊗I(m₄)⊗I(n₄)) + 𝛔ᵥ⁴)];  
  EQ3₄ = sum(es .⊗ eq3s₄);
  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [δᵢⱼ(4,i,(6,6)) for i=[1,4]]
  eq4s₁ = [Dy₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ₕ¹)]
  eq4s₂ = [Dy₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ₕ²)]
  eq4s₃ = [Dy₃, -(α*(I(2)⊗I(m₃)⊗I(n₃)) + 𝛔ₕ³)]
  eq4s₄ = [Dy₄, -(α*(I(2)⊗I(m₄)⊗I(n₄)) + 𝛔ₕ⁴)]
  EQ4₁ = sum(es .⊗ eq4s₁)
  EQ4₂ = sum(es .⊗ eq4s₂)
  EQ4₃ = sum(es .⊗ eq4s₃)
  EQ4₄ = sum(es .⊗ eq4s₄)
  # Equation 5 ∂q/∂t = α(u-q)
  es = [δᵢⱼ(5,i,(6,6)) for i=[1,5]]
  eq5s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  EQ5₁ = sum(es .⊗ eq5s₁)
  eq5s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ5₂ = sum(es .⊗ eq5s₂)
  eq5s₃ = [α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃))]  
  EQ5₃ = sum(es .⊗ eq5s₃)
  eq5s₄ = [α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄))]  
  EQ5₄ = sum(es .⊗ eq5s₄)
  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [δᵢⱼ(6,i,(6,6)) for i=[1,5,6]]
  eq6s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  EQ6₁ = sum(es .⊗ eq6s₁)
  eq6s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ6₂ = sum(es .⊗ eq6s₂)
  eq6s₃ = [α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃)), -α*(I(2)⊗I(m₃)⊗I(n₃))]  
  EQ6₃ = sum(es .⊗ eq6s₃)
  eq6s₄ = [α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄)), -α*(I(2)⊗I(m₄)⊗I(n₄))]  
  EQ6₄ = sum(es .⊗ eq6s₄)

  ##### ##### ##### ##### ##### ##### ##### ##### 
  # Traction-free boundary on the top layer 
  ##### ##### ##### ##### ##### ##### ##### ##### 
  # On Layer 1:
  SJ_𝐇q₀⁻¹₁, SJ_𝐇qₙ⁻¹₁, _, SJ_𝐇rₙ⁻¹₁ = compute_surface_integration_operators(sbp_2d₁, (SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹))
  es = [δᵢⱼ(2,i,(6,6)) for i=[1,3,4]];
  elastic_traction_on_top = elasticity_traction_operator(𝒫₁, Ω₁, qr₁, [0;1]).A
  pml_elastic_traction_on_top₁, pml_elastic_traction_on_top₂ = elasticity_traction_operator(𝒫₁ᴾᴹᴸ, Ω₁, qr₁, [0;1]).A
  Trₙ¹ = [elastic_traction_on_top, pml_elastic_traction_on_top₁, pml_elastic_traction_on_top₂]
  SAT₁ = sum(es.⊗(SJ_𝐇rₙ⁻¹₁[1:3].*Trₙ¹)); 
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # PML characteristic boundary conditions on the left and right boundaries of the two layers
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  es = [δᵢⱼ(2,i,(6,6)) for i=1:6];
  χq₀¹, χqₙ¹, _, _ = compute_absorbing_boundary_conditions_on_domain(Ω₁, qr₁, abc_coeffs₁)  
  # -- The SAT Terms on the boundary of Layer 1: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₁ += sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹))
  # On Layer 2:
  abc_coeffs₂ = 𝒫₂, 𝒫₂ᴾᴹᴸ, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², J₂;
  χq₀², χqₙ², _, _ = compute_absorbing_boundary_conditions_on_domain(Ω₂, qr₂, abc_coeffs₂)
  SJ_𝐇q₀⁻¹₂, SJ_𝐇qₙ⁻¹₂, _, _ = compute_surface_integration_operators(sbp_2d₂, (SJq₀², SJqₙ², SJr₀², SJrₙ²))
  # -- The SAT Terms on the boundary of Layer 2: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²)) 
  # On Layer 3:
  abc_coeffs₃ = 𝒫₃, 𝒫₃ᴾᴹᴸ, 𝐙₁₂³ , 𝛔₁₂³, 𝛕₁₂³, J₃;
  χq₀³, χqₙ³, _, _ = compute_absorbing_boundary_conditions_on_domain(Ω₃, qr₃, abc_coeffs₃)
  SJ_𝐇q₀⁻¹₃, SJ_𝐇qₙ⁻¹₃, _, _ = compute_surface_integration_operators(sbp_2d₃, (SJq₀³, SJqₙ³, SJr₀³, SJrₙ³))
  # -- The SAT Terms on the boundary of Layer 3: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₃ = sum(es.⊗(SJ_𝐇q₀⁻¹₃.*χq₀³)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₃.*χqₙ³)) 
  # On Layer 4:
  abc_coeffs₄ = 𝒫₄, 𝒫₄ᴾᴹᴸ, 𝐙₁₂⁴, 𝛔₁₂⁴, 𝛕₁₂⁴, J₄;
  χq₀⁴, χqₙ⁴, χr₀⁴, _ = compute_absorbing_boundary_conditions_on_domain(Ω₄, qr₄, abc_coeffs₄)
  SJ_𝐇q₀⁻¹₄, SJ_𝐇qₙ⁻¹₄, SJ_𝐇r₀⁻¹₄, _ = compute_surface_integration_operators(sbp_2d₄, (SJq₀⁴, SJqₙ⁴, SJr₀⁴, SJrₙ⁴))
  # -- The SAT Terms on the boundary of Layer 2: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₄ = sum(es.⊗(SJ_𝐇q₀⁻¹₄.*χq₀⁴)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₄.*χqₙ⁴)) + sum(es.⊗(SJ_𝐇r₀⁻¹₄.*χr₀⁴))

  ##### ##### ##### ##### ##### ##### ##### ##### 
  # Imposing the interface continuity conditions
  ##### ##### ##### ##### ##### ##### ##### ##### 
  # Get the jump matrices
  layer_12_jump₁, layer_12_jump₂, _ = interface_SAT_operator((Ω₁,qr₁), (Ω₂,qr₂), [0;-1], [0;1]; X = (δᵢⱼ(2,1,(6,6))⊗I(2)))
  layer_12_jump₁ᵀ, _, layer_12_𝐇₁⁻¹, layer_12_𝐇₂⁻¹ = interface_SAT_operator((Ω₁,qr₁), (Ω₂,qr₂), [0;-1], [0;1]; X = (δᵢⱼ(1,1,(6,6))⊗I(2)))  
  layer_23_jump₁, layer_23_jump₂, _ = interface_SAT_operator((Ω₂,qr₂), (Ω₃,qr₃), [0;-1], [0;1]; X = (δᵢⱼ(2,1,(6,6))⊗I(2)))
  layer_23_jump₁ᵀ, _, layer_23_𝐇₁⁻¹, layer_23_𝐇₂⁻¹ = interface_SAT_operator((Ω₂,qr₂), (Ω₃,qr₃), [0;-1], [0;1]; X = (δᵢⱼ(1,1,(6,6))⊗I(2)))  
  layer_34_jump₁, layer_34_jump₂, _ = interface_SAT_operator((Ω₃,qr₃), (Ω₄,qr₄), [0;-1], [0;1]; X = (δᵢⱼ(2,1,(6,6))⊗I(2)))
  layer_34_jump₁ᵀ, _, layer_34_𝐇₁⁻¹, layer_34_𝐇₂⁻¹ = interface_SAT_operator((Ω₃,qr₃), (Ω₄,qr₄), [0;-1], [0;1]; X = (δᵢⱼ(1,1,(6,6))⊗I(2)))  
  # Traction on interface From Layer 1
  traction_on_layer_1 = elasticity_traction_operator(𝒫₁, Ω₁, qr₁, [0;-1]).A
  pml_traction_on_layer_1 = elasticity_traction_pml_operator(𝒫₁ᴾᴹᴸ, Ω₁, qr₁, [0;-1]).A   
  # Traction on interfaces From Layer 2
  traction_on_layer_2_top = elasticity_traction_operator(𝒫₂, Ω₂, qr₂, [0;1]).A
  pml_traction_on_layer_2_top = elasticity_traction_pml_operator(𝒫₂ᴾᴹᴸ, Ω₂, qr₂, [0;1]).A   
  traction_on_layer_2_bottom = elasticity_traction_operator(𝒫₂, Ω₂, qr₂, [0;-1]).A
  pml_traction_on_layer_2_bottom = elasticity_traction_pml_operator(𝒫₂ᴾᴹᴸ, Ω₂, qr₂, [0;-1]).A
  # Traction on interfaces From Layer 3
  traction_on_layer_3_top = elasticity_traction_operator(𝒫₃, Ω₃, qr₃, [0;1]).A
  pml_traction_on_layer_3_top = elasticity_traction_pml_operator(𝒫₃ᴾᴹᴸ, Ω₃, qr₃, [0;1]).A   
  traction_on_layer_3_bottom = elasticity_traction_operator(𝒫₃, Ω₃, qr₃, [0;-1]).A
  pml_traction_on_layer_3_bottom = elasticity_traction_pml_operator(𝒫₃ᴾᴹᴸ, Ω₃, qr₃, [0;-1]).A  
  # Traction on interface From Layer 4
  traction_on_layer_4 = elasticity_traction_operator(𝒫₄, Ω₄, qr₄, [0;1]).A
  pml_traction_on_layer_4 = elasticity_traction_pml_operator(𝒫₄ᴾᴹᴸ, Ω₄, qr₄, [0;1]).A   
  # Assemble the traction on the two layers
  es = [δᵢⱼ(1,i,(6,6)) for i=[1,3,4]]; 
  total_traction_on_layer_1 = sum(es .⊗ [traction_on_layer_1, pml_traction_on_layer_1[1], pml_traction_on_layer_1[2]])
  total_traction_on_layer_2_top = sum(es .⊗ [traction_on_layer_2_top, pml_traction_on_layer_2_top[1], pml_traction_on_layer_2_top[2]])
  total_traction_on_layer_2_bottom = sum(es .⊗ [traction_on_layer_2_bottom, pml_traction_on_layer_2_bottom[1], pml_traction_on_layer_2_bottom[2]])
  total_traction_on_layer_3_top = sum(es .⊗ [traction_on_layer_3_top, pml_traction_on_layer_3_top[1], pml_traction_on_layer_3_top[2]])
  total_traction_on_layer_3_bottom = sum(es .⊗ [traction_on_layer_3_bottom, pml_traction_on_layer_3_bottom[1], pml_traction_on_layer_3_bottom[2]])
  total_traction_on_layer_4 = sum(es .⊗ [traction_on_layer_4, pml_traction_on_layer_4[1], pml_traction_on_layer_4[2]])
  es = [δᵢⱼ(2,i,(6,6)) for i=[1,3,4]]; 
  total_traction_on_layer_1ᵀ = sum(es .⊗ [(traction_on_layer_1)', (pml_traction_on_layer_1[1])', (pml_traction_on_layer_1[2])'])  
  total_traction_on_layer_2_topᵀ = sum(es .⊗ [(traction_on_layer_2_top)', (pml_traction_on_layer_2_top[1])', (pml_traction_on_layer_2_top[2])'])
  total_traction_on_layer_2_bottomᵀ = sum(es .⊗ [(traction_on_layer_2_bottom)', (pml_traction_on_layer_2_bottom[1])', (pml_traction_on_layer_2_bottom[2])'])
  total_traction_on_layer_3_topᵀ = sum(es .⊗ [(traction_on_layer_3_top)', (pml_traction_on_layer_3_top[1])', (pml_traction_on_layer_3_top[2])'])
  total_traction_on_layer_3_bottomᵀ = sum(es .⊗ [(traction_on_layer_3_bottom)', (pml_traction_on_layer_3_bottom[1])', (pml_traction_on_layer_3_bottom[2])'])
  total_traction_on_layer_4ᵀ = sum(es .⊗ [(traction_on_layer_4)', (pml_traction_on_layer_4[1])', (pml_traction_on_layer_4[2])'])  

  interface_traction_1 = blockdiag(total_traction_on_layer_1, total_traction_on_layer_2_top)      
  interface_traction_1ᵀ = blockdiag(total_traction_on_layer_1ᵀ, total_traction_on_layer_2_topᵀ)   
  interface_traction_2 = blockdiag(total_traction_on_layer_2_bottom, total_traction_on_layer_3_top)      
  interface_traction_2ᵀ = blockdiag(total_traction_on_layer_2_bottomᵀ, total_traction_on_layer_3_topᵀ)      
  interface_traction_3 = blockdiag(total_traction_on_layer_3_bottom, total_traction_on_layer_4)      
  interface_traction_3ᵀ = blockdiag(total_traction_on_layer_3_bottomᵀ, total_traction_on_layer_4ᵀ)      
  h = norm(Ω₁(qr₁[1,2]) - Ω₁(qr₁[1,1]))  
  ζ₀ = 30*5.196/h  
  # Assemble the interface SAT
  inverse_jacobian_12 = blockdiag(δᵢⱼ(2,2,(6,6))⊗J₁⁻¹, δᵢⱼ(2,2,(6,6))⊗J₂⁻¹)
  inverse_jacobian_23 = blockdiag(δᵢⱼ(2,2,(6,6))⊗J₂⁻¹, δᵢⱼ(2,2,(6,6))⊗J₃⁻¹)
  inverse_jacobian_34 = blockdiag(δᵢⱼ(2,2,(6,6))⊗J₃⁻¹, δᵢⱼ(2,2,(6,6))⊗J₄⁻¹)
  interface_jump_terms_12 = (0.5*layer_12_jump₁*interface_traction_1 - 0.5*interface_traction_1ᵀ*layer_12_jump₁ᵀ - ζ₀*layer_12_jump₂)
  interface_jump_terms_23 = (0.5*layer_23_jump₁*interface_traction_2 - 0.5*interface_traction_2ᵀ*layer_23_jump₁ᵀ - ζ₀*layer_23_jump₂)
  interface_jump_terms_34 = (0.5*layer_34_jump₁*interface_traction_3 - 0.5*interface_traction_3ᵀ*layer_34_jump₁ᵀ - ζ₀*layer_34_jump₂)
  layer_12_SATᵢ = blockdiag(blockdiag(I(12)⊗layer_12_𝐇₁⁻¹, I(12)⊗layer_12_𝐇₂⁻¹)*inverse_jacobian_12*interface_jump_terms_12, zero(EQ1₃), zero(EQ1₄)) # Interface SAT
  layer_23_SATᵢ = blockdiag(zero(EQ1₁), blockdiag(I(12)⊗layer_23_𝐇₁⁻¹, I(12)⊗layer_23_𝐇₂⁻¹)*inverse_jacobian_23*interface_jump_terms_23, zero(EQ1₄)) # Interface SAT
  layer_34_SATᵢ = blockdiag(zero(EQ1₁), zero(EQ1₂), blockdiag(I(12)⊗layer_34_𝐇₁⁻¹, I(12)⊗layer_34_𝐇₂⁻¹)*inverse_jacobian_34*interface_jump_terms_34) # Interface SAT

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), 
                   (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂),
                   (EQ1₃ + EQ2₃ + EQ3₃ + EQ4₃ + EQ5₃ + EQ6₃),
                   (EQ1₄ + EQ2₄ + EQ3₄ + EQ4₄ + EQ5₄ + EQ6₄));  # All the bulk equations
  SATₙ = blockdiag(SAT₁, SAT₂, SAT₃, SAT₄); # Neumann boundary SAT
  bulk - layer_12_SATᵢ - layer_23_SATᵢ - layer_34_SATᵢ - SATₙ
end

"""
Function to compute the inverse of the 4-layer PML mass matrix
"""
function four_layer_elasticity_pml_mass_matrix(domains::NTuple{4, domain_2d}, reference_grids::NTuple{4, AbstractMatrix{SVector{2,Float64}}}, ρ)
  ρ₁, ρ₂, ρ₃, ρ₄ = ρ
  domain₁, domain₂, domain₃, domain₄ = domains
  qr₁, qr₂, qr₃, qr₄ = reference_grids
  n₁, m₁ = size(qr₁)
  n₂, m₂ = size(qr₂)
  n₃, m₃ = size(qr₃)
  n₄, m₄ = size(qr₄)
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Id₃ = sparse(I(2)⊗I(m₃)⊗I(n₃))
  Id₄ = sparse(I(2)⊗I(m₄)⊗I(n₄))
  Ω₁(qr) = transfinite_interpolation(qr, domain₁);
  Ω₂(qr) = transfinite_interpolation(qr, domain₂);
  Ω₃(qr) = transfinite_interpolation(qr, domain₃);
  Ω₄(qr) = transfinite_interpolation(qr, domain₄);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁.(Ω₁.(qr₁))))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂.(Ω₂.(qr₂))))
  ρᵥ³ = I(2)⊗spdiagm(vec(1 ./ρ₃.(Ω₃.(qr₃))))
  ρᵥ⁴ = I(2)⊗spdiagm(vec(1 ./ρ₄.(Ω₄.(qr₄))))
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁, Id₁), 
            blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂, Id₂),
            blockdiag(Id₃, ρᵥ³, Id₃, Id₃, Id₃, Id₃),
            blockdiag(Id₄, ρᵥ⁴, Id₄, Id₄, Id₄, Id₄))
end 