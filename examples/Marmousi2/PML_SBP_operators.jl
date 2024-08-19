"""
Function to obtain the PML stiffness matrix for the two-layered Marmousi model (-3.34 < y < -0.44964 and 0 < x < 16.9864).
- We impose the absorbing boundary conditions on all the outer boundaries, except at the topmost layer (y = -0.44964).
- At the topmost layer, we impose traction-free boundary conditions
- At the interface between the layers, we impose continuity of traction and displacements.
"""
function marmousi_two_layer_elasticity_pml_stiffness_matrix(domains::NTuple{2, domain_2d}, reference_grids::NTuple{2, AbstractMatrix{SVector{2,Float64}}}, material_properties, ζ₀::Float64)
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

  ##### ##### ##### ##### ##### ##### ##### ##### 
  # Traction-free boundary on the top layer 
  ##### ##### ##### ##### ##### ##### ##### ##### 
  # On Layer 1:
  SJ_𝐇q₀⁻¹₁, SJ_𝐇qₙ⁻¹₁, _, SJ_𝐇rₙ⁻¹₁ = compute_surface_integration_operators(sbp_2d₁, (SJq₀¹, SJqₙ¹, SJr₀¹, SJrₙ¹))
  es = [δᵢⱼ(2,i,(6,6)) for i=[1,3,4]];
  elastic_traction_on_top = elasticity_traction_operator(𝒫₁, Ω₁, qr₁, [0;1]).A
  pml_elastic_traction_on_top₁, pml_elastic_traction_on_top₂ = elasticity_traction_pml_operator(𝒫₁ᴾᴹᴸ, Ω₁, qr₁, [0;1]).A
  Trₙ¹ = [elastic_traction_on_top, pml_elastic_traction_on_top₁, pml_elastic_traction_on_top₂]
  SAT₁ = sum(es.⊗(SJ_𝐇rₙ⁻¹₁[1:3].*Trₙ¹)); 
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # PML characteristic boundary conditions on the outer boundaries of the two layers
  ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
  # On Layer 1:
  es = [δᵢⱼ(2,i,(6,6)) for i=1:6];
  abc_coeffs₁ = 𝒫₁, 𝒫₁ᴾᴹᴸ, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, J₁
  χq₀¹, χqₙ¹, _, _ = compute_absorbing_boundary_conditions_on_domain(Ω₁, qr₁, abc_coeffs₁)
  # -- The SAT Terms on the boundary of Layer 1: Obtained after summing up the boundary integral of the absorbing boundary condition
  SAT₁ += sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹))
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
  # h = norm(Ω₁(qr₁[1,2]) - Ω₁(qr₁[1,1]))
  # ζ₀ = 400/h  
  # Assemble the interface SAT
  inverse_jacobian = blockdiag(δᵢⱼ(2,2,(6,6))⊗J₁⁻¹, δᵢⱼ(2,2,(6,6))⊗J₂⁻¹)
  interface_jump_terms = (0.5*jump₁*interface_traction - 0.5*interface_tractionᵀ*jump₁ᵀ - ζ₀*jump₂)
  SATᵢ = blockdiag(I(12)⊗𝐇₁⁻¹, I(12)⊗𝐇₂⁻¹)*inverse_jacobian*interface_jump_terms # Interface SAT

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂));  # All the bulk equations
  SATₙ = blockdiag(SAT₁, SAT₂); # Neumann boundary SAT
  bulk - SATᵢ - SATₙ
end