#########################################################################
# File containing functions used to implement the 3-layer PML functions #
#########################################################################

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Function to obtain the PML stiffness matrix
"""
function 𝐊2ₚₘₗ(𝒫, 𝒫ᴾᴹᴸ, Z₁₂, 𝛒, 𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫)
  # Extract domains
  𝛀₁, 𝛀₂ = 𝛀
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);  

  𝐪𝐫₁, 𝐪𝐫₂ = 𝐪𝐫

  # Extract the material property functions
  # (Z₁¹, Z₂¹), (Z₁², Z₂²) = Z₁₂
  Z¹₁₂, Z²₁₂ = Z₁₂
  Z₁¹, Z₂¹ = Z¹₁₂
  Z₁², Z₂² = Z²₁₂  

  # 𝒫₁, 𝒫₂ = 𝒫
  # 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ = 𝒫ᴾᴹᴸ
  Pqr₁, Pqr₂ = 𝒫
  Pᴾᴹᴸqr₁, Pᴾᴹᴸqr₂ = 𝒫ᴾᴹᴸ

  # Get the bulk terms for layer 1
  # Pqr₁ = P2R.(𝒫₁,Ω₁,𝐪𝐫₁);
  # Pᴾᴹᴸqr₁ = P2Rᴾᴹᴸ.(𝒫₁ᴾᴹᴸ, Ω₁, 𝐪𝐫₁);  
  𝐏₁ = Pᴱ(Pqr₁).A;
  𝐏₁ᴾᴹᴸ₁, 𝐏₁ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₁).A;

  # Get the bulk terms for layer 2
  # Pqr₂ = P2R.(𝒫₂,Ω₂,𝐪𝐫₂);
  # Pᴾᴹᴸqr₂ = P2Rᴾᴹᴸ.(𝒫₂ᴾᴹᴸ, Ω₂, 𝐪𝐫₂);  
  𝐏₂ = Pᴱ(Pqr₂).A;
  𝐏₂ᴾᴹᴸ₁, 𝐏₂ᴾᴹᴸ₂ = Pᴾᴹᴸ(Pᴾᴹᴸqr₂).A;

  ρ₁, ρ₂ = 𝛒

  # Get the 2d SBP operators on the reference grid on all 3 layers
  # Layer 1
  m₁, n₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀⁻¹₁, 𝐇qₙ⁻¹₁, _, 𝐇rₙ⁻¹₁ = sbp_2d₁.norm
  Dq₁, Dr₁ = sbp_2d₁.D1
  Dqr₁ = [I(2)⊗Dq₁, I(2)⊗Dr₁]
  # Layer 2
  m₂, n₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀⁻¹₂, 𝐇qₙ⁻¹₂, 𝐇r₀⁻¹₂, _ = sbp_2d₂.norm
  Dq₂, Dr₂ = sbp_2d₂.D1
  Dqr₂ = [I(2)⊗Dq₂, I(2)⊗Dr₂]

  # Obtain some quantities on the grid points on Layer 1
  # Bulk Jacobian
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫₁)
  𝐉₁⁻¹ = 𝐉₁\(I(size(𝐉₁,1))) 
  # Impedance matrices
  𝐙₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j], Z₂¹[i,j] ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:m₁, j=1:n₁], 2)
  𝛔₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j]*σₕ(Ω₁(𝐪𝐫₁[i,j])), Z₂¹[i,j]*σᵥ(Ω₁(𝐪𝐫₁[i,j])) ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:m₁, j=1:n₁], 2)
  𝛕₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j]*σₕ(Ω₁(𝐪𝐫₁[i,j]))*σᵥ(Ω₁(𝐪𝐫₁[i,j])), Z₂¹[i,j]*σᵥ(Ω₁(𝐪𝐫₁[i,j]))*σₕ(Ω₁(𝐪𝐫₁[i,j])) ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:m₁, j=1:n₁], 2)  
  𝛔ᵥ¹ = I(2) ⊗ spdiagm(σᵥ.(Ω₁.(vec(𝐪𝐫₁))));  
  𝛔ₕ¹ = I(2) ⊗ spdiagm(σₕ.(Ω₁.(vec(𝐪𝐫₁))));
  𝛒₁ = I(2) ⊗ spdiagm(vec(ρ₁))
  # Get the transformed gradient
  Jqr₁ = J⁻¹.(𝐪𝐫₁, Ω₁);
  J_vec₁ = get_property_matrix_on_grid(Jqr₁, 2);
  J_vec_diag₁ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₁];
  Dx₁, Dy₁ = J_vec_diag₁*Dqr₁; 

  # Obtain some quantities on the grid points on Layer 2
  # Bulk Jacobian
  𝐉₂ = Jb(𝛀₂, 𝐪𝐫₂)
  𝐉₂⁻¹ = 𝐉₂\(I(size(𝐉₂,1))) 
  # # Impedance matrices  
  𝐙₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j], Z₂²[i,j] ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:m₂, j=1:n₂], 2)
  𝛔₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j]*σₕ(Ω₂(𝐪𝐫₂[i,j])), Z₂²[i,j]*σᵥ(Ω₂(𝐪𝐫₂[i,j])) ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:m₂, j=1:n₂], 2)
  𝛕₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j]*σₕ(Ω₂(𝐪𝐫₂[i,j]))*σᵥ(Ω₂(𝐪𝐫₂[i,j])), Z₂²[i,j]*σᵥ(Ω₂(𝐪𝐫₂[i,j]))*σₕ(Ω₂(𝐪𝐫₂[i,j])) ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:m₂, j=1:n₂], 2) 
  𝛔ᵥ² = I(2) ⊗ spdiagm(σᵥ.(Ω₂.(vec(𝐪𝐫₂))));  
  𝛔ₕ² = I(2) ⊗ spdiagm(σₕ.(Ω₂.(vec(𝐪𝐫₂))));
  𝛒₂ = I(2) ⊗ spdiagm(vec(ρ₂))
  # Get the transformed gradient
  Jqr₂ = J⁻¹.(𝐪𝐫₂, Ω₂);
  J_vec₂ = get_property_matrix_on_grid(Jqr₂, 2);
  J_vec_diag₂ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₂];
  Dx₂, Dy₂ = J_vec_diag₂*Dqr₂;

  # Surface Jacobian Matrices on Layer 1
  _, SJq₀¹, SJrₙ¹, SJqₙ¹ =  𝐉₁⁻¹*Js(𝛀₁, [0,-1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [-1,0];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [0,1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 2
  SJr₀², SJq₀², _, SJqₙ² =  𝐉₂⁻¹*Js(𝛀₂, [0,-1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [-1,0];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [0,1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [1,0];  X=I(2))

  # We build the governing equations on both layer simultaneously
  # Equation 1: ∂u/∂t = p
  EQ1₁ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₁)⊗I(n₁))
  EQ1₂ = E1(1,2,(6,6)) ⊗ (I(2)⊗I(m₂)⊗I(n₂))  
  
  # Equation 2 (Momentum Equation): ρ(∂p/∂t) = ∇⋅(σ(u)) + σᴾᴹᴸ - ρ(σᵥ+σₕ)p + ρ(σᵥ+σₕ)α(u-q) - ρ(σᵥσₕ)(u-q-r)
  es = [E1(2,i,(6,6)) for i=1:6];
  eq2s₁ = [(𝐉₁⁻¹*𝐏₁)+α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)-𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, -𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹), 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₁, 𝐉₁⁻¹*𝐏₁ᴾᴹᴸ₂, -α*𝛒₁*(𝛔ᵥ¹+𝛔ₕ¹)+𝛒₁*𝛔ᵥ¹*𝛔ₕ¹, 𝛒₁*𝛔ᵥ¹*𝛔ₕ¹];
  eq2s₂ = [(𝐉₂⁻¹*𝐏₂)+α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)-𝛒₂*𝛔ᵥ²*𝛔ₕ², -𝛒₂*(𝛔ᵥ²+𝛔ₕ²), 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₁, 𝐉₂⁻¹*𝐏₂ᴾᴹᴸ₂, -α*𝛒₂*(𝛔ᵥ²+𝛔ₕ²)+𝛒₂*𝛔ᵥ²*𝛔ₕ², 𝛒₂*𝛔ᵥ²*𝛔ₕ²];  
  EQ2₁ = sum(es .⊗ eq2s₁);  
  EQ2₂ = sum(es .⊗ eq2s₂);  

  # Equation 3: ∂v/∂t = -(α+σᵥ)v + ∂u/∂x
  es = [E1(3,i,(6,6)) for i=[1,3]];
  eq3s₁ = [Dx₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ᵥ¹)];
  eq3s₂ = [Dx₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ᵥ²)];  
  EQ3₁ = sum(es .⊗ eq3s₁);
  EQ3₂ = sum(es .⊗ eq3s₂);

  # Equation 4 ∂w/∂t = -(α+σᵥ)w + ∂u/∂y
  es = [E1(4,i,(6,6)) for i=[1,4]]
  eq4s₁ = [Dy₁, -(α*(I(2)⊗I(m₁)⊗I(n₁)) + 𝛔ₕ¹)]
  eq4s₂ = [Dy₂, -(α*(I(2)⊗I(m₂)⊗I(n₂)) + 𝛔ₕ²)]  
  EQ4₁ = sum(es .⊗ eq4s₁)
  EQ4₂ = sum(es .⊗ eq4s₂)

  # Equation 5 ∂q/∂t = α(u-q)
  es = [E1(5,i,(6,6)) for i=[1,5]]
  eq5s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  eq5s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ5₁ = sum(es .⊗ eq5s₁)
  EQ5₂ = sum(es .⊗ eq5s₂)  

  # Equation 6 ∂q/∂t = α(u-q-r)
  es = [E1(6,i,(6,6)) for i=[1,5,6]]
  eq6s₁ = [α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁)), -α*(I(2)⊗I(m₁)⊗I(n₁))]
  eq6s₂ = [α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂)), -α*(I(2)⊗I(m₂)⊗I(n₂))]  
  EQ6₁ = sum(es .⊗ eq6s₁)
  EQ6₂ = sum(es .⊗ eq6s₂)  
  display("Done building the bulk equations. Applying boundary conditions.")

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
  
  # Characteristic boundary conditions on the outer boundaries
  PQRᵪ² = Pqr₂, Pᴾᴹᴸqr₂, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², 𝐉₂;
  χr₀², χq₀², χqₙ² = χᴾᴹᴸ(PQRᵪ², 𝛀₂, [0,-1]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [-1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [1,0]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇r₀⁻¹₂ = (fill(SJr₀²,6).*fill((I(2)⊗𝐇r₀⁻¹₂),6));
  SJ_𝐇q₀⁻¹₂ = (fill(SJq₀²,6).*fill((I(2)⊗𝐇q₀⁻¹₂),6));
  SJ_𝐇qₙ⁻¹₂ = (fill(SJqₙ²,6).*fill((I(2)⊗𝐇qₙ⁻¹₂),6));  
  SAT₂ = sum(es.⊗(SJ_𝐇r₀⁻¹₂.*χr₀²)) + sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²));

  # The interface part
  Eᵢ¹ = E1(2,1,(6,6)) ⊗ I(2)
  Eᵢ² = E1(1,1,(6,6)) ⊗ I(2)
  
  # Get the jump matrices
  # Layer 1-2
  B̂₁₂,  B̃₁₂, _ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂₁₂ᵀ, _, 𝐇₁⁻¹₁, 𝐇₂⁻¹₁ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)

  # Traction on interface From Layer 1
  Tr₀¹ = Tᴱ(Pqr₁, 𝛀₁, [0;-1]).A
  Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁ = Tᴾᴹᴸ(Pᴾᴹᴸqr₁, 𝛀₁, [0;-1]).A  
  # Tractions on interface From Layer 2
  # 1)
  Trₙ² = Tᴱ(Pqr₂, 𝛀₂, [0;1]).A
  Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂ = Tᴾᴹᴸ(Pᴾᴹᴸqr₂, 𝛀₂, [0;1]).A  

  # Assemble the traction on the two layers
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓r₀¹ = sum(es .⊗ [Tr₀¹, Tr₀ᴾᴹᴸ₁₁, Tr₀ᴾᴹᴸ₂₁])
  es = [E1(1,i,(6,6)) for i=[1,3,4]]; 𝐓rₙ² = sum(es .⊗ [Trₙ², Trₙᴾᴹᴸ₁₂, Trₙᴾᴹᴸ₂₂])  

  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀ₀¹ = sum(es .⊗ [(Tr₀¹)', (Tr₀ᴾᴹᴸ₁₁)', (Tr₀ᴾᴹᴸ₂₁)'])  
  es = [E1(2,i,(6,6)) for i=[1,3,4]]; 𝐓rᵀₙ² = sum(es .⊗ [(Trₙ²)', (Trₙᴾᴹᴸ₁₂)', (Trₙᴾᴹᴸ₂₂)'])  

  𝐓rᵢ¹² = blockdiag(𝐓r₀¹, 𝐓rₙ²)      
  𝐓rᵢᵀ₁₂ = blockdiag(𝐓rᵀ₀¹, 𝐓rᵀₙ²)     

  h = norm(Ω₂(𝐪𝐫₂[end,1]) - Ω₂(𝐪𝐫₂[end-1,1]))
  ζ₀ = 300/h  
  # Assemble the interface SAT
  𝐉₁₂ = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₁⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₂⁻¹)  
  SATᵢ¹² = blockdiag(I(12)⊗𝐇₁⁻¹₁, I(12)⊗𝐇₂⁻¹₁)*𝐉₁₂*(0.5*B̂₁₂*𝐓rᵢ¹² - 0.5*𝐓rᵢᵀ₁₂*B̂₁₂ᵀ - ζ₀*B̃₁₂)  

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), 
                   (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂));  
  SATₙ = blockdiag(SAT₁, SAT₂)
  display("Done building the LHS.")
  bulk - SATᵢ¹² - SATₙ;
end

"""
Inverse of the mass matrix
"""
function 𝐌2⁻¹ₚₘₗ(𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫, 𝛒)
  ρ₁, ρ₂ = 𝛒
  𝛀₁, 𝛀₂ = 𝛀  
  m₁, n₁ = 𝛀₁.mn; m₂, n₂ = 𝛀₂.mn  
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁));  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Ω₁(qr) = S(qr, 𝛀₁.domain); Ω₂(qr) = S(qr, 𝛀₂.domain);  
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁)); ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂))  
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁, Id₁), 
            blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂, Id₂))
end 

"""
A non-allocating implementation of the RK4 scheme
"""
function RK4_1!(Δt, M, sol)  
  X₀, k₁, k₂, k₃, k₄ = sol  
  k₁ .= M*(X₀)
  k₂ .= M*(X₀ + 0.5*Δt*k₁)
  k₃ .= M*(X₀ + 0.5*Δt*k₂)
  k₄ .= M*(X₀ + Δt*k₃)
  X₀ .+= (Δt/6)*(k₁ + 2*k₂ + 2*k₃ + k₄)
end

"""
Right hand side function
"""
function f(t::Float64, x::SVector{2,Float64}, params)
  s₁, s₂, M₀, pos_x, pos_y = params
  @assert length(pos_x) == length(pos_y)
  res = @SVector [0.0, 0.0]
  for i=1:lastindex(pos_x)
    res += @SVector[-1/(2π*√(s₁*s₂))*exp(-(x[1]-pos_x[i]*(16.9864))^2/(2s₁) - (x[2]-(pos_y[i])*(-3.4972))^2/(2s₂))*(x[1]-pos_x[i]*(16.9864))/s₁*exp(-(t-0.215)^2/0.15)*M₀,
                    -1/(2π*√(s₁*s₂))*exp(-(x[1]-pos_x[i]*(16.9864))^2/(2s₁) - (x[2]-(pos_y[i])*(-3.4972))^2/(2s₂))*(x[2]-pos_y[i]*(-3.4972))/s₂*exp(-(t-0.215)^2/0.15)*M₀]
  end
  res
end

"""
A non-allocating implementation of the RK4 scheme with forcing
"""
function RK4_1!(MK, sol, Δt, F, M)  
  X₀, k₁, k₂, k₃, k₄ = sol
  F₁, F₂, F₄ = F
  k₁ .= MK*(X₀) + M*F₁
  k₂ .= MK*(X₀ + 0.5*Δt*k₁) + M*F₂
  k₃ .= MK*(X₀ + 0.5*Δt*k₂) + M*F₂
  k₄ .= MK*(X₀ + Δt*k₃) + M*F₄
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
Functions to get the X and Y grids
"""
getX(A) = A[1]
getY(A) = A[2]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##
#   Transform material properties to reference grid    #
##### ##### ##### ##### ##### ##### ##### ##### ##### ##
"""
Function to transform to the reference coordinates
1) The BULK Terms
2) The PML Terms
3) The Impedance Matrices
"""
function Pt(𝒫, 𝒮, qr)    
  invJ = J⁻¹(qr, 𝒮)
  detJ = (det∘J)(qr, 𝒮)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫*S)*detJ
end
function Ptᴾᴹᴸ(𝒫, 𝒮, qr)
  invJ = J⁻¹(qr, 𝒮)
  detJ = (det∘J)(qr, 𝒮)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(detJ*S'*𝒫)
end
function 𝐙_t(𝒫, Ω, qr)
  𝒫₁, 𝒫₂ = 𝒫
  𝐉⁻¹ = J⁻¹(qr, Ω) ⊗ I(size(𝒫₁,1))
  𝐏 = (E1(1,1,(2,2)) ⊗ 𝒫₁) + (E1(2,2,(2,2)) ⊗ 𝒫₂)  
  𝐉⁻¹*𝐏  
end