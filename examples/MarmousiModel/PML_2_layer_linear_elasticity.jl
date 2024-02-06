# include("2d_elasticity_problem.jl");
using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Plots

using SplitApplyCombine
using LoopVectorization

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

  # Get the 2d SBP operators on the reference grid
  n₁, m₁ = size(𝐪𝐫₁)
  sbp_q₁ = SBP_1_2_CONSTANT_0_1(m₁)
  sbp_r₁ = SBP_1_2_CONSTANT_0_1(n₁)
  sbp_2d₁ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₁, sbp_r₁)
  𝐇q₀⁻¹₁, 𝐇qₙ⁻¹₁, 𝐇r₀⁻¹₁, 𝐇rₙ⁻¹₁ = sbp_2d₁.norm
  Dq₁, Dr₁ = sbp_2d₁.D1
  Dqr₁ = [I(2)⊗Dq₁, I(2)⊗Dr₁]
  n₂, m₂ = size(𝐪𝐫₂)
  sbp_q₂ = SBP_1_2_CONSTANT_0_1(m₂)
  sbp_r₂ = SBP_1_2_CONSTANT_0_1(n₂)
  sbp_2d₂ = SBP_1_2_CONSTANT_0_1_0_1(sbp_q₂, sbp_r₂)
  𝐇q₀⁻¹₂, 𝐇qₙ⁻¹₂, 𝐇r₀⁻¹₂, 𝐇rₙ⁻¹₂ = sbp_2d₂.norm
  Dq₂, Dr₂ = sbp_2d₂.D1
  Dqr₂ = [I(2)⊗Dq₂, I(2)⊗Dr₂]

  # Obtain some quantities on the grid points on Layer 1
  # Bulk Jacobian
  𝐉₁ = Jb(𝛀₁, 𝐪𝐫₁)
  𝐉₁⁻¹ = 𝐉₁\(I(size(𝐉₁,1))) 
  # Impedance matrices
  # 𝐙₁₂¹ = 𝐙((Z₁¹,Z₂¹), Ω₁, 𝐪𝐫₁);
  # 𝛔₁₂¹ = 𝐙((x->σₕ(x)*Z₁¹(x), x->σᵥ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫₁)
  # 𝛕₁₂¹ = 𝐙((x->σₕ(x)*σᵥ(x)*Z₁¹(x), x->σₕ(x)*σᵥ(x)*Z₂¹(x)), Ω₁, 𝐪𝐫₁)
  𝐙₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j], Z₂¹[i,j] ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:n₁, j=1:m₁], 2)
  𝛔₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j]*σₕ(Ω₁(𝐪𝐫₁[i,j])), Z₂¹[i,j]*σᵥ(Ω₁(𝐪𝐫₁[i,j])) ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:n₁, j=1:m₁], 2)
  𝛕₁₂¹ = get_property_matrix_on_grid([𝐙_t(( Z₁¹[i,j]*σₕ(Ω₁(𝐪𝐫₁[i,j]))*σᵥ(Ω₁(𝐪𝐫₁[i,j])), Z₂¹[i,j]*σᵥ(Ω₁(𝐪𝐫₁[i,j]))*σₕ(Ω₁(𝐪𝐫₁[i,j])) ), Ω₁, 𝐪𝐫₁[i,j]) for i=1:n₁, j=1:m₁], 2)  
  𝛔ᵥ¹ = I(2) ⊗ spdiagm(σᵥ.(Ω₁.(vec(𝐪𝐫₁))));  𝛔ₕ¹ = I(2) ⊗ spdiagm(σₕ.(Ω₁.(vec(𝐪𝐫₁))));
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
  # 𝐙₁₂² = 𝐙((Z₁²,Z₂²), Ω₂, 𝐪𝐫₂);
  # 𝛔₁₂² = 𝐙((x->σₕ(x)*Z₁²(x), x->σᵥ(x)*Z₂²(x)), Ω₂, 𝐪𝐫₂)
  # 𝛕₁₂² = 𝐙((x->σᵥ(x)*σₕ(x)*Z₁²(x), x->σᵥ(x)*σₕ(x)*Z₂²(x)), Ω₂, 𝐪𝐫₂)  
  𝐙₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j], Z₂²[i,j] ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:n₂, j=1:m₂], 2)
  𝛔₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j]*σₕ(Ω₂(𝐪𝐫₂[i,j])), Z₂²[i,j]*σᵥ(Ω₂(𝐪𝐫₂[i,j])) ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:n₂, j=1:m₂], 2)
  𝛕₁₂² = get_property_matrix_on_grid([𝐙_t(( Z₁²[i,j]*σₕ(Ω₂(𝐪𝐫₂[i,j]))*σᵥ(Ω₂(𝐪𝐫₂[i,j])), Z₂²[i,j]*σᵥ(Ω₂(𝐪𝐫₂[i,j]))*σₕ(Ω₂(𝐪𝐫₂[i,j])) ), Ω₂, 𝐪𝐫₂[i,j]) for i=1:n₂, j=1:m₂], 2) 
  𝛔ᵥ² = I(2) ⊗ spdiagm(σᵥ.(Ω₂.(vec(𝐪𝐫₂))));  𝛔ₕ² = I(2) ⊗ spdiagm(σₕ.(Ω₂.(vec(𝐪𝐫₂))));
  𝛒₂ = I(2) ⊗ spdiagm(vec(ρ₂))
  # Get the transformed gradient
  Jqr₂ = J⁻¹.(𝐪𝐫₂, Ω₂);
  J_vec₂ = get_property_matrix_on_grid(Jqr₂, 2);
  J_vec_diag₂ = [I(2)⊗spdiagm(vec(p)) for p in J_vec₂];
  Dx₂, Dy₂ = J_vec_diag₂*Dqr₂;

  # Surface Jacobian Matrices on Layer 1
  SJr₀¹, SJq₀¹, SJrₙ¹, SJqₙ¹ =  𝐉₁⁻¹*Js(𝛀₁, [0,-1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [-1,0];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [0,1];  X=I(2)), 𝐉₁⁻¹*Js(𝛀₁, [1,0];  X=I(2))
  # Surface Jacobian Matrices on Layer 2
  SJr₀², SJq₀², SJrₙ², SJqₙ² =  𝐉₂⁻¹*Js(𝛀₂, [0,-1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [-1,0];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [0,1];  X=I(2)), 𝐉₂⁻¹*Js(𝛀₂, [1,0];  X=I(2))

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

  # PML characteristic boundary conditions
  es = [E1(2,i,(6,6)) for i=1:6];
  PQRᵪ¹ = Pqr₁, Pᴾᴹᴸqr₁, 𝐙₁₂¹, 𝛔₁₂¹, 𝛕₁₂¹, 𝐉₁;
  χq₀¹, χr₀¹, χqₙ¹, χrₙ¹ = χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [-1,0]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [0,-1]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [1,0]).A, χᴾᴹᴸ(PQRᵪ¹, 𝛀₁, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₁ = (fill(SJq₀¹,6).*fill((I(2)⊗𝐇q₀⁻¹₁),6));
  SJ_𝐇qₙ⁻¹₁ = (fill(SJqₙ¹,6).*fill((I(2)⊗𝐇qₙ⁻¹₁),6));
  SJ_𝐇r₀⁻¹₁ = (fill(SJr₀¹,6).*fill((I(2)⊗𝐇r₀⁻¹₁),6));
  SJ_𝐇rₙ⁻¹₁ = (fill(SJrₙ¹,6).*fill((I(2)⊗𝐇rₙ⁻¹₁),6));
  SAT₁ = sum(es.⊗(SJ_𝐇q₀⁻¹₁.*χq₀¹)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₁.*χqₙ¹)) + sum(es.⊗(SJ_𝐇rₙ⁻¹₁.*χrₙ¹));
  
  PQRᵪ² = Pqr₂, Pᴾᴹᴸqr₂, 𝐙₁₂², 𝛔₁₂², 𝛕₁₂², 𝐉₂;
  χq₀², χr₀², χqₙ², χrₙ² = χᴾᴹᴸ(PQRᵪ², 𝛀₂, [-1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [0,-1]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [1,0]).A, χᴾᴹᴸ(PQRᵪ², 𝛀₂, [0,1]).A;
  # The SAT Terms on the boundary 
  SJ_𝐇q₀⁻¹₂ = (fill(SJq₀²,6).*fill((I(2)⊗𝐇q₀⁻¹₂),6));
  SJ_𝐇qₙ⁻¹₂ = (fill(SJqₙ²,6).*fill((I(2)⊗𝐇qₙ⁻¹₂),6));
  SJ_𝐇r₀⁻¹₂ = (fill(SJr₀²,6).*fill((I(2)⊗𝐇r₀⁻¹₂),6));
  SJ_𝐇rₙ⁻¹₂ = (fill(SJrₙ²,6).*fill((I(2)⊗𝐇rₙ⁻¹₂),6));
  SAT₂ = sum(es.⊗(SJ_𝐇q₀⁻¹₂.*χq₀²)) + sum(es.⊗(SJ_𝐇qₙ⁻¹₂.*χqₙ²)) + sum(es.⊗(SJ_𝐇r₀⁻¹₂.*χr₀²));

  # The interface part
  Eᵢ¹ = E1(2,1,(6,6)) ⊗ I(2)
  Eᵢ² = E1(1,1,(6,6)) ⊗ I(2)
  # Get the jump matrices
  B̂,  B̃, _ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ¹)
  B̂ᵀ, _, 𝐇₁⁻¹, 𝐇₂⁻¹ = SATᵢᴱ(𝛀₁, 𝛀₂, [0; -1], [0; 1], ConformingInterface(); X=Eᵢ²)
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
  h = 3/(max(m₁,n₁,m₂,n₂)-1)
  ζ₀ = 200/h  
  # Assemble the interface SAT
  𝐉 = blockdiag(E1(2,2,(6,6)) ⊗ 𝐉₁⁻¹, E1(2,2,(6,6)) ⊗ 𝐉₂⁻¹)
  SATᵢ = blockdiag(I(12)⊗𝐇₁⁻¹, I(12)⊗𝐇₂⁻¹)*𝐉*(0.5*B̂*𝐓rᵢ - 0.5*𝐓rᵢᵀ*B̂ᵀ - ζ₀*B̃)

  # The SBP-SAT Formulation
  bulk = blockdiag((EQ1₁ + EQ2₁ + EQ3₁ + EQ4₁ + EQ5₁ + EQ6₁), (EQ1₂ + EQ2₂ + EQ3₂ + EQ4₂ + EQ5₂ + EQ6₂));  
  SATₙ = blockdiag(SAT₁, SAT₂)
  bulk - SATᵢ - SATₙ;
end

"""
Inverse of the mass matrix
"""
function 𝐌2⁻¹ₚₘₗ(𝛀::Tuple{DiscreteDomain,DiscreteDomain}, 𝐪𝐫, 𝛒)
  ρ₁, ρ₂ = 𝛒
  𝛀₁, 𝛀₂ = 𝛀
  𝐪𝐫₁, 𝐪𝐫₂ = 𝐪𝐫
  m₁, n₁ = size(𝐪𝐫₁)
  m₂, n₂ = size(𝐪𝐫₂)
  Id₁ = sparse(I(2)⊗I(m₁)⊗I(n₁))
  Id₂ = sparse(I(2)⊗I(m₂)⊗I(n₂))
  Ω₁(qr) = S(qr, 𝛀₁.domain);
  Ω₂(qr) = S(qr, 𝛀₂.domain);
  ρᵥ¹ = I(2)⊗spdiagm(vec(1 ./ρ₁))
  ρᵥ² = I(2)⊗spdiagm(vec(1 ./ρ₂))
  blockdiag(blockdiag(Id₁, ρᵥ¹, Id₁, Id₁, Id₁, Id₁), blockdiag(Id₂, ρᵥ², Id₂, Id₂, Id₂, Id₂))
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
Function to split the solution into the corresponding variables
"""
function split_solution(X, MN, P)    
  res = splitdimsview(reshape(X, (prod(MN), P)))
  u1, u2 = res[1:2]
  (u1,u2)
end

using MAT
vars1 = matread("./examples/MarmousiModel/marmousi2_crop_x_7206_9608_z0_1_1401_downsampled_10.mat");
X₁ = vars1["X_e"]/1000
Z₁ = vars1["Z_e"]/1000
x₁ = X₁[1,:]
z₁ = Z₁[:,1]
n₁, m₁ = size(X₁);
XZ₁ = [[X₁[i,j], Z₁[i,j]] for i=1:n₁, j=1:m₁] # X-Z coordinates from data
## Define the physical domain
c₀¹(r) = @SVector [x₁[1], z₁[1] + (z₁[end]-z₁[1])*r] # Left boundary 
c₁¹(q) = @SVector [x₁[1] + (x₁[end]-x₁[1])*q, z₁[1]] # Bottom boundary
c₂¹(r) = @SVector [x₁[end], z₁[1] + (z₁[end]-z₁[1])*r] # Right boundary
c₃¹(q) = @SVector [x₁[1] + (x₁[end]-x₁[1])*q, z₁[end]] # Top boundary
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
𝛀₁ = DiscreteDomain(domain₁, (m₁,n₁));
Ω₁(qr) = S(qr, 𝛀₁.domain);

vars2 = matread("./examples/MarmousiModel/marmousi2_crop_x_7206_9608_z0_1401_2801_downsampled_10.mat");
X₂ = vars2["X"]/1000
Z₂ = vars2["Z"]/1000
x₂ = X₂[1,:]
z₂ = Z₂[:,1]
n₂, m₂ = size(X₂);
XZ₂ = [[X₂[i,j], Z₂[i,j]] for i=1:n₂, j=1:m₂] # X-Z coordinates from data
## Define the physical domain
c₀²(r) = @SVector [x₂[1], z₂[1] + (z₂[end]-z₂[1])*r] # Left boundary 
c₁²(q) = @SVector [x₂[1] + (x₂[end]-x₂[1])*q, z₂[1]] # Bottom boundary
c₂²(r) = @SVector [x₂[end], z₂[1] + (z₂[end]-z₂[1])*r] # Right boundary
c₃²(q) = @SVector [x₂[1] + (x₂[end]-x₂[1])*q, z₂[end]] # Top boundary
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)
𝛀₂ = DiscreteDomain(domain₂, (m₂,n₂));
Ω₂(qr) = S(qr, 𝛀₂.domain);

𝐪𝐫₁ = generate_2d_grid((m₁,n₁));
𝐪𝐫₂ = generate_2d_grid((m₂,n₂));
using Test
@test Ω₁.(𝐪𝐫₁) ≈ XZ₁
@test Ω₂.(𝐪𝐫₂) ≈ XZ₂

##### ##### ##### ##### ##### ##### ##### ##### 
#   Build the material properties function    #
##### ##### ##### ##### ##### ##### ##### #####
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

# Properties on Layer 1
vp₁ = vars1["vp_e"]/1000;
vs₁ = vars1["vs_e"]/1000;
rho₁ = vars1["rho_e"]/1000;
mu₁ = (vs₁.^2).*rho₁;
lambda₁ = (vp₁.^2).*rho₁ - 2*mu₁;
C₁₁¹ = C₂₂¹ = 2*mu₁ + lambda₁;
C₃₃¹ = mu₁;
C₁₂¹ = lambda₁;
P₁ = [@SMatrix [C₁₁¹[i,j] 0 0 C₁₂¹[i,j]; 0 C₃₃¹[i,j] C₃₃¹[i,j] 0; 0 C₃₃¹[i,j] C₃₃¹[i,j] 0; C₁₂¹[i,j] 0  0 C₂₂¹[i,j]] for i=1:n₁, j=1:m₁]
Z₁¹ = [@SMatrix [sqrt(C₁₁¹[i,j]*rho₁[i,j]) 0; 0 sqrt(C₃₃¹[i,j]*rho₁[i,j])] for i=1:n₁, j=1:m₁]
Z₂¹ = [@SMatrix [sqrt(C₃₃¹[i,j]*rho₁[i,j]) 0; 0 sqrt(C₂₂¹[i,j]*rho₁[i,j])] for i=1:n₁, j=1:m₁]

# Properties on Layer 2
vp₂ = vars2["vp"]/1000;
vs₂ = vars2["vs"]/1000;
rho₂ = vars2["rho"]/1000;
mu₂ = (vs₂.^2).*rho₂;
lambda₂ = (vp₂.^2).*rho₂ - 2*mu₂;
C₁₁² = C₂₂² = 2*mu₂ + lambda₂;
C₃₃² = mu₂;
C₁₂² = lambda₂;
P₂ = [@SMatrix [C₁₁²[i,j] 0 0 C₁₂²[i,j]; 0 C₃₃²[i,j] C₃₃²[i,j] 0; 0 C₃₃²[i,j] C₃₃²[i,j] 0; C₁₂²[i,j] 0  0 C₂₂²[i,j]] for i=1:n₂, j=1:m₂]
Z₁² = [@SMatrix [sqrt(C₁₁²[i,j]*rho₂[i,j]) 0; 0 sqrt(C₃₃²[i,j]*rho₂[i,j])] for i=1:n₂, j=1:m₂]
Z₂² = [@SMatrix [sqrt(C₃₃²[i,j]*rho₂[i,j]) 0; 0 sqrt(C₂₂²[i,j]*rho₂[i,j])] for i=1:n₂, j=1:m₂]

"""
The PML damping
"""
const Lᵥ = abs(z₂[1]-z₁[end])
const Lₕ = x₁[end] - x₁[1]
const δ = 0.1*(Lₕ)
const σ₀ᵛ = 8*(√(max(maximum(vp₁), maximum(vp₂))))/(2*δ)*log(10^3) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*(√(max(maximum(vp₁), maximum(vp₂))))/(2*δ)*log(10^3) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ (x₁[1]+0.9*Lₕ)) || x[1] > (x₁[1]+0.9*Lₕ))
    return σ₀ᵛ*((x[1] - x₁[1] - 0.9*Lₕ)/δ)^3  
  elseif((x[1] ≈ (x₁[1]+0.1*Lₕ)) || x[1] < (x₁[1]+0.1*Lₕ))
    return σ₀ᵛ*((x₁[1] + 0.1*Lₕ - x[1])/δ)^3  
  else
    return 0.0
  end
end

"""
Horizontal PML strip
"""
function σₕ(x)
  0.0
end


Pᴾᴹᴸ₁ = [@SMatrix [C₁₁¹[i,j]*(σₕ(Ω₁(𝐪𝐫₁[i,j])) - σᵥ(Ω₁(𝐪𝐫₁[i,j]))) 0 0 0; 
                   0 C₃₃¹[i,j]*(σₕ(Ω₁(𝐪𝐫₁[i,j])) - σᵥ(Ω₁(𝐪𝐫₁[i,j]))) 0 0; 
                   0 0 C₃₃¹[i,j]*(σᵥ(Ω₁(𝐪𝐫₁[i,j])) - σₕ(Ω₁(𝐪𝐫₁[i,j]))) 0; 
                   0 0 0 C₂₂¹[i,j]*(σᵥ(Ω₁(𝐪𝐫₁[i,j])) - σₕ(Ω₁(𝐪𝐫₁[i,j])))] 
                   for i=1:n₁, j=1:m₁]
Pᴾᴹᴸ₂ = [@SMatrix [C₁₁²[i,j]*(σₕ(Ω₂(𝐪𝐫₂[i,j])) - σᵥ(Ω₂(𝐪𝐫₂[i,j]))) 0 0 0; 
                   0 C₃₃²[i,j]*(σₕ(Ω₂(𝐪𝐫₂[i,j])) - σᵥ(Ω₂(𝐪𝐫₂[i,j]))) 0 0; 
                   0 0 C₃₃²[i,j]*(σᵥ(Ω₂(𝐪𝐫₂[i,j])) - σₕ(Ω₂(𝐪𝐫₂[i,j]))) 0; 
                   0 0 0 C₂₂²[i,j]*(σᵥ(Ω₂(𝐪𝐫₂[i,j])) - σₕ(Ω₂(𝐪𝐫₂[i,j])))] 
                   for i=1:n₂, j=1:m₂]

𝒫₁ = [Pt(P₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:n₁, j=1:m₁];
𝒫₂ = [Pt(P₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:n₂, j=1:m₂];
𝒫ᴾᴹᴸ₁ = [Pt(Pᴾᴹᴸ₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:n₁, j=1:m₁];
𝒫ᴾᴹᴸ₂ = [Pt(Pᴾᴹᴸ₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:n₂, j=1:m₂];

stima = 𝐊2ₚₘₗ((𝒫₁, 𝒫₂), (𝒫ᴾᴹᴸ₁, 𝒫ᴾᴹᴸ₂), ((Z₁¹, Z₂¹), (Z₁², Z₂²)), (rho₁, rho₂), (𝛀₁,𝛀₂), (𝐪𝐫₁,𝐪𝐫₂));
massma =  𝐌2⁻¹ₚₘₗ((𝛀₁, 𝛀₂), (𝐪𝐫₁, 𝐪𝐫₂), (rho₁, rho₂));

𝐔(x) = @SVector [exp(-200*((x[1]-(x₁[end]*0.75+x₁[1]*0.25))^2 + (x[2]-(0.25*z₁[end]+0.75*z₁[1]))^2)) + exp(-200*((x[1]-(x₁[end]*0.25+x₁[1]*0.75))^2 + (x[2]-(0.25*z₂[end]+0.75*z₂[1]))^2)) , 
                -exp(-200*((x[1]-(x₁[end]*0.75+x₁[1]*0.25))^2 + (x[2]-(0.25*z₁[end]+0.75*z₁[1]))^2)) + exp(-200*((x[1]-(x₁[end]*0.25+x₁[1]*0.75))^2 + (x[2]-(0.25*z₂[end]+0.75*z₂[1]))^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

const Δt = 1e-3
tf = 2.0
ntime = ceil(Int, tf/Δt)

let
  t = 0.0
  X₀ = vcat(eltocols(vec(𝐔.(XZ₁))), eltocols(vec(𝐏.(XZ₁))), eltocols(vec(𝐕.(XZ₁))), eltocols(vec(𝐖.(XZ₁))), eltocols(vec(𝐐.(XZ₁))), eltocols(vec(𝐑.(XZ₁))))
  Y₀ = vcat(eltocols(vec(𝐔.(XZ₂))), eltocols(vec(𝐏.(XZ₂))), eltocols(vec(𝐕.(XZ₂))), eltocols(vec(𝐖.(XZ₂))), eltocols(vec(𝐐.(XZ₂))), eltocols(vec(𝐑.(XZ₂))))
  global Z₀ = vcat(X₀, Y₀)
  global maxvals₁ = zeros(Float64, ntime)
  global maxvals₂ = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma*stima
  @gif for i=1:ntime
  # for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(M, sol)    
    t += Δt        
    (i%100 == 0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
    u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
    absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
    absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;

    plt3 = scatter(Tuple.(XZ₁ |> vec), zcolor=vec(absu1), colormap=:matter, markersize=8, msw=0.0, label="", size=(800,800), clims=(0,0.15)); 
    scatter!(plt3, Tuple.(XZ₂ |> vec), zcolor=vec(absu2), colormap=:matter, markersize=8, msw=0.0, label="", size=(800,800), clims=(0,0.15));    
    hline!(plt3, [z₁[1]], lc=:black, lw=2, label="Interface")
    vline!(plt3, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
    vline!(plt3, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
    xlims!(plt3, (x₁[1], x₁[end]))
    ylims!(plt3, (z₂[1], z₁[end]))
    title!(plt3, "\$|u(x,y)|\$ at Time t="*string(round(t,digits=4)));

    plt4 = heatmap(x₁, z₁, vp₁, markersize=4, msw=0.0, label="", size=(800,800));   
    heatmap!(plt4, x₂, z₂, vp₂, markersize=4, msw=0.0, label="", size=(800,800));
    hline!(plt4, [z₁[1]], lc=:black, lw=2, label="Interface")
    vline!(plt4, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
    vline!(plt4, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
    title!(plt4, "Density of the material")

    plot(plt3, plt4, layout=(1,2), size=(1200,800))

    maxvals₁[i] = sqrt(norm(u1ref₁,2)^2 + norm(u2ref₁)^2)
    maxvals₂[i] = sqrt(norm(u1ref₂,2)^2 + norm(u2ref₂)^2)
  # end
  end  every 10 
end  

using ColorSchemes
u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;
# plt3 = heatmap(x₁, z₁, reshape(absu1, (m₁,n₁)), colormap=:matter, ylabel="y(=r)", label="", size=(800,800), xtickfontsize=18, ytickfontsize=18, bottommargin=12*Plots.mm, topmargin=15*Plots.mm, rightmargin=20*Plots.mm, titlefontsize=20, clims=(0, 0.02));  

plt3 = scatter(Tuple.(XZ₁ |> vec), zcolor=vec(absu1), colormap=:matter, markersize=8, msw=0.0, label="", size=(800,800), clims=(0,0.15)); 
scatter!(plt3, Tuple.(XZ₂ |> vec), zcolor=vec(absu2), colormap=:matter, markersize=8, msw=0.0, label="", size=(800,800), clims=(0,0.15));
hline!(plt3, [z₁[1]], lc=:black, lw=2, label="Interface")
vline!(plt3, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
vline!(plt3, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
xlims!(plt3, (x₁[1], x₁[end]))
ylims!(plt3, (z₂[1], z₁[end]))
title!(plt3, "\$|u(x,y)|\$ at Time t="*string(tf));

plt4 = heatmap(x₁, z₁, vp₁, ylabel="y(=r)", markersize=4, msw=0.0, label="", size=(800,800));   
heatmap!(plt4, x₂, z₂, vp₂, ylabel="y(=r)", markersize=4, msw=0.0, label="", size=(800,800));
hline!(plt4, [z₁[1]], lc=:black, lw=2, label="Interface")
vline!(plt4, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
vline!(plt4, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
title!(plt4, "Density of the material")

plot(plt3, plt4, layout=(1,2), size=(1200,800), rightmargin=12*Plots.mm)

plt5_1 = plot();
plt5_2 = plot();
plot!(plt5_1, LinRange(0,tf,ntime), maxvals₁, yaxis = :log10, title="L²-norm Layer 1", label="PML", lw = 2)
plot!(plt5_2, LinRange(0,tf,ntime), maxvals₂, yaxis = :log10, title="L²-norm Layer 2", label="PML", lw = 2)
plot(plt5_1, plt5_2, layout=(1,2), size=(1200,800))