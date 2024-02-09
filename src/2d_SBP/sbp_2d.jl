"""
SBP operators in two-dimensions obtained using Kronecker Product of 1d operators
"""
struct SBP_1_2_CONSTANT_0_1_0_1 <: SBP_TYPE
  D1::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
  D2::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
  S::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
  norm::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
  E::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
end

"""
Construct the 2d sbp operator using the 1d versions
- D1: Contains the first derivatives approximating ∂/∂q, ∂/∂r
- D2: Contains the second derivatives approximating ∂²/∂q², ∂²/∂r²
- S: Contains the first derivative approximation on the trace 
- norm: Contains the inverse of the diagonal norms on the trace.
- E: Matrix that computes the restriction of the solution on the trace.
"""
function SBP_1_2_CONSTANT_0_1_0_1(sbp_q::SBP_1_2_CONSTANT_0_1, sbp_r::SBP_1_2_CONSTANT_0_1)
  # Extract all the matrices from the 1d version
  Hq = sbp_q.norm;  Hr = sbp_r.norm
  Dq = sbp_q.D1; Dr = sbp_r.D1
  Dqq = sbp_q.D2[1]; Drr = sbp_r.D2[1]
  Sq = sbp_q.S; 
  Sr = sbp_r.S; 
  Iq = sbp_q.E[1]; E₀q = sbp_q.E[2]; Eₙq = sbp_q.E[3];   
  Ir = sbp_r.E[1]; E₀r = sbp_r.E[2]; Eₙr =  sbp_r.E[3]
  # Create the 2d operators from the 1d operators
  𝐃𝐪 = Dq ⊗ Ir; 𝐃𝐫 = Iq ⊗ Dr
  𝐒𝐪 = Sq ⊗ Ir; 𝐒𝐫 = Iq ⊗ Sr
  𝐃𝐪𝐪 = Dqq ⊗ Ir; 𝐃𝐫𝐫 = Iq ⊗ Drr
  𝐄₀q = E₀q ⊗ Ir; 𝐄ₙq = Eₙq ⊗ Ir
  𝐄₀r = Iq ⊗ E₀r; 𝐄ₙr = Iq ⊗ Eₙr
  𝐇𝐪₀ = ((Hq\Iq)*E₀q) ⊗ Ir; 𝐇𝐫₀ = Iq ⊗ ((Hr\Ir)*E₀r)
  𝐇𝐪ₙ = ((Hq\Iq)*Eₙq) ⊗ Ir; 𝐇𝐫ₙ = Iq ⊗ ((Hr\Ir)*Eₙr)
  𝐄 = Iq ⊗ Ir
  SBP_1_2_CONSTANT_0_1_0_1( (𝐃𝐪,𝐃𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐒𝐪,𝐒𝐫), (𝐇𝐪₀,𝐇𝐪ₙ,𝐇𝐫₀,𝐇𝐫ₙ), (𝐄, 𝐄₀q, 𝐄₀r, 𝐄ₙq, 𝐄ₙr) )
end

"""
Function to generate the 2d grid on the reference domain (0,1)×(0,1)
"""
function generate_2d_grid(mn::Tuple{Int64,Int64})
  m,n = mn
  q = LinRange(0,1,m); r = LinRange(0,1,n)
  qr = [@SVector [q[j],r[i]] for i=1:n, j=1:m];
  qr
end

"""
Function to transform the material properties in the physical grid to the reference grid.
  res = P2R(𝒫, 𝒮, qr)
  Input: 1) 𝒫 is the material property tensor
         2) 𝒮 is the function that returns the physical coordinates as a function of reference coordinates
         3) qr is a point in the reference domain 
"""
function P2R(𝒫, 𝒮, qr)
  x = 𝒮(qr)
  invJ = J⁻¹(qr, 𝒮)
  detJ = (det∘J)(qr, 𝒮)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*𝒫(x)*S)*detJ
end

"""
Dqq(A::AbstractMatrix{Float64}): 
- Constructs the operator that approximates Dqqᴬ ≈ ∂/∂q(a(q,r)∂/∂q)
    where Aᵢⱼ = a(qᵢ,rⱼ)
    The result is a sparse matrix that is stored in the field A.
"""
struct Dqq <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function Dqq(a_qr::AbstractMatrix{Float64})    
  m,n = size(a_qr)
  D2q = [SBP_2_VARIABLE_0_1(n, a_qr[i,:]).D2 for i=1:m]
  Er = [E1(i,i,(m,m)) for i=1:m]
  Dqq(sum(D2q .⊗ Er))
end

"""
Drr(A::AbstractMatrix{Float64}): 
- Constructs the operator that approximates Drrᴬ = ∂/∂r(a(q,r)∂/∂r)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct Drr <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function Drr(a_qr::AbstractMatrix{Float64})
  m,n = size(a_qr)
  D2r = [SBP_2_VARIABLE_0_1(m, a_qr[:,i]).D2 for i=1:n]
  Eq = [E1(i,i,(n,n)) for i=1:n]
  Drr(sum(Eq .⊗ D2r))
end

"""
Dqr(A::AbstractMatrix{Float64}): 
- Constructs the operator that approximates Dqrᴬ = ∂/∂q(a(q,r)∂/∂r)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct Dqr <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function Dqr(a_qr::AbstractMatrix{Float64})    
  A = spdiagm(vec(a_qr))    
  m,n = size(a_qr)
  sbp_q = SBP_1_2_CONSTANT_0_1(n)
  sbp_r = SBP_1_2_CONSTANT_0_1(m)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)   
  D1q, D1r = sbp_2d.D1
  Dqr(D1q*A*D1r)
end

"""
Drq(A::AbstractMatrix{Float64}): 
- Constructs the operator that approximates Drqᴬ = ∂/∂r(a(q,r)∂/∂q)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct Drq <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function Drq(a_qr::AbstractMatrix{Float64})    
  A = spdiagm(vec(a_qr))    
  m,n = size(a_qr)
  sbp_q = SBP_1_2_CONSTANT_0_1(n)
  sbp_r = SBP_1_2_CONSTANT_0_1(m)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r) 
  D1q, D1r = sbp_2d.D1
  Drq(D1r*A*D1q)
end

#######################################################
# Sample SBP operators for Linear Elasticity problems #
#######################################################
"""
Linear Elasticity bulk SBP operator:

1) Pᴱ(Pqr) 
  - Input: (Pqr) is the material property tensor evaluated at every grid points.
  - Output: SparseMatrixCSC{Float64, Int64} containing the individual matrices approximating the derivatives in the elastic wave equations
             ≈ 𝛛/𝛛𝐪(𝐀 𝛛/𝛛𝐪) + 𝛛/𝛛𝐫(𝐁 𝛛/𝛛𝐫) + 𝛛/𝛛𝐪(𝐂 𝛛/𝛛𝐫) + 𝛛/𝛛𝐫(𝐂ᵀ 𝛛/𝛛𝐪)
"""

struct Pᴱ <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function Pᴱ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
  P_vec = get_property_matrix_on_grid(Pqr,2)
  Dᴱ₂ = [Dqq Dqq Dqr Dqr; Dqq Dqq Dqr Dqr; Drq Drq Drr Drr; Drq Drq Drr Drr]
  D = [Dᴱ₂[i,j](P_vec[i,j]).A for i=1:4, j=1:4]
  res = [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[3,3] D[3,4]; D[4,3] D[4,4]] +
        [D[1,3] D[1,4]; D[2,3] D[2,4]] + [D[3,1] D[3,2]; D[4,1] D[4,2]]
  Pᴱ(res)
end

"""
Linear Elasticity traction SBP operator:

1) Tᴱ(Pqr)
  - Input: (Pqr) is the material property tensor evaluated at every grid points.
  - Output: Sparse matrices
              Tᴱ.A = A(I₂⊗Sq) + C(I₂⊗Dr) ≈ 𝐀 𝛛/𝛛𝐪 + 𝐂 𝛛/𝛛𝐫
              Tᴱ.B = Cᵀ(I₂⊗Dq) + B(I₂⊗Sr) ≈ 𝐂ᵀ 𝛛/𝛛𝐪 + 𝐁 𝛛/𝛛𝐫
            where [A C; Cᵀ B] = spdiagm.(vec.(get_property_matrix_on_grid(Pqr)))
"""
struct Tᴱ <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}  
end
function Tᴱ(Pqr::Matrix{SMatrix{4,4,Float64,16}}, 𝛀::DiscreteDomain, 𝐧::AbstractVecOrMat{Int64}; X=[1])    
  P_vec = spdiagm.(vec.(get_property_matrix_on_grid(Pqr,2)))
  m,n = 𝛀.mn
  Ω(qr) = S(qr, 𝛀.domain)
  sbp_q = SBP_1_2_CONSTANT_0_1(m)
  sbp_r = SBP_1_2_CONSTANT_0_1(n)
  sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r) 
  Dq, Dr = sbp_2d.D1
  Sq, Sr = sbp_2d.S
  # Compute the traction  
  JJ = Js(𝛀, 𝐧; X=I(2))
  JJ⁻¹ = JJ\I(size(JJ,1))
  Pn = ([P_vec[1,1]  P_vec[1,2]; P_vec[2,1]  P_vec[2,2]]*𝐧[1] + [P_vec[3,1]   P_vec[3,2]; P_vec[4,1]  P_vec[4,2]]*𝐧[2], 
        [P_vec[1,3]  P_vec[1,4]; P_vec[2,3]  P_vec[2,4]]*𝐧[1] + [P_vec[3,3]   P_vec[3,4]; P_vec[4,3]  P_vec[4,4]]*𝐧[2])
  ∇n = ((I(2)⊗Sq)*𝐧[1] + (I(2)⊗Dq)*𝐧[2], (I(2)⊗Dr)*𝐧[1] + (I(2)⊗Sr)*𝐧[2])
  𝐓𝐧 = Pn[1]*∇n[1] + Pn[2]*∇n[2]   
  Tr = JJ⁻¹*𝐓𝐧
  Tᴱ(X⊗Tr)
end

"""
Get the surface Jacobian matrix defined as 
  Js[i,i] = 1.0,    i ∉ Boundary(𝐧)  
          = J⁻¹s(Ω, 𝐧),   i ∈ Boundary(𝐧)
"""
function Js(𝛀::DiscreteDomain, 𝐧::AbstractVecOrMat{Int64}; X=[1])
  𝐧 = vec(𝐧)
  Ω(qr) = S(qr, 𝛀.domain) 
  qr = generate_2d_grid(𝛀.mn) 
  JJ1 = _surface_jacobian(qr, Ω, 𝐧; X=X)  
  JJ0 = spdiagm(ones(size(JJ1,1)))  
  i,j,v = findnz(JJ1)
  for k=1:lastindex(v)
    JJ0[i[k], j[k]] = v[k]
  end
  JJ0
end

"""
Get the bulk Jacobian of the transformation
  Jb[i,i] = J(qr[i,i], Ω)
"""
function Jb(𝛀::DiscreteDomain, 𝐪𝐫)
  Ω(qr) = S(qr, 𝛀.domain)
  detJ(x) = (det∘J)(x,Ω)    
  spdiagm([1,1] ⊗ vec(detJ.(𝐪𝐫)))
end

"""
Struct to dispatch interface SAT routine SATᵢᴱ for Conforming Interface
"""
struct ConformingInterface <: Any end

"""
Function to return the SAT term on the interface. 
Input: SATᵢᴱ(𝛀₁::DiscreteDomain, 
             𝛀₂::DiscreteDomain, 
             𝐧₁::AbstractVecOrMat{Int64}, 
             𝐧₂::AbstractVecOrMat{Int64}, 
             ::ConformingInterface)

The normal 𝐧₁ decides the boundary in Layer 1 on which the interface is situated. 
The normal 𝐧₂ must satisfy the condition 𝐧₂ = -𝐧₁

The function only works for ::ConformingInterface
"""
function SATᵢᴱ(𝛀₁::DiscreteDomain, 𝛀₂::DiscreteDomain, 𝐧₁::AbstractVecOrMat{Int64}, 𝐧₂::AbstractVecOrMat{Int64}, ::ConformingInterface; X=[1])  
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  # @assert 𝐧₁ == -𝐧₂ "Sides chosen should be shared between the two domains"
  # @assert 𝛀₁.mn == 𝛀₂.mn "The interface needs to be conforming"
  m₁, n₁ = 𝛀₁.mn
  m₂, n₂ = 𝛀₂.mn
  sbp_q₁, sbp_r₁ =  SBP_1_2_CONSTANT_0_1(m₁), SBP_1_2_CONSTANT_0_1(n₁)
  sbp_q₂, sbp_r₂ =  SBP_1_2_CONSTANT_0_1(m₂), SBP_1_2_CONSTANT_0_1(n₂)
  B̂, B̃ = jump((m₁,n₁), (m₂,n₂), 𝐧₁; X=X)
  Y = I(size(X,2))
  n1, m1 =  N2S((m₁,n₁), 0, (n₁,m₁))[findall(𝐧₁ .!= [0,0])[1]-1]
  n2, m2 =  N2S((m₂,n₂), 0, (n₂,m₂))[findall(𝐧₂ .!= [0,0])[1]-1]    
  @assert n1==n2
  m = n1
  sbp = SBP_1_2_CONSTANT_0_1(m)   
  H = sbp.norm
  𝐃 = blockdiag(Y⊗(kron(N2S(E1(m1,m1,m1), E1(1,1,m1), H).(𝐧₁)...)*Js(𝛀₁, 𝐧₁)), Y⊗(kron(N2S(E1(m2,m2,m2), E1(1,1,m2), H).(𝐧₂)...)*Js(𝛀₂, 𝐧₂)))      
  H₁⁻¹ = (sbp_q₁.norm\I(m₁)) ⊗ (sbp_r₁.norm\I(n₁))
  H₂⁻¹ = (sbp_q₂.norm\I(m₂)) ⊗ (sbp_r₂.norm\I(n₂))
  (𝐃*B̂, 𝐃*B̃, sparse(H₁⁻¹), sparse(H₂⁻¹))
end

"""
Struct to dispatch inteface SAT routine SATᵢᴱ for non-conforming interface 
"""
struct NonConformingInterface <: Any end

"""
Function to return the SAT term on the interface. 
Input: SATᵢᴱ(𝛀₁::DiscreteDomain, 
             𝛀₂::DiscreteDomain, 
             𝐧₁::AbstractVecOrMat{Int64}, 
             𝐧₂::AbstractVecOrMat{Int64}, 
             ::NonConformingInterface)

The normal 𝐧₁ decides the boundary in Layer 1 on which the interface is situated. 
The normal 𝐧₂ must satisfy the condition 𝐧₂ = -𝐧₁

The function only works for ::NonConformingInterface
"""
function SATᵢᴱ(𝛀₁::DiscreteDomain, 𝛀₂::DiscreteDomain, 𝐧₁::AbstractVecOrMat{Int64}, 𝐧₂::AbstractVecOrMat{Int64}, ::NonConformingInterface; X=[1])  
  Ω₁(qr) = S(qr, 𝛀₁.domain)
  Ω₂(qr) = S(qr, 𝛀₂.domain)
  @assert 𝐧₁ == -𝐧₂ "Sides chosen should be shared between the two domains"
  m₁, n₁ = 𝛀₁.mn
  m₂, n₂ = 𝛀₂.mn
  qr₁ = generate_2d_grid(𝛀₁.mn)
  qr₂ = generate_2d_grid(𝛀₂.mn)    
  n1, m1 =  N2S((m₁,n₁), 0, (n₁,m₁))[findall(𝐧₁ .!= [0,0])[1]-1]
  n2, m2 =  N2S((m₂,n₂), 0, (n₂,m₂))[findall(𝐧₂ .!= [0,0])[1]-1]  
  B̂, B̃ = jump((n1,m1), (n2,m2), (qr₁, qr₂), (Ω₁, Ω₂), 𝐧₁; X=X)    
  sbp_q₁, sbp_r₁ =  SBP_1_2_CONSTANT_0_1(m1), SBP_1_2_CONSTANT_0_1(n1)
  sbp_q₂, sbp_r₂ =  SBP_1_2_CONSTANT_0_1(m2), SBP_1_2_CONSTANT_0_1(n2)
  Hq₁ = sbp_q₁.norm;  Hr₁ = sbp_r₁.norm
  Hq₂ = sbp_q₂.norm;  Hr₂ = sbp_r₂.norm    
  Y = I(size(X,2))   
  𝐃 = blockdiag(Y⊗(kron(N2S(E1(m1,m1,m1), E1(1,1,m1), Hr₁).(𝐧₁)...)*Js(𝛀₁, 𝐧₁)), Y⊗(kron(N2S(E1(m2,m2,m2), E1(1,1,m2), Hr₂).(𝐧₂)...)*Js(𝛀₂, 𝐧₂)))  
  Hq₁⁻¹ = (sbp_q₁.norm\I(m1))
  Hr₁⁻¹ = (sbp_r₁.norm\I(n1))
  Hq₂⁻¹ = (sbp_q₂.norm\I(m2))
  Hr₂⁻¹ = (sbp_r₂.norm\I(n2))
  (𝐃*B̂, 𝐃*B̃, sparse(Hr₁⁻¹⊗Hq₁⁻¹), sparse(Hr₂⁻¹⊗Hq₂⁻¹))
end