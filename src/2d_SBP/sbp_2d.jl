"""
SBP operators in two-dimensions obtained using Kronecker Product of 1d operators
"""
struct SBP4_2D <: SBP_TYPE
  D1::NTuple{2,SparseMatrixCSC{Float64,Int64}}
  D2::NTuple{2,SparseMatrixCSC{Float64,Int64}}
  S::NTuple{2,SparseMatrixCSC{Float64,Int64}}
  norm::NTuple{4,SparseMatrixCSC{Float64,Int64}}
  E::NTuple{5,SparseMatrixCSC{Float64,Int64}}
end

"""
Construct the 2d sbp operator using the 1d versions
- D1: Contains the first derivatives approximating ∂/∂q, ∂/∂r
- D2: Contains the second derivatives approximating ∂²/∂q², ∂²/∂r²
- S: Contains the first derivative approximation on the trace 
- norm: Contains the inverse of the diagonal norms on the trace.
- E: Matrix that computes the restriction of the solution on the trace.
"""
function SBP4_2D(sbp_q::SBP4_1D, sbp_r::SBP4_1D)
  # Extract all the matrices from the 1d version
  Hq = sbp_q.norm;  Hr = sbp_r.norm
  Dq = sbp_q.D1; Dr = sbp_r.D1
  Dqq = sbp_q.D2[1]; Drr = sbp_r.D2[1]
  Sq = sbp_q.S; 
  Sr = sbp_r.S; 
  Iq, E0q, Enq = sbp_q.E;
  Ir, E0r, Enr = sbp_r.E;
  # Create the 2d operators from the 1d operators
  Dq2 = Dq ⊗ Ir; Dr2 = Iq ⊗ Dr
  Sq2 = Sq ⊗ Ir; Sr2 = Iq ⊗ Sr
  Dqq2 = Dqq ⊗ Ir; Drr2 = Iq ⊗ Drr
  E0q2 = E0q ⊗ Ir; Enq2 = Enq ⊗ Ir
  E0r2 = Iq ⊗ E0r; Enr2 = Iq ⊗ Enr

  Hq02 = ((Hq\Iq)*E0q) ⊗ Ir; Hr02 = Iq ⊗ ((Hr\Ir)*E0r)
  Hqn2 = ((Hq\Iq)*Enq) ⊗ Ir; Hrn2 = Iq ⊗ ((Hr\Ir)*Enr)
  Iqr = Iq ⊗ Ir

  SBP4_2D( (Dq2, Dr2), (Dqq2, Drr2), (Sq2, Sr2), (Hq02, Hqn2, Hr02, Hrn2), (Iqr, E0q2, E0r2, Enq2, Enr2))
end

"""
Function to generate the 2d grid on the reference domain (0,1)×(0,1)
"""
function reference_grid_2d(mn::Tuple{Int64,Int64})
  m,n = mn
  q = LinRange(0,1,m); r = LinRange(0,1,n)
  qr = [@SVector [q[j],r[i]] for i=1:n, j=1:m];
  qr
end

"""
SBP4_2D_Dqq(A::AbstractMatrix{Float64}): 
- Constructs the operator ≈ ∂/∂q(a(q,r)∂/∂q)
    where Aᵢⱼ = a(qᵢ,rⱼ)
    The result is a sparse matrix that is stored in the field A.
"""
struct SBP4_2D_Dqq <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function SBP4_2D_Dqq(a_qr::AbstractMatrix{Float64})    
  m,n = size(a_qr)
  D2q = [SBP4_VARIABLE_1D(n, a_qr[i,:]).D2 for i=1:m]
  Er = [δᵢⱼ(i,i,(m,m)) for i=1:m]
  SBP4_2D_Dqq(sum(D2q .⊗ Er))
end

"""
SBP4_2D_Drr(A::AbstractMatrix{Float64}): 
- Constructs the operator ≈ ∂/∂r(a(q,r)∂/∂r)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct SBP4_2D_Drr <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function SBP4_2D_Drr(a_qr::AbstractMatrix{Float64})
  m,n = size(a_qr)
  D2r = [SBP4_VARIABLE_1D(m, a_qr[:,i]).D2 for i=1:n]
  Eq = [δᵢⱼ(i,i,(n,n)) for i=1:n]
  SBP4_2D_Drr(sum(Eq .⊗ D2r))
end

"""
SBP4_2D_Dqr(A::AbstractMatrix{Float64}): 
- Constructs the operator ≈ ∂/∂q(a(q,r)∂/∂r)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct SBP4_2D_Dqr <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function SBP4_2D_Dqr(a_qr::AbstractMatrix{Float64})    
  A = spdiagm(vec(a_qr))    
  m,n = size(a_qr)
  sbp_q = SBP4_1D(n)
  sbp_r = SBP4_1D(m)
  sbp_2d = SBP4_2D(sbp_q, sbp_r)   
  D1q, D1r = sbp_2d.D1
  SBP4_2D_Dqr(D1q*A*D1r)
end

"""
SBP4_2D_Drq(A::AbstractMatrix{Float64}): 
- Constructs the operator ≈ ∂/∂r(a(q,r)∂/∂q)
    where Aᵢⱼ = a(qᵢ,rⱼ)
The result is a sparse matrix that is stored in the field A.    
"""
struct SBP4_2D_Drq<: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function SBP4_2D_Drq(a_qr::AbstractMatrix{Float64})    
  A = spdiagm(vec(a_qr))    
  m,n = size(a_qr)
  sbp_q = SBP4_1D(n)
  sbp_r = SBP4_1D(m)
  sbp_2d = SBP4_2D(sbp_q, sbp_r) 
  D1q, D1r = sbp_2d.D1
  SBP4_2D_Drq(D1r*A*D1q)
end

#######################################################
# Sample SBP operators for Linear Elasticity problems #
#######################################################
"""
Linear Elasticity bulk SBP operator:

1) elasticity(Pqr) 
  - Input: (Pqr) is the material property tensor evaluated at every grid points.
  - Output: SparseMatrixCSC{Float64, Int64} containing the individual matrices approximating the derivatives in the elastic wave equations
             ≈ 𝛛/𝛛𝐪(𝐀 𝛛/𝛛𝐪) + 𝛛/𝛛𝐫(𝐁 𝛛/𝛛𝐫) + 𝛛/𝛛𝐪(𝐂 𝛛/𝛛𝐫) + 𝛛/𝛛𝐫(𝐂ᵀ 𝛛/𝛛𝐪)
"""

struct elasticity_operator <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}
end
function elasticity_operator(P::Function, Ω::Function, qr::AbstractMatrix{SVector{2,Float64}})
  P_on_grid = transform_material_properties.(P, Ω, qr)
  P_vec = get_property_matrix_on_grid(P_on_grid, 2)
  _compute_divergence_stress_tensor(P_vec)
end

function elasticity_operator(P_on_grid::AbstractMatrix{SMatrix{4,4,Float64,16}})
  P_vec = get_property_matrix_on_grid(P_on_grid, 2)
  _compute_divergence_stress_tensor(P_vec)
end

function _compute_divergence_stress_tensor(P_vec)
  Dqq2 = [SBP4_2D_Dqq SBP4_2D_Dqq; 
          SBP4_2D_Dqq SBP4_2D_Dqq];
  Dqr2 = [SBP4_2D_Dqr SBP4_2D_Dqr; 
          SBP4_2D_Dqr SBP4_2D_Dqr]
  Drq2 = [SBP4_2D_Drq SBP4_2D_Drq; 
          SBP4_2D_Drq SBP4_2D_Drq];
  Drr2 = [SBP4_2D_Drr SBP4_2D_Drr; 
          SBP4_2D_Drr SBP4_2D_Drr]; 
  De2 = [Dqq2 Dqr2; Drq2 Drr2];
  D = [De2[i,j](P_vec[i,j]).A for i=1:4, j=1:4]
  # Divergence
  res = [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[3,3] D[3,4]; D[4,3] D[4,4]] +
        [D[1,3] D[1,4]; D[2,3] D[2,4]] + [D[3,1] D[3,2]; D[4,1] D[4,2]]
  elasticity_operator(res)
end

"""
Linear Elasticity traction SBP operator:

1) elasticity_traction_operator(Pqr)
  - Input: (Pqr) is the material property tensor evaluated at every grid points.
  - Output: Sparse matrices
              elasticity_traction_operator.A = A(I₂⊗Sq) + C(I₂⊗Dr) ≈ 𝐀 𝛛/𝛛𝐪 + 𝐂 𝛛/𝛛𝐫
              elasticity_traction_operator.B = Cᵀ(I₂⊗Dq) + B(I₂⊗Sr) ≈ 𝐂ᵀ 𝛛/𝛛𝐪 + 𝐁 𝛛/𝛛𝐫
            where [A C; Cᵀ B] = spdiagm.(vec.(get_property_matrix_on_grid(Pqr)))
"""
struct elasticity_traction_operator <: SBP_TYPE
  A::SparseMatrixCSC{Float64, Int64}  
end
function elasticity_traction_operator(P::Function, Ω::Function, qr::AbstractMatrix{SVector{2,Float64}}, 𝐧::AbstractVecOrMat{Int64}; X=[1])    
  # Compute the material properties on the reference grid
  P_on_grid = transform_material_properties.(P, Ω, qr)
  P_vec = spdiagm.(vec.(get_property_matrix_on_grid(P_on_grid, 2)))
  n,m = size(qr)
  sbp_q = SBP4_1D(m)
  sbp_r = SBP4_1D(n)
  sbp_2d = SBP4_2D(sbp_q, sbp_r) 
  Dq, Dr = sbp_2d.D1
  Sq, Sr = sbp_2d.S
  # Compute the traction  
  J = surface_jacobian(Ω, qr, 𝐧; X=I(2))
  J⁻¹ = J\I(size(J,1))
  Pn = ([P_vec[1,1]  P_vec[1,2]; P_vec[2,1]  P_vec[2,2]]*𝐧[1] + [P_vec[3,1]  P_vec[3,2]; P_vec[4,1]  P_vec[4,2]]*𝐧[2], 
        [P_vec[1,3]  P_vec[1,4]; P_vec[2,3]  P_vec[2,4]]*𝐧[1] + [P_vec[3,3]  P_vec[3,4]; P_vec[4,3]  P_vec[4,4]]*𝐧[2])
  ∇n = ((I(2)⊗Sq)*𝐧[1] + (I(2)⊗Dq)*𝐧[2], (I(2)⊗Dr)*𝐧[1] + (I(2)⊗Sr)*𝐧[2])
  𝐓𝐧 = Pn[1]*∇n[1] + Pn[2]*∇n[2]   
  Tr = J⁻¹*𝐓𝐧
  elasticity_traction_operator(X⊗Tr)
end

"""
Function to return the SAT term on the interface. 
Input: interface_SAT_operator(𝛀₁::Tuple{Function, AbstractMatrix{SVector{2,Float64}}}, 
                              𝛀₂::Tuple{Function, AbstractMatrix{SVector{2,Float64}}}, 
                              𝐧₁::AbstractVecOrMat{Int64}, 𝐧₂::AbstractVecOrMat{Int64})

The normal 𝐧₁ (𝐧₂) is the outward normal corresponding to the interface on Layer 1 (Layer 2).  
The condition 𝐧₂ = -𝐧₁ must be satisfied

NOTE: 
1) This function only works for conforming interfaces.
2) All the normals are specified on the reference grid [0,1]^2

(Needs improvement...)
"""
function interface_SAT_operator(𝛀₁::Tuple{Function, AbstractMatrix{SVector{2,Float64}}}, 𝛀₂::Tuple{Function, AbstractMatrix{SVector{2,Float64}}}, 𝐧₁::AbstractVecOrMat{Int64}, 𝐧₂::AbstractVecOrMat{Int64}; X=[1])  
  Ω₁, qr₁ = 𝛀₁
  Ω₂, qr₂ = 𝛀₂
  
  @assert 𝐧₁ == -𝐧₂ "Sides chosen should be shared between the two domains"
  # @assert 𝛀₁.mn == 𝛀₂.mn "The interface needs to be conforming"
  n₁, m₁ = size(qr₁)
  n₂, m₂ = size(qr₂)
  sbp_q₁, sbp_r₁ =  SBP4_1D(m₁), SBP4_1D(n₁)
  sbp_q₂, sbp_r₂ =  SBP4_1D(m₂), SBP4_1D(n₂)
  B̂, B̃ = compute_jump_operators((m₁,n₁), (m₂,n₂), 𝐧₁; X=X)
  Y = I(size(X,2))
  # Get the axis of the normal 
  # (0 => x, 1 => y)
  axis = findall(𝐧₁ .!= [0,0])[1]-1;   
  # Place the number of points on the corresponding edge at the leading position
  n1, m1 =  normal_to_side((m₁,n₁), 0, (n₁,m₁))[axis] 
  n2, m2 =  normal_to_side((m₂,n₂), 0, (n₂,m₂))[axis]
  # Check if the interface is conforming
  @assert n1 == n2 "Interface must be conforming (i.e., equal number of grid points on the common edge of both the domains)"
  # Extract the norm corresponding to the side
  H = SBP4_1D(n1).norm 
  # Expand the surface norm on the 2d grid
  H1 = kron(normal_to_side(δᵢⱼ(m1,m1,(m1,m1)), δᵢⱼ(1,1,(m1,m1)), H).(𝐧₁)...)
  H2 = kron(normal_to_side(δᵢⱼ(m2,m2,(m2,m2)), δᵢⱼ(1,1,(m2,m2)), H).(𝐧₂)...)
  D2 = blockdiag(Y⊗(H1*surface_jacobian(Ω₁, qr₁, 𝐧₁)), Y⊗(H2*surface_jacobian(Ω₂, qr₂, 𝐧₂)))
  H₁⁻¹ = (sbp_q₁.norm\I(m₁)) ⊗ (sbp_r₁.norm\I(n₁))
  H₂⁻¹ = (sbp_q₂.norm\I(m₂)) ⊗ (sbp_r₂.norm\I(n₂))
  (D2*B̂, D2*B̃, sparse(H₁⁻¹), sparse(H₂⁻¹))
end