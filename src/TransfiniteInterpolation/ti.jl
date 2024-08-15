struct domain_2d
    left::Function
    bottom::Function
    right::Function
    top::Function
    intersection_points::NTuple{4,AbstractVector{Float64}}
end

"""
Function to compute the intersection point of two curves 
"""
function compute_intersection_points(a, b; guess=[0.0,0.0])
    @inline function f!(F,x)
        F[1] = a(x[1])[1] - b(x[2])[1]
        F[2] = a(x[1])[2] - b(x[2])[2]
    end
    x0 = guess
    x1 = nlsolve(f!, x0, autodiff=:forward).zero
    a(x1[1])
end

function domain_2d(left::Function, bottom::Function, right::Function, top::Function)
    P1 = SVector{2}(compute_intersection_points(left, bottom))
    P2 = SVector{2}(compute_intersection_points(bottom, right))
    P3 = SVector{2}(compute_intersection_points(right, top))
    P4 = SVector{2}(compute_intersection_points(top, left))
    domain_2d(left, bottom, right, top, (P1, P2, P3, P4))
end

"""
Transfinite interpolation formula
Define new function
       x->transfinite_interpolation(x, domain) with domain parameter defined.
"""
function transfinite_interpolation(x, domain::domain_2d)
    c₀ = domain.left
    c₁ = domain.bottom
    c₂ = domain.right
    c₃ = domain.top
    P₀₁, P₁₂, P₂₃, P₃₀ = domain.intersection_points
    (1-x[1])*c₀(x[2]) + x[1]*c₂(x[2]) + (1-x[2])*c₁(x[1]) + x[2]*c₃(x[1]) -
        ((1-x[2])*(1-x[1])*P₀₁ + x[2]*x[1]*P₂₃ + x[2]*(1-x[1])*P₃₀ + (1-x[2])*x[1]*P₁₂)
end

"""
Discretization of the Domain
"""
struct DiscreteDomain
  domain::domain_2d
  mn::NTuple{2,Int64}
end

"""
Function to return the Jacobian of the transformation
The entries of the matrices are 
  J(f)[j,k] = ∂f(x)[j]/∂x[k]
i.e.,
  J = [∂f₁/∂x₁ ∂f₁/∂x₂
       ∂f₂/∂x₁ ∂f₂/∂x₂]
We require the transpose in our computations.
Here the parameters
- SS is the transfinite interpolation operator. SS(x) = S(x,domain)
- x is the (x[1],x[2]) pair in the reference grid
"""
function transfinite_interpolation_jacobian(x, S)
  d = size(x,1)
  SMatrix{d,d,Float64}(ForwardDiff.jacobian(S, x))'
end

"""
Function to return the inverse of the Jacobian matrix
Here the parameters
- SS is the transfinite interpolation operator. SS(x) = S(x,domain)
- x is the (x[1],x[2]) pair in the reference grid
"""
function inverse_transfinite_interpolation_jacobian(x, S)
    inv(transfinite_interpolation_jacobian(x, S))
end

"""
Function to reshape the material properties on the grid.

𝐈𝐧𝐩𝐮𝐭 a matrix of tensors (an n×n matrix) evaluated on the grid points.
   Pqr::Matrix{SMatrix{m,n,Float64}} = [𝐏(x₁₁) 𝐏(x₁₂) ... 𝐏(x₁ₙ)
                                        𝐏(x₂₁) 𝐏(x₂₂) ... 𝐏(x₂ₙ)
                                                      ...
                                        𝐏(xₙ₁) 𝐏(xₙ₂)  ... 𝐏(xₙₙ)]
  where 𝐏(x) = [P₁₁(x) P₁₂(x)
                P₂₁(x) P₂₂(x)]
𝐎𝐮𝐭𝐩𝐮𝐭 a matrix of matrix with the following form
result = [ [P₁₁(x₁₁) ... P₁₁(x₁ₙ)        [P₁₂(x₁₁) ... P₁₂(x₁ₙ)
                     ...                          ...
            P₁₁(xₙ₁) ... P₁₁(xₙₙ)],         P₁₂(xₙ₁) ... P₁₂(x₁ₙ)];              
           [P₂₁(x₁₁) ... P₂₁(x₁ₙ)        [P₂₂(x₁₁) ... P₂₂(x₁ₙ)
                     ...                          ...
            P₂₁(xₙ₁) ... P₂₁(xₙₙ)],         P₂₂(xₙ₁) ... P₂₂(x₁ₙ)] 
         ]
"""
function get_property_matrix_on_grid(Pqr, l::Int64)
  m,n = size(Pqr[1])
  Ptuple = Tuple.(Pqr)
  P_page = reinterpret(reshape, Float64, Ptuple)
  dim = length(size(P_page))  
  reshape(splitdimsview(P_page, dim-l), (m,n))
end

"""
Get the surface Jacobian matrix defined as 
  surface_jacobian[i,i] = 1.0,    i ∉ Boundary(𝐧)  
          = |J|*|J⁻¹*𝐧|,   i ∈ Boundary(𝐧)
"""
function surface_jacobian(Ω::Function, qr::AbstractMatrix{SVector{2,Float64}}, 𝐧::AbstractVecOrMat{Int64}; X=[1])
  𝐧 = vec(𝐧)
  # Function to compute the surface jacobian
  m1, m2 = size(qr)
  n(x) = reshape(Float64.(𝐧), (length(𝐧),1))
  nqr = n.(qr)
  Jqr = ((det∘transfinite_interpolation_jacobian).(qr, Ω)).*(inverse_transfinite_interpolation_jacobian.(qr, Ω))
  J_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(Jqr, length(𝐧))))
  n_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(nqr, length(𝐧))))  
    # Get the axis of the normal 
  # (0 => x, 1 => y)
  axis = findall(𝐧 .!= [0,0])[1]-1
  # Place the number of points on the corresponding edge at the leading position
  m2, m1 = normal_to_side((m1,m2), 0, (m2,m1))[axis]
  # Fill in the entries of the matrix corresponding to the edge = 1
  n2s = kron(normal_to_side(δᵢⱼ(m2,m2,(m2,m2)), δᵢⱼ(1,1,(m2,m2)), sparse(I(m1))).(𝐧)...)
  Jn_on_grid = (J_on_grid)*(n_on_grid);
  JJ1 = X⊗sqrt.(sum([(Ji*n2s) for Ji in Jn_on_grid].^2))  
  # Replace "0" with "1"
  JJ0 = spdiagm(ones(size(JJ1,1)))  
  i,j,v = findnz(JJ1)
  for k=1:lastindex(v)
    JJ0[i[k], j[k]] = v[k]
  end
  JJ0
end

"""
Get the bulk Jacobian of the transformation
  bulk_jacobian[i,i] = J(Ω, qr[i,i])
"""
function bulk_jacobian(Ω::Function, qr::AbstractMatrix{SVector{2,Float64}})  
  detJ(x) = (det∘transfinite_interpolation_jacobian)(x,Ω)    
  spdiagm([1,1] ⊗ vec(detJ.(qr)))
end

"""
Function to transform the material properties in the physical grid to the reference grid.
  res = transform_material_properties(material_property_tensor, reference_to_physical_map, point_on_reference_domain)
  Input: 1) "material_property_tensor" is the material property tensor as a function of point in the physical domain
         2) "reference_to_physical_map" is the function that returns the physical coordinates as a function of reference coordinates
         3) "point_on_reference_domain" is a point in the reference domain 
"""
function transform_material_properties(P::Function, Ω::Function, qr::SVector{2,Float64})
  x = Ω(qr)
  invJ = inverse_transfinite_interpolation_jacobian(qr, Ω)
  detJ = (det∘transfinite_interpolation_jacobian)(qr, Ω)
  S = invJ ⊗ I(2)
  m,n = size(S)
  SMatrix{m,n,Float64}(S'*P(x)*S)*detJ
end