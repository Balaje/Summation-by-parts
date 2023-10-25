struct domain_2d
    left::Function
    bottom::Function
    right::Function
    top::Function
    Ps::Tuple{AbstractVector{Float64},AbstractVector{Float64},AbstractVector{Float64},AbstractVector{Float64}}
end

"""
Function to compute the intersection point of two curves 
"""
function P(a, b; guess=[0.0,0.0])
    @inline function f!(F,x)
        F[1] = a(x[1])[1] - b(x[2])[1]
        F[2] = a(x[1])[2] - b(x[2])[2]
    end
    x0 = guess
    x1 = nlsolve(f!, x0, autodiff=:forward).zero
    a(x1[1])
end

function domain_2d(left::Function, bottom::Function, right::Function, top::Function)
    P1 = SVector{2}(P(left, bottom))
    P2 = SVector{2}(P(bottom, right))
    P3 = SVector{2}(P(right, top))
    P4 = SVector{2}(P(top, left))
    domain_2d(left, bottom, right, top, (P1, P2, P3, P4))
end

"""
Transfinite interpolation formula
Define new function
       x->S(x, domain) with domain parameter defined.
"""
function S(x, domain::domain_2d)
    c₀ = domain.left
    c₁ = domain.bottom
    c₂ = domain.right
    c₃ = domain.top
    P₀₁, P₁₂, P₂₃, P₃₀ = domain.Ps
    (1-x[1])*c₀(x[2]) + x[1]*c₂(x[2]) + (1-x[2])*c₁(x[1]) + x[2]*c₃(x[1]) -
        ((1-x[2])*(1-x[1])*P₀₁ + x[2]*x[1]*P₂₃ + x[2]*(1-x[1])*P₃₀ + (1-x[2])*x[1]*P₁₂)
end

"""
Discretization of the Domain
"""
struct DiscreteDomain
  domain::domain_2d
  mn::Tuple{Int64,Int64}
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
function J(x, S)
  d = size(x,1)
  SMatrix{d,d,Float64}(ForwardDiff.jacobian(S, x))'
end

"""
Function to return the inverse of the Jacobian matrix
Here the parameters
- SS is the transfinite interpolation operator. SS(x) = S(x,domain)
- x is the (x[1],x[2]) pair in the reference grid
"""
function J⁻¹(x, S)
    inv(J(x, S))
end

"""
Function to compute the surface jacobian. 
Here the parameters
- n is the normal vector in the reference domain.
- S is the transfinite interpolation operator
- qr is the (q,r) pair in the reference grid
"""
function J⁻¹s(x, S, n)  
  norm(J⁻¹(x, S)*n)
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
