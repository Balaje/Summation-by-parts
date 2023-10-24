###############################################################################
# Contains functions to implement the n-dimensional version of the SBP method #
###############################################################################

"""
Function to obtain the jump matrix corresponding to the normal vector
"""
function jump(m::Int64, 𝐧::AbstractVector{Int64}; X=I(2))
  BH = [-(X ⊗ kron(N2S(E1(m,m,m), E1(1,1,m)).(𝐧)...))  (X ⊗ kron(N2S(E1(m,1,m), E1(1,m,m)).(𝐧)...)); 
        -(X ⊗ kron(N2S(E1(1,m,m), E1(m,1,m)).(𝐧)...))  (X ⊗ kron(N2S(E1(1,1,m), E1(m,m,m)).(𝐧)...))]
  BT = [-(X ⊗ kron(N2S(E1(m,m,m), E1(1,1,m)).(𝐧)...))  (X ⊗ kron(N2S(E1(m,1,m), E1(1,m,m)).(𝐧)...)); 
       (X ⊗ kron(N2S(E1(1,m,m), E1(m,1,m)).(𝐧)...))  -(X ⊗ kron(N2S(E1(1,1,m), E1(m,m,m)).(𝐧)...))]
  BH, BT
end

"""
ith unit vector in m-dimensions
"""
eᵢ(i::Int64,m::Int64) = diag(E1(i,i,m))

"""
A Dictionary to obtain the side corresponding to the normal in the ith direction
"""
N2S(x,y) = Dict([(0,I(21)), (1,x), (-1,y)])
(d::Dict)(k) = d[k];

import SBP.E1
"""
Function to return the rectangular version of the boundary marker
"""
function SBP.E1(i,j,mn::Tuple{Int64,Int64})
  m,n = mn
  res = spzeros(Float64, m, n)
  res[i,j] = 1.0
  res
end

"""
Surface Jacobian matrix
"""
function SJ(qr, Ω, 𝐧::AbstractVecOrMat{Int64}; X=I(2))  
  m = size(qr,1)
  n(x) = reshape(Float64.(𝐧), (2,1))
  nqr = n.(qr)
  Jqr = J⁻¹.(qr, Ω)
  J_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(Jqr)))
  n_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(nqr)))  
  n2s = kron(N2S(E1(1,1,m), E1(m,m,m)).(𝐧)...)
  Jn_on_grid = ((J_on_grid)*(J_on_grid));
  [X⊗(Ji*n2s) for Ji in Jn_on_grid]
end