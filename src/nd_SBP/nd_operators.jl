###############################################################################
# Contains functions to implement the n-dimensional version of the SBP method #
###############################################################################
"""
A Dictionary to establish the correspondence between the normal and the grid points
"""
normal_to_side(x,y,z) = Dict([(0,z), (1,x), (-1,y)])
(d::Dict)(k) = d[k];

"""
Function to obtain the jump matrix corresponding to the normal vector
"""
function compute_jump_operators(mn₁::NTuple{2,Int64}, mn₂::NTuple{2,Int64}, 𝐧::AbstractVecOrMat{Int64}; X=[1])
  m₁, n₁ = mn₁  
  m₂, n₂ = mn₂
  # Get the axis of the normal 
  # (0 => x, 1 => y)
  axis = findall(𝐧 .!= [0,0])[1]-1
  # Place the number of points on the corresponding edge at the leading position
  n1, m1 =  normal_to_side((m₁,n₁), 0, (n₁,m₁))[axis]
  n2, m2 =  normal_to_side((m₂,n₂), 0, (n₂,m₂))[axis]
  # Components of the jump matrix
  B11 = kron(normal_to_side(δᵢⱼ(m1,m1,(m1,m1)), δᵢⱼ(1,1,(m1,m1)), I(n1)).(𝐧)...)
  B12 = kron(normal_to_side(δᵢⱼ(m1,1,(m1,m2)), δᵢⱼ(1,m2,(m1,m2)), I(n2)).(𝐧)...)
  B21 = kron(normal_to_side(δᵢⱼ(1,m1,(m2,m1)), δᵢⱼ(m2,1,(m2,m1)), I(n1)).(𝐧)...)
  B22 = kron(normal_to_side(δᵢⱼ(1,1,(m2,m2)), δᵢⱼ(m2,m2,(m2,m2)), I(n2)).(𝐧)...)
  # The jump matrices used in the SAT terms
  BH = [-(X ⊗ B11)  (X ⊗ B12); -(X ⊗ B21)  (X ⊗ B22)]
  BT = [-(X ⊗ B11)  (X ⊗ B12); (X ⊗ B21)  -(X ⊗ B22)]
  BH, BT
end