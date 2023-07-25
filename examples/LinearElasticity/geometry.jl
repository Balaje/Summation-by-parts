# Module to define the computational domain

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');

"""
Unit normals on the boundary parameterized as c(u).
  𝐧(c, u; o=1.0)
The parameter o=1 or -1 depends on how the surface is oriented. Can be skipped, but defaults to 1.

EXPLANATION:
  I traverse left to right (u = 0..1) always, so the normal direction obeys the right-hand rule.
  Thus -1 needs to be multiplied to obtain the correct normal on the bottom and right boundaries.
"""
function 𝐧(c,u; o=1.0) 
  res = ForwardDiff.derivative(t->c(t), u) # Tangent vector in the physical domain
  r = @SMatrix [0 -1; 1 0] # Rotation matrix
  o*r*res/norm(res) # Normal vector
end
