# Module to define the computational domain
using NLsolve
using ForwardDiff

"""
Function to compute the intersection point of the two curves c₁,c₂
P₁₂ = P(c₁,c₂)
"""
function P(c₁, c₂; guess=[domain[1], domain[2]])   
  function f!(F, x)
    F[1] = c₁(x[1])[1] - c₂(x[2])[1]
    F[2] = c₁(x[1])[2] - c₂(x[2])[2]
  end
  function j!(J, x)
    J[1,1] = ForwardDiff.derivative(y->c₁(y), x[1])[1]
    J[1,2] = -ForwardDiff.derivative(y->c₂(y), x[2])[1]
    J[2,1] = ForwardDiff.derivative(y->c₁(y), x[1])[2]
    J[2,2] = -ForwardDiff.derivative(y->c₂(y), x[2])[2]
  end
  x0 = guess
  nlsolve(f!, j!, x0).zero
end

"""
Parametric Representation of the boundary
Define c₁, c₂, c₃, c₄
"""
c₁(u) = [0.1*sin(2π*u), u]
c₃(u) = [1.0 + 0.1*sin(2π*u), u]
c₂(v) = [v, 0.1*sin(2π*v)]
c₄(v) = [v, 1.0 + 0.1*sin(2π*v)]

"""
The actual transformation
"""
S(x) = Tuple((1-x[2])*c₁(x[1]) + x[2]*c₃(x[1]) + (1-x[1])*c₂(x[2]) + x[1]*c₄(x[2]) - 
((1-x[1])*(1-x[2])*P(c₁,c₂) + x[1]*x[2]*P(c₃,c₄) + x[1]*(1-x[2])*P(c₄,c₁) + (1-x[1])*x[2]*P(c₂,c₃)))


# Testing the transformation
domain = (0.0,1.0,0.0,1.0)
x = LinRange(0,1,51);
y = LinRange(0,1,51);
# Points in the reference domain
X = vec([(x[j], y[i]) for i=1:lastindex(x), j=1:lastindex(y)])
# Convert to Physical domain
SX = S.(X)
plt1 = scatter(SX, markershape=:circle, markersize=1, label="Physical")
plt2 = scatter(X, markershape=:square, markersize=1, label="Reference")
plt3 = plot(plt1, plt2, layout=(1,2),size=(800,400))