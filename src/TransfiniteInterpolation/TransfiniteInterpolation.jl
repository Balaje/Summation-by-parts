module TransfiniteInterpolation

export domain_2d, P, S, J, J⁻¹, J⁻¹s, DiscreteDomain

using NLsolve
using ForwardDiff
using StaticArrays
using LinearAlgebra

include("./ti.jl")

end
