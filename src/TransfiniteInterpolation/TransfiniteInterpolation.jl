module TransfiniteInterpolation

export domain_2d, P, S, J, J⁻¹, J⁻¹s, DiscreteDomain, get_property_matrix_on_grid

using NLsolve
using ForwardDiff
using StaticArrays
using LinearAlgebra
using SplitApplyCombine

include("./ti.jl")

end
