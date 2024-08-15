module TransfiniteInterpolation

import SummationByPartsPML.SBP_1d: ⊗, δᵢⱼ
import SummationByPartsPML.nd_SBP: normal_to_side

export domain_2d, compute_intersection_points, transfinite_interpolation, transfinite_interpolation_jacobian
export inverse_transfinite_interpolation_jacobian, get_property_matrix_on_grid, transform_material_properties
export surface_jacobian, bulk_jacobian

using NLsolve
using ForwardDiff
using StaticArrays
using LinearAlgebra
using SplitApplyCombine
using SparseArrays

include("./ti.jl")

end
