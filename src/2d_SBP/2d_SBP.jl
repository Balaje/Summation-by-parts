module SBP_2d

import SBP.SBP_1d: SBP_TYPE, SBP_1_2_CONSTANT_0_1, SBP_2_VARIABLE_0_1, E1, ⊗
import SBP.TransfiniteInterpolation: J, J⁻¹, J⁻¹s, DiscreteDomain, S, get_property_matrix_on_grid
import SBP.nd_SBP: jump, N2S, _surface_jacobian

export SBP_1_2_CONSTANT_0_1_0_1
export Dqq, Drr, Dqr, Drq, Pᴱ
export Tᴱ
export generate_2d_grid, P2R, Js, Jb
export ConformingInterface, SATᵢᴱ

using SparseArrays
using LazyArrays
using LinearAlgebra
using StaticArrays

include("./sbp_2d.jl")

end
