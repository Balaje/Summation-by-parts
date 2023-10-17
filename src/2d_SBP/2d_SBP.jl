module SBP_2d

import SBP.SBP_1d: SBP_TYPE, SBP_1_2_CONSTANT_0_1, SBP_2_VARIABLE_0_1
import SBP.TransfiniteInterpolation: J, J⁻¹, J⁻¹s

export SBP_1_2_CONSTANT_0_1_0_1
export ⊗, Dqq, Drr, Dqr, Drq, Dᴱ, Pᴱ
export Tᴱ
export generate_2d_grid, get_property_matrix_on_grid, P2R, E1
export get_surf_J

using SparseArrays
using LazyArrays
using LinearAlgebra
using StaticArrays
using SplitApplyCombine

include("./sbp_2d.jl")

end
