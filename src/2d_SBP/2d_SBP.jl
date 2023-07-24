module SBP_2d

import SBP.SBP_1d: SBP_TYPE, SBP_1_2_CONSTANT_0_1, SBP_2_VARIABLE_0_1

export SBP_1_2_CONSTANT_0_1_0_1
export ⊗, Dqq, Drr, Dqr
export Tq, Tr
export generate_2d_grid

using SparseArrays
using LazyArrays
using LinearAlgebra
using StaticArrays

include("./sbp_2d.jl")

end
