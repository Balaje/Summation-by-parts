module SBP

include("./TransfiniteInterpolation/TransfiniteInterpolation.jl")
include("./1d_SBP/1d_SBP.jl")
include("./2d_SBP/2d_SBP.jl")

using SBP.SBP_1d: SBP_TYPE, SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1, INTERPOLATION_4
using SBP.SBP_2d: SBP_1_2_CONSTANT_0_1_0_1, ⊗, Dqq, Drr, Dqr, Drq, generate_2d_grid, get_property_matrix_on_grid, P2R, Pᴱ, Tᴱ, E1, Js, Jb, ConformingInterface, SATᵢᴱ
using SBP.TransfiniteInterpolation: domain_2d, P, S, J, J⁻¹, J⁻¹s, DiscreteDomain

export SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1, INTERPOLATION_4
export SBP_1_2_CONSTANT_0_1_0_1, ⊗, Dqq, Drr, Dqr, Drq, generate_2d_grid, get_property_matrix_on_grid, P2R, Pᴱ, Tᴱ, Js, Jb
export domain_2d, P, S, J, J⁻¹, J⁻¹s, E1, DiscreteDomain
export ConformingInterface, SATᵢᴱ
    
## END OF MODULE
end
