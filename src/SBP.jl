module SBP

include("./TransfiniteInterpolation/TransfiniteInterpolation.jl")
include("./1d_SBP/1d_SBP.jl")
include("./nd_SBP/nd_SBP.jl")
include("./2d_SBP/2d_SBP.jl")
include("./PerfectlyMatchedLayers/PerfectlyMatchedLayers.jl")

using SBP.SBP_1d: SBP_TYPE, SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1, INTERPOLATION_4, E1, ⊗
using SBP.SBP_2d: SBP_1_2_CONSTANT_0_1_0_1, Dqq, Drr, Dqr, Drq, generate_2d_grid, P2R, Pᴱ, Tᴱ, Js, Jb, ConformingInterface, NonConformingInterface, SATᵢᴱ
using SBP.TransfiniteInterpolation: domain_2d, P, S, J, J⁻¹, J⁻¹s, DiscreteDomain, get_property_matrix_on_grid
using SBP.nd_SBP: _surface_jacobian, N2S, jump
using SBP.PerfectlyMatchedLayers: P2Rᴾᴹᴸ, Pᴾᴹᴸ, 𝐙, Tᴾᴹᴸ, χᴾᴹᴸ

export SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1, INTERPOLATION_4, E1, ⊗
export SBP_1_2_CONSTANT_0_1_0_1, Dqq, Drr, Dqr, Drq, generate_2d_grid, P2R, Pᴱ, Tᴱ, Js, Jb
export domain_2d, P, S, J, J⁻¹, J⁻¹s, DiscreteDomain, get_property_matrix_on_grid
export ConformingInterface, NonConformingInterface, SATᵢᴱ
export SJ, N2S, jump
export P2Rᴾᴹᴸ, Pᴾᴹᴸ, 𝐙, Tᴾᴹᴸ, χᴾᴹᴸ
    
## END OF MODULE
end
