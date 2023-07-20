module SBP

include("./1d_SBP/1d_SBP.jl")
include("./2d_SBP/2d_SBP.jl")

using SBP.SBP_1d: SBP_TYPE, SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1
using SBP.SBP_2d: SBP_1_2_CONSTANT_0_1_0_1

export SBP_2_VARIABLE_0_1, SBP_1_2_CONSTANT_0_1
export SBP_1_2_CONSTANT_0_1_0_1
    
## END OF MODULE
end
