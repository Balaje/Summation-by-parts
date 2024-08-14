module nd_SBP

import SummationByPartsPML.SBP_1d: δᵢⱼ, ⊗

export normal_to_side, compute_jump_operators

using LinearAlgebra
using SparseArrays

include("./nd_operators.jl")

end