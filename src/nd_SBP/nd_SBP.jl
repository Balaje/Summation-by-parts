module nd_SBP

import SBP.SBP_1d: INTERPOLATION_4, E1, ⊗
import SBP.TransfiniteInterpolation: J, J⁻¹, get_property_matrix_on_grid

export _surface_jacobian, N2S, jump

using LinearAlgebra
using SparseArrays

include("./nd_operators.jl")

end