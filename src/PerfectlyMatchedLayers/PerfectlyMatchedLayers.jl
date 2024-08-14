module PerfectlyMatchedLayers

import SummationByPartsPML.SBP_1d: SBP_TYPE, SBP4_1D, ⊗
import SummationByPartsPML.SBP_2d: SBP4_2D, SBP4_2D_Dqq, SBP4_2D_Drr, SBP4_2D_Dqr, SBP4_2D_Drq, reference_grid_2d, surface_jacobian, elasticity_traction_operator, δᵢⱼ
import SummationByPartsPML.TransfiniteInterpolation: transfinite_interpolation_jacobian, inverse_transfinite_interpolation_jacobian, get_property_matrix_on_grid, transform_material_properties

export transform_material_properties_pml, elasticity_pml_operator, compute_impedance_function, elasticity_traction_pml_operator, elasticity_absorbing_boundary_pml_operator

using SparseArrays
using LinearAlgebra
using StaticArrays

include("./pml.jl")

end