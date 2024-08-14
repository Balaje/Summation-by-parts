module SummationByPartsPML

include("./TransfiniteInterpolation/TransfiniteInterpolation.jl")
include("./1d_SBP/1d_SBP.jl")
include("./nd_SBP/nd_SBP.jl")
include("./2d_SBP/2d_SBP.jl")
include("./PerfectlyMatchedLayers/PerfectlyMatchedLayers.jl")

using SummationByPartsPML.SBP_1d: SBP_TYPE, SBP4_VARIABLE_1D, SBP4_1D, δᵢⱼ, ⊗

using SummationByPartsPML.SBP_2d: SBP4_2D
using SummationByPartsPML.SBP_2d: SBP4_2D_Dqq, SBP4_2D_Dqr, SBP4_2D_Drq, SBP4_2D_Drr, elasticity_operator
using SummationByPartsPML.SBP_2d: elasticity_traction_operator
using SummationByPartsPML.SBP_2d: reference_grid_2d, transform_material_properties, surface_jacobian, bulk_jacobian
using SummationByPartsPML.SBP_2d: interface_SAT_operator

using SummationByPartsPML.TransfiniteInterpolation: domain_2d, compute_intersection_points, transfinite_interpolation, transfinite_interpolation_jacobian
using SummationByPartsPML.TransfiniteInterpolation: inverse_transfinite_interpolation_jacobian, get_property_matrix_on_grid, transform_material_properties
using SummationByPartsPML.TransfiniteInterpolation: surface_jacobian, bulk_jacobian

using SummationByPartsPML.nd_SBP: normal_to_side, compute_jump_operators

using SummationByPartsPML.PerfectlyMatchedLayers: transform_material_properties_pml, elasticity_pml_operator, compute_impedance_function, elasticity_traction_pml_operator, elasticity_absorbing_boundary_pml_operator


# Export all operators
export SBP_TYPE, SBP4_VARIABLE_1D, SBP4_1D, δᵢⱼ, ⊗
export SBP4_2D
export SBP4_2D_Dqq, SBP4_2D_Dqr, SBP4_2D_Drq, SBP4_2D_Drr, elasticity_operator
export elasticity_traction_operator
export reference_grid_2d, transform_material_properties, surface_jacobian, bulk_jacobian
export interface_SAT_operator
export domain_2d, compute_intersection_points, transfinite_interpolation, transfinite_interpolation_jacobian
export inverse_transfinite_interpolation_jacobian, get_property_matrix_on_grid, transform_material_properties
export surface_jacobian, bulk_jacobian
export normal_to_side, compute_jump_operators
export transform_material_properties_pml, elasticity_pml_operator, compute_impedance_function, elasticity_traction_pml_operator, elasticity_absorbing_boundary_pml_operator
    
## END OF MODULE
end
