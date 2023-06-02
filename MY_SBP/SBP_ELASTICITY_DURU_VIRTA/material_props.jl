#################################################################
# Refer Duru, Virta 2014
# Two layer linear elastic wave propagation:
# Contains the functions that describes the material parameters
#################################################################

using FillArrays
using LazyArrays

## Material parameters

const E = 1.0;
const ν = 0.33;

const μ = E/(2*(1+ν));
const λ = E*ν/((1+ν)*(1-2ν));

const ρ = 1.0

const c₁₁ = c₂₂ = 2μ+λ
const c₃₃ = μ
const c₁₂ = λ

#= # These are the material property functions. 
# In this case we assume orthotropic, anisotropic media
A₁₁(x) = c₁₁; A₁₂(x) = 0.0; A₂₁(x) = 0.0; A₂₂(x) = c₃₃;
B₁₁(x) = c₃₃; B₁₂(x) = 0.0; B₂₁(x) = 0.0; B₂₂(x) = c₂₂;
C₁₁(x) = 0.0; C₁₂(x) = c₁₂; C₂₁(x) = c₃₃; C₂₂(x) = 0.0;

"""
Material property tensors
"""
A(x) = @SMatrix [A₁₁(x) A₁₂(x); A₂₁(x) A₂₂(x)];
B(x) = @SMatrix [B₁₁(x) B₁₂(x); B₂₁(x) B₂₂(x)];
C(x) = @SMatrix [C₁₁(x) C₁₂(x); C₂₁(x) C₂₂(x)];
Cᵀ(x) = @SMatrix [C₁₁(x) C₂₁(x); C₁₂(x) C₂₂(x)]; =#

"""
The material properties are ideally functions of the grid points.
But as a first try let us use the constant case to see if the code is working.
"""
A = [c₁₁ 0; 0 c₃₃];
B = [c₃₃ 0; 0 c₂₂];
C = [0 c₁₂; c₃₃ 0];
Cᵀ = [0 c₃₃; c₁₂ 0];

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain (Defined in material_props.jl)
"""
𝒫 = [c₁₁   0  0    c₁₂; 
      0   c₃₃  c₃₃  0; 
      0   c₃₃  c₃₃  0;
     c₁₂   0  0    c₂₂];