#################################################################
# Refer Duru, Virta 2014
# Two layer linear elastic wave propagation:
# Contains the functions that describes the material parameters
#################################################################

## Material parameters

const E = 1.0;
const ν = 0.33;

const μ = E/(2*(1+ν));
const λ = E*ν/((1+ν)*(1-2ν));

const ρ = 1.0

const c₁₁ = const c₂₂ = 2μ+λ
const c₃₃ = μ
const c₁₂ = λ

"""
The material properties are ideally functions of the grid points.
But as a first try let us use the constant case to see if the code is working.
"""
A(x) = @SMatrix [c₁₁ 0; 0 c₃₃];
B(x) = @SMatrix [c₃₃ 0; 0 c₂₂];
C(x) = @SMatrix [0 c₁₂; c₃₃ 0];
Cᵀ(x) = @SMatrix [0 c₃₃; c₁₂ 0];

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain (Defined in material_props.jl)
"""
𝒫(x) = @SMatrix [c₁₁ 0 0 c₁₂; 0 c₃₃ c₃₃ 0; 0 c₃₃ c₃₃ 0; c₁₂ 0 0 c₂₂];


"""
Gradient (Jacobian) of the displacement field
"""
@inline function ∇(u,x)
 vec(ForwardDiff.jacobian(u, x))
end

"""
Cauchy Stress tensor using the displacement field.

"""
@inline function σ(∇u,x)  
  𝒫(x)*∇u
end

"""
Divergence of a tensor field
  v is a 2×2 matrix here, where each entries are scalar functions
"""
function div(v,x)
  v₁₁(x) = v(x)[1]; 
  v₁₂(x) = v(x)[2]; 
  v₂₁(x) = v(x)[3];
  v₂₂(x) = v(x)[4];   
  ∂xv₁₁ = ForwardDiff.gradient(v₁₁,x)[1];
  ∂xv₁₂ = ForwardDiff.gradient(v₁₂,x)[1];
  ∂yv₂₁ = ForwardDiff.gradient(v₂₁,x)[2];
  ∂yv₂₂ = ForwardDiff.gradient(v₂₂,x)[2];
  @SVector [∂xv₁₁ + ∂yv₂₁; ∂xv₁₂ + ∂yv₂₂]
end