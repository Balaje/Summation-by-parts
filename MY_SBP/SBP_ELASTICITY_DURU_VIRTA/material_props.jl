#################################################################
# Refer Duru, Virta 2014
# Two layer linear elastic wave propagation:
# Contains the functions that describes the material parameters
#################################################################

## Material parameters

const E = 1.0;
const ν = 0.33;

μ(x) = E/(2*(1+ν)) + 0.5*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;
λ(x) = E*ν/((1+ν)*(1-2ν)) + 0.5*(sin(2π*x[1]))^2*(sin(2π*x[2]))^2;

const ρ = 1.0

c₁₁(x) = 2*μ(x)+λ(x)
c₂₂(x) = 2*μ(x)+λ(x)
c₃₃(x) = μ(x)
c₁₂(x) = λ(x)

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain (Defined in material_props.jl)
"""
𝒫(x) = @SMatrix [c₁₁(x) 0 0 c₁₂(x); 0 c₃₃(x) c₃₃(x) 0; 0 c₃₃(x) c₃₃(x) 0; c₁₂(x) 0 0 c₂₂(x)];

"""
The material properties are extracted from the bigger matrix.
"""
A(x) = @view 𝒫(x)[1:2,1:2]
B(x) = @view 𝒫(x)[3:4,3:4]
C(x) = @view 𝒫(x)[1:2,3:4]
Cᵀ(x) = @view 𝒫(x)[3:4,1:2]

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