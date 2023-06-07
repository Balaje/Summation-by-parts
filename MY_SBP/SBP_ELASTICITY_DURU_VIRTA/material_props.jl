#################################################################
# Refer Duru, Virta 2014
# Two layer linear elastic wave propagation:
# Contains the functions that describes the material parameters
#################################################################

using FillArrays
using LazyArrays
using Test

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
const A = [c₁₁ 0; 0 c₃₃];
const B = [c₃₃ 0; 0 c₂₂];
const C = [0 c₁₂; c₃₃ 0];
const Cᵀ = [0 c₃₃; c₁₂ 0];

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain (Defined in material_props.jl)
"""
const 𝒫 = [c₁₁   0  0    c₁₂; 
            0   c₃₃  c₃₃  0; 
            0   c₃₃  c₃₃  0;
           c₁₂   0  0    c₂₂];


"""
Gradient (Jacobian) of the displacement field
"""
function ∇(u,x)
  ForwardDiff.jacobian(u, x)
end

"""
Cauchy Stress tensor using the displacement field.
NOTE: x is unused here since we code it for the general case
"""
function σ(∇u)  
  hcat(A*∇u[:,1] + C*∇u[:,2], Cᵀ*∇u[:,1] + B*∇u[:,2])
end

"""
Divergence of a tensor field
(Needs to be simplified)
"""
function divσ(v,x)
  𝛔(x) = σ(∇(v, x));
  j_σ_v = ∇(𝛔,x)
  @SVector [j_σ_v[1,1] + j_σ_v[2,2], j_σ_v[3,1] + j_σ_v[4,2]];
end

@testset "Some tests to verify the Gradient, Stress and Divergence." begin 
  v(x) = [sin(π*x[1])*sin(π*x[2]), sin(2π*x[1])*sin(2π*x[2])];
  ∇v(x) = [π*cos(π*x[1])*sin(π*x[2]) π*sin(π*x[1])*cos(π*x[2]); 
         2π*cos(2π*x[1])*sin(2π*x[2]) 2π*sin(2π*x[1])*cos(2π*x[2])];
  σv(x) = hcat(A*([π*cos(π*x[1])*sin(π*x[2]), 2π*cos(2π*x[1])*sin(2π*x[2])]) + C*([π*sin(π*x[1])*cos(π*x[2]), 2π*sin(2π*x[1])*cos(2π*x[2])]),
         Cᵀ*([π*cos(π*x[1])*sin(π*x[2]), 2π*cos(2π*x[1])*sin(2π*x[2])]) + B*([π*sin(π*x[1])*cos(π*x[2]), 2π*sin(2π*x[1])*cos(2π*x[2])]));
  div_σ_v(x) = A*([-π^2*sin(π*x[1])*sin(π*x[2]), -4π^2*sin(2π*x[1])*sin(2π*x[2])]) + C*([π^2*cos(π*x[1])*cos(π*x[2]), 4π^2*cos(2π*x[1])*cos(2π*x[2])]) + 
             Cᵀ*([π^2*cos(π*x[1])*cos(π*x[2]), 4π^2*cos(2π*x[1])*cos(2π*x[2])]) + B*([-π^2*sin(π*x[1])*sin(π*x[2]), -4π^2*sin(2π*x[1])*sin(2π*x[2])]);

  pt = @SVector rand(2)
  @test ∇v(pt) ≈ ∇(v, pt);  
  @test σv(pt) ≈ σ(∇(v, pt));
  @test div_σ_v(pt) ≈ divσ(v, pt);
end;