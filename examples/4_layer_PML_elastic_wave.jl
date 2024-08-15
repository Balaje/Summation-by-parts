# include("2d_elasticity_problem.jl");
using SBP
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff

# Needs pyplot() for this to work ...
using PyPlot
using Plots
pyplot()
using LaTeXStrings
using ColorSchemes
PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

using SplitApplyCombine

"""
Flatten the 2d function as a single vector for the time iterations.
  (...Basically convert vector of vectors to matrix...)
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');
eltocols(v::Vector{MVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');


# Define the domain
interface₁(q) = @SVector [-4 + 48*q, -10.0]
interface₂(q) = @SVector [-4 + 48*q, -20.0]
interface₃(q) = @SVector [-4 + 48*q, -30.0]

c₀¹(r) = @SVector [-4.0, 10*(r-1)] # Left
c₁¹(q) = interface₁(q) # Bottom
c₂¹(r) = @SVector [44.0, 10*(r-1)] # Right
c₃¹(q) = @SVector [-4 + 48*q, 0.0] # Top
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)

c₀²(r) = @SVector [-4.0, 10*r-20] # Left
c₁²(q) = interface₂(q) # Bottom
c₂²(r) = @SVector [44.0, 10*r-20] # Right
c₃²(q) = interface₁(q) # Top
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)

c₀³(r) = @SVector [-4.0, 10*r-30] # Left
c₁³(q) = interface₃(q) # Bottom
c₂³(r) = @SVector [44.0, 10*r-30] # Right
c₃³(q) = interface₂(q) # Top
domain₃ = domain_2d(c₀³, c₁³, c₂³, c₃³)

c₀⁴(r) = @SVector [-4.0, -44 + 14*r] # Left
c₁⁴(q) = @SVector [-4 + 48*q, -44.0] # Bottom
c₂⁴(r) = @SVector [44.0, -44 + 14*r] # Right
c₃⁴(q) = interface₃(q) # Top
domain₄ = domain_2d(c₀⁴, c₁⁴, c₂⁴, c₃⁴)

##### ##### ##### ##### ##### ##### 
# We consider an isotropic domain
##### ##### ##### ##### ##### ##### 
"""
Density functions
"""
ρ₁(x) = 1.5
ρ₂(x) = 1.9
ρ₃(x) = 2.1
ρ₄(x) = 3.0

"""
The Lamé parameters μ₁, λ₁ on Layer 1
"""
μ₁(x) = 1.8^2*ρ₁(x)
λ₁(x) = 3.118^2*ρ₁(x) - 2μ₁(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 2
"""
μ₂(x) = 2.3^2*ρ₂(x)
λ₂(x) = 3.984^2*ρ₂(x) - 2μ₂(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 3
"""
μ₃(x) = 2.7^2*ρ₃(x)
λ₃(x) = 4.667^2*ρ₃(x) - 2μ₃(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 4
"""
μ₄(x) = 3^2*ρ₄(x)
λ₄(x) = 5.196^2*ρ₄(x) - 2μ₄(x)


"""
Material properties coefficients of an anisotropic material
"""
c₁₁¹(x) = 2*μ₁(x)+λ₁(x)
c₂₂¹(x) = 2*μ₁(x)+λ₁(x)
c₃₃¹(x) = μ₁(x)
c₁₂¹(x) = λ₁(x)

c₁₁²(x) = 2*μ₂(x)+λ₂(x)
c₂₂²(x) = 2*μ₂(x)+λ₂(x)
c₃₃²(x) = μ₂(x)
c₁₂²(x) = λ₂(x)

c₁₁³(x) = 2*μ₃(x)+λ₃(x)
c₂₂³(x) = 2*μ₃(x)+λ₃(x)
c₃₃³(x) = μ₃(x)
c₁₂³(x) = λ₃(x)

c₁₁⁴(x) = 2*μ₄(x)+λ₄(x)
c₂₂⁴(x) = 2*μ₄(x)+λ₄(x)
c₃₃⁴(x) = μ₄(x)
c₁₂⁴(x) = λ₄(x)

"""
The p- and s- wave speeds
"""
cpx₁ = √(c₁₁¹(1.0)/ρ₁(1.0))
cpy₁ = √(c₂₂¹(1.0)/ρ₁(1.0))
csx₁ = √(c₃₃¹(1.0)/ρ₁(1.0))
csy₁ = √(c₃₃¹(1.0)/ρ₁(1.0))
cp₁ = max(cpx₁, cpy₁)
cs₁ = max(csx₁, csy₁)

cpx₂ = √(c₁₁²(1.0)/ρ₂(1.0))
cpy₂ = √(c₂₂²(1.0)/ρ₂(1.0))
csx₂ = √(c₃₃²(1.0)/ρ₂(1.0))
csy₂ = √(c₃₃²(1.0)/ρ₂(1.0))
cp₂ = max(cpx₂, cpy₂)
cs₂ = max(csx₂, csy₂)

cpx₃ = √(c₁₁³(1.0)/ρ₃(1.0))
cpy₃ = √(c₂₂³(1.0)/ρ₃(1.0))
csx₃ = √(c₃₃³(1.0)/ρ₃(1.0))
csy₃ = √(c₃₃³(1.0)/ρ₃(1.0))
cp₃ = max(cpx₃, cpy₃)
cs₃ = max(csx₃, csy₃)

cpx₄ = √(c₁₁⁴(1.0)/ρ₄(1.0))
cpy₄ = √(c₂₂⁴(1.0)/ρ₄(1.0))
csx₄ = √(c₃₃⁴(1.0)/ρ₄(1.0))
csy₄ = √(c₃₃⁴(1.0)/ρ₄(1.0))
cp₄ = max(cpx₄, cpy₄)
cs₄ = max(csx₄, csy₄)


"""
The PML damping
"""
const L = 40
const δ = 0.1*L
const σ₀ = 4*((max(cp₁, cp₂, cp₃, cp₄)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σ(x)
  if((x[1] ≈ L) || x[1] > L)
    return σ₀*((x[1] - L)/δ)^3  
  elseif((x[1] ≈ 0.0) || x[1] < 0.0)
    return σ₀*((0.0 - x[1])/δ)^3
  else
    return 0.0
  end
end

"""
Horizontal PML strip
"""
function τ(x)
  if((x[2] ≈ -L) || x[2] < -L)
    return σ₀*(((-L) - x[2])/δ)^3
  else
    return 0.0
  end
end

"""
The material property tensor in the physical coordinates
𝒫(x) = [A(x) C(x); 
        C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain. 
"""
𝒫₁(x) = @SMatrix [c₁₁¹(x) 0 0 c₁₂¹(x); 0 c₃₃¹(x) c₃₃¹(x) 0; 0 c₃₃¹(x) c₃₃¹(x) 0; c₁₂¹(x) 0 0 c₂₂¹(x)];
𝒫₂(x) = @SMatrix [c₁₁²(x) 0 0 c₁₂²(x); 0 c₃₃²(x) c₃₃²(x) 0; 0 c₃₃²(x) c₃₃²(x) 0; c₁₂²(x) 0 0 c₂₂²(x)];
𝒫₃(x) = @SMatrix [c₁₁³(x) 0 0 c₁₂³(x); 0 c₃₃³(x) c₃₃³(x) 0; 0 c₃₃³(x) c₃₃³(x) 0; c₁₂³(x) 0 0 c₂₂³(x)];
𝒫₄(x) = @SMatrix [c₁₁⁴(x) 0 0 c₁₂⁴(x); 0 c₃₃⁴(x) c₃₃⁴(x) 0; 0 c₃₃⁴(x) c₃₃⁴(x) 0; c₁₂⁴(x) 0 0 c₂₂⁴(x)];


"""
The material property tensor with the PML is given as follows:
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫₁ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁¹(x) 0 0 0; 0 -σ(x)*c₃₃¹(x) 0 0; 0 0 σ(x)*c₃₃¹(x)  0; 0 0 0 σ(x)*c₂₂¹(x)];
𝒫₂ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁²(x) 0 0 0; 0 -σ(x)*c₃₃²(x) 0 0; 0 0 σ(x)*c₃₃²(x)  0; 0 0 0 σ(x)*c₂₂²(x)];
𝒫₃ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁³(x) 0 0 0; 0 -σ(x)*c₃₃³(x) 0 0; 0 0 σ(x)*c₃₃³(x)  0; 0 0 0 σ(x)*c₂₂³(x)];
𝒫₄ᴾᴹᴸ(x) = @SMatrix [-σ(x)*c₁₁⁴(x) 0 0 0; 0 -σ(x)*c₃₃⁴(x) 0 0; 0 0 σ(x)*c₃₃⁴(x)  0; 0 0 0 σ(x)*c₂₂⁴(x)];


"""
Material velocity tensors
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

Z₁³(x) = @SMatrix [√(c₁₁³(x)*ρ₃(x))  0;  0 √(c₃₃³(x)*ρ₃(x))]
Z₂³(x) = @SMatrix [√(c₃₃³(x)*ρ₃(x))  0;  0 √(c₂₂³(x)*ρ₃(x))]

Z₁⁴(x) = @SMatrix [√(c₁₁⁴(x)*ρ₄(x))  0;  0 √(c₃₃⁴(x)*ρ₄(x))]
Z₂⁴(x) = @SMatrix [√(c₃₃⁴(x)*ρ₄(x))  0;  0 √(c₂₂⁴(x)*ρ₄(x))]