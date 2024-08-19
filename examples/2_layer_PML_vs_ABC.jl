using SummationByPartsPML
using SplitApplyCombine
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

include("elastic_wave_operators.jl");
include("plotting_functions.jl");
include("pml_stiffness_mass_matrices.jl");
include("time_discretization.jl");

"""
Density function 
"""
ρ₁(x) = 1.5
ρ₂(x) = 3.0

"""
The Lamé parameters μ₁, λ₁ on Layer 1
"""
μ₁(x) = 1.8^2*ρ₁(x)
λ₁(x) = 3.118^2*ρ₁(x) - 2μ₁(x)

"""
The Lamé parameters μ₁, λ₁ on Layer 2
"""
μ₂(x) = 3^2*ρ₂(x)
λ₂(x) = 5.196^2*ρ₂(x) - 2μ₂(x)

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

"""
The PML damping
"""
const Lᵥ = 4π
const Lₕ = 4π
const δ = 0.0*4π  
const δ′ = δ # For constructing the geometry
const σ₀ᵛ = (δ > 0.0) ? 4*((max(cp₁, cp₂)))/(2*δ)*log(10^4) : 0.0 #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = (δ > 0.0) ? 0*((max(cs₁, cs₂)))/(2*δ)*log(10^4) : 0.0 #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ Lₕ) || x[1] > Lₕ)
    return (δ > 0.0) ? σ₀ᵛ*((x[1] - Lₕ)/δ)^3 : 0.0
  elseif((x[1] ≈ δ) || x[1] < δ)
    # return (δ > 0.0) ? σ₀ᵛ*((δ - x[1])/δ)^3 : 0.0
    0.0
  else 
    return 0.0      
  end
end

function σₕ(x)
  if((x[2] ≈ Lᵥ) || (x[2] > Lᵥ))
    return (δ > 0.0) ? σ₀ʰ*((x[2] - Lᵥ)/δ)^3 : 0.0
  elseif( (x[2] ≈ -Lᵥ) || (x[2] < -Lᵥ) )
    return (δ > 0.0) ? σ₀ʰ*abs((x[2] + Lᵥ)/δ)^3 : 0.0
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

"""
The material property tensor with the PML is given as follows:
𝒫ᴾᴹᴸ(x) = [-σᵥ(x)*A(x) + σₕ(x)*A(x)      0; 
              0         σᵥ(x)*B(x) - σₕ(x)*B(x)]
where A(x), B(x), C(x) and σₚ(x) are the material coefficient matrices and the damping parameter in the physical domain
"""
𝒫₁ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁¹(x) + σₕ(x)*c₁₁¹(x) 0 0 0; 0 -σᵥ(x)*c₃₃¹(x) + σₕ(x)*c₃₃¹(x) 0 0; 0 0 σᵥ(x)*c₃₃¹(x) - σₕ(x)*c₃₃¹(x)  0; 0 0 0 σᵥ(x)*c₂₂¹(x) - σₕ(x)*c₂₂¹(x)];
𝒫₂ᴾᴹᴸ(x) = @SMatrix [-σᵥ(x)*c₁₁²(x) + σₕ(x)*c₁₁²(x) 0 0 0; 0 -σᵥ(x)*c₃₃²(x) + σₕ(x)*c₃₃²(x) 0 0; 0 0 σᵥ(x)*c₃₃²(x) - σₕ(x)*c₃₃²(x)  0; 0 0 0 σᵥ(x)*c₂₂²(x) - σₕ(x)*c₂₂²(x)];

"""
Impedance matrices
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Define the two domains 
# 1) The first domain is the smaller one with the PML truncation applied at x = Lₕ+δ
# 2) The second domain is the bigger domain (extended 3 times along x-direction) which will act as the reference solution.
# We then compute the difference between 1) and 2) in the maximum norm.
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Define the domain for PML computation
cᵢ_pml(q) = @SVector [(Lₕ+δ′)*q,  0.0π*exp(-40π*(q-0.5)^2)]
c₀¹_pml(r) = @SVector [0.0, (Lᵥ)*r]
c₁¹_pml(q) = cᵢ_pml(q)
c₂¹_pml(r) = @SVector [(Lₕ+δ′), (Lᵥ)*r]
c₃¹_pml(q) = @SVector [(Lₕ+δ′)*q, (Lᵥ)]
domain₁_pml = domain_2d(c₀¹_pml, c₁¹_pml, c₂¹_pml, c₃¹_pml)
c₀²_pml(r) = @SVector [0.0, (Lᵥ)*r-(Lᵥ)]
c₁²_pml(q) = @SVector [(Lₕ+δ′)*q, -(Lᵥ)]
c₂²_pml(r) = @SVector [(Lₕ+δ′), (Lᵥ)*r-(Lᵥ)]
c₃²_pml(q) = cᵢ_pml(q)
domain₂_pml = domain_2d(c₀²_pml, c₁²_pml, c₂²_pml, c₃²_pml)
# Define the domain for full elasticity computation
cᵢ(q) = @SVector [3(Lₕ+δ′)*q,  0.0π*exp(-40*9*π*(q-1/6)^2)]
c₀¹(r) = @SVector [0.0, (Lᵥ)*r]
c₁¹(q) = cᵢ(q)
c₂¹(r) = @SVector [3(Lₕ+δ′), (Lᵥ)*r]
c₃¹(q) = @SVector [3(Lₕ+δ′)*q, (Lᵥ)]
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
c₀²(r) = @SVector [0.0, (Lᵥ)*r-(Lᵥ)]
c₁²(q) = @SVector [3(Lₕ+δ′)*q, -(Lᵥ)]
c₂²(r) = @SVector [3(Lₕ+δ′), (Lᵥ)*r-(Lᵥ)]
c₃²(q) = cᵢ(q)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)

##### ##### ##### ##### 
# Initial condition
##### ##### ##### ##### 
𝐔(x) = @SVector [exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2)), exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

##### ##### ##### ##### ##### #####
# Discretize the domain with PML
##### ##### ##### ##### ##### #####
N₂ = 81;
reference_coordsᴾᴹᴸ = reference_grid_2d((N₂,N₂));
Ω₁ᴾᴹᴸ(qr) = transfinite_interpolation(qr, domain₁_pml);
Ω₂ᴾᴹᴸ(qr) = transfinite_interpolation(qr, domain₂_pml);
xy₁ᴾᴹᴸ = Ω₁ᴾᴹᴸ.(reference_coordsᴾᴹᴸ); 
xy₂ᴾᴹᴸ = Ω₂ᴾᴹᴸ.(reference_coordsᴾᴹᴸ);
𝒫 = 𝒫₁, 𝒫₂
𝒫ᴾᴹᴸ = 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ
Z₁₂ = (Z₁¹, Z₂¹), (Z₁², Z₂²)
σₕσᵥ = σₕ, σᵥ
ρ = ρ₁, ρ₂
h = norm(xy₁ᴾᴹᴸ[1,2] - xy₁ᴾᴹᴸ[1,1])
# Compute the stiffness and mass matrices
stima2_pml =  two_layer_elasticity_pml_stiffness_matrix((domain₁_pml, domain₂_pml), (reference_coordsᴾᴹᴸ, reference_coordsᴾᴹᴸ), (𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α), 400/h);
massma2_pml =  two_layer_elasticity_pml_mass_matrix((domain₁_pml, domain₂_pml), (reference_coordsᴾᴹᴸ, reference_coordsᴾᴹᴸ), (ρ₁, ρ₂));

##### ##### ##### ##### ##### ##### ##### ##### #####
# Discretize the domain for the reference solution
##### ##### ##### ##### ##### ##### ##### ##### #####
N₁ = 3N₂-2
Ω₁(qr) = transfinite_interpolation(qr, domain₁)
Ω₂(qr) = transfinite_interpolation(qr, domain₂)
reference_coords = reference_grid_2d((N₁,N₂))
xy₁ = Ω₁.(reference_coords) 
xy₂ = Ω₂.(reference_coords)
ℙ₁ᴾᴹᴸ(x) = 0*𝒫₁ᴾᴹᴸ(x)
ℙ₂ᴾᴹᴸ(x) = 0*𝒫₂ᴾᴹᴸ(x)
τₕ(x) = 0*σₕ(x)
τᵥ(x) = 0*σᵥ(x)
ℙ = 𝒫₁, 𝒫₂
ℙᴾᴹᴸ = ℙ₁ᴾᴹᴸ, ℙ₂ᴾᴹᴸ
τₕτᵥ = τₕ, τᵥ
h = norm(xy₁[1,2] - xy₁[1,1])
# Compute the stiffness and mass matrices
stima2 =  two_layer_elasticity_pml_stiffness_matrix((domain₁, domain₂), (reference_coords, reference_coords), (ℙ, ℙᴾᴹᴸ, Z₁₂, τₕτᵥ, ρ, 0.0), 400/h);
massma2 =  two_layer_elasticity_pml_mass_matrix((domain₁, domain₂), (reference_coords, reference_coords), (ρ₁, ρ₂));

##### ##### ##### ##### ##### ##### ##### ##### #####
# Parameters for time discretization 
##### ##### ##### ##### ##### ##### ##### ##### #####
Δt = 0.15*norm(xy₁[1,1] - xy₁[1,2])/sqrt(max(cp₁, cp₂)^2 + max(cs₁,cs₂)^2);
tf = 5.0;
ntime = ceil(Int, tf/Δt)
max_abs_error = zeros(Float64, ntime)

##### ##### ##### ##### ##### ##### ##### ##### #####
# Extract the PML subdomain from the reference domain
# Perform a check by comparing with PML solution
##### ##### ##### ##### ##### ##### ##### ##### #####
comput_domain = findall(σᵥ.(xy₁ᴾᴹᴸ) .≈ 0.0)
indices_x = 1:N₂
indices_y = 1:N₂
xy_PML₁ = xy₁ᴾᴹᴸ[comput_domain]
xy_FULL₁ = xy₁[indices_x, indices_y][comput_domain]
@assert xy_PML₁ ≈ xy_FULL₁
##### ##### ##### ##### 
# Begin time loop
##### ##### ##### ##### 
let
  t = 0.0

  ##### ##### ##### ##### ##### ##### 
  # Initialize Reference solution vectors
  ##### ##### ##### ##### ##### ##### 
  X₀¹ = vcat(eltocols(vec(𝐔.(xy₁))), eltocols(vec(𝐏.(xy₁))), eltocols(vec(𝐕.(xy₁))), eltocols(vec(𝐖.(xy₁))), eltocols(vec(𝐐.(xy₁))), eltocols(vec(𝐑.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔.(xy₂))), eltocols(vec(𝐏.(xy₂))), eltocols(vec(𝐕.(xy₂))), eltocols(vec(𝐖.(xy₂))), eltocols(vec(𝐐.(xy₂))), eltocols(vec(𝐑.(xy₂))));
  X₀ = vcat(X₀¹, X₀²)
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  K = massma2*stima2

  ##### ##### ##### ##### ##### #####
  # Initialize PML solution vectors
  ##### ##### ##### ##### ##### #####
  X₀¹_pml = vcat(eltocols(vec(𝐔.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐏.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐕.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐖.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐐.(xy₁ᴾᴹᴸ))), eltocols(vec(𝐑.(xy₁ᴾᴹᴸ))));
  X₀²_pml = vcat(eltocols(vec(𝐔.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐏.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐕.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐖.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐐.(xy₂ᴾᴹᴸ))), eltocols(vec(𝐑.(xy₂ᴾᴹᴸ))));
  X₀_pml = vcat(X₀¹_pml, X₀²_pml)
  k₁_pml = zeros(Float64, length(X₀_pml))
  k₂_pml = zeros(Float64, length(X₀_pml))
  k₃_pml = zeros(Float64, length(X₀_pml))
  k₄_pml = zeros(Float64, length(X₀_pml)) 
  K_pml = massma2_pml*stima2_pml  

  for i=1:ntime
    X₀ = RK4_1!(K, (X₀, k₁, k₂, k₃, k₄), Δt)    
    X₀_pml = RK4_1!(K_pml, (X₀_pml, k₁_pml, k₂_pml, k₃_pml, k₄_pml), Δt)    

    t += Δt        

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    #  Extract the displacement field from the raw solution vector
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    u1ref₁,u2ref₁ = split_solution(X₀[1:12*(N₁*N₂)], (N₁,N₂), 12);
    u1ref₂,u2ref₂ = split_solution(X₀[12*(N₁*N₂)+1:12*(N₁*N₂+N₁*N₂)], (N₁,N₂), 12);
    u1ref₁_pml,u2ref₁_pml = split_solution(X₀_pml[1:12*(N₂*N₂)], (N₂,N₂), 12);
    u1ref₂_pml,u2ref₂_pml = split_solution(X₀_pml[12*(N₂*N₂)+1:12*(N₂*N₂ + N₂*N₂)], (N₂,N₂), 12);

    ##### ##### ##### ##### ##### ##### ##### ##### 
    # Get the domain of interest i.e., Ω - Ωₚₘₗ
    ##### ##### ##### ##### ##### ##### ##### ##### 
    comput_domain = findall(σᵥ.(xy₁ᴾᴹᴸ) .≈ 0.0)
    indices_x = 1:N₂
    indices_y = 1:N₂
    
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    # Compute the error between the reference and PML solutions
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    U_PML₁ = reshape(u1ref₁_pml, (N₂,N₂))[comput_domain]
    U_FULL₁ = reshape(u1ref₁, (N₂,N₁))[indices_x, indices_y][comput_domain]
    DU_FULL_PML₁ = abs.(U_PML₁-U_FULL₁);
    U_PML₂ = reshape(u1ref₂_pml, (N₂,N₂))[comput_domain]
    U_FULL₂ = reshape(u1ref₂, (N₂,N₁))[indices_x, indices_y][comput_domain]
    DU_FULL_PML₂ = abs.(U_PML₂-U_FULL₂);
    V_PML₁ = reshape(u2ref₁_pml, (N₂,N₂))[comput_domain]
    V_FULL₁ = reshape(u2ref₁, (N₂,N₁))[indices_x, indices_y][comput_domain]
    DV_FULL_PML₁ = abs.(V_PML₁-V_FULL₁);
    V_PML₂ = reshape(u2ref₂_pml, (N₂,N₂))[comput_domain]
    V_FULL₂ = reshape(u2ref₂, (N₂,N₁))[indices_x, indices_y][comput_domain]
    DV_FULL_PML₂ = abs.(V_PML₂-V_FULL₂);
    max_abs_error[i] = max(maximum(DU_FULL_PML₁), maximum(DU_FULL_PML₂), maximum(DV_FULL_PML₁), maximum(DV_FULL_PML₂))

    (i%100==0) && println("Done t = "*string(t)*"\t Error = "*string(max_abs_error[i]))
  end
  global Xref = X₀;
  global Xref_pml = X₀_pml;
end

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#  Extract the displacement field from the raw solution vector
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
u1ref₁,u2ref₁ = split_solution(Xref[1:12*(N₁*N₂)], (N₁,N₂), 12);
u1ref₂,u2ref₂ = split_solution(Xref[12*(N₁*N₂)+1:12*(N₁*N₂+N₁*N₂)], (N₁,N₂), 12);
u1ref₁_pml,u2ref₁_pml = split_solution(Xref_pml[1:12*(N₂*N₂)], (N₂,N₂), 12);
u1ref₂_pml,u2ref₂_pml = split_solution(Xref_pml[12*(N₂*N₂)+1:12*(N₂*N₂ + N₂*N₂)], (N₂,N₂), 12);

##### ##### ##### ##### ##### ##### ##### ##### 
# Get the domain of interest i.e., Ω - Ωₚₘₗ
##### ##### ##### ##### ##### ##### ##### ##### 
comput_domain = findall(σᵥ.(xy₁ᴾᴹᴸ) .≈ 0.0);
indices_x = 1:N₂;
indices_y = 1:N₂;

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the error between the reference and PML solutions
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
U_PML₁ = reshape(u1ref₁_pml, (N₂,N₂))[comput_domain]
U_FULL₁ = reshape(u1ref₁, (N₂,N₁))[comput_domain]
DU_FULL_PML₁ = abs.(U_PML₁-U_FULL₁);

##### ##### ##### ##### #####
# Plot the PML solution
##### ##### ##### ##### #####
plt3 = Plots.contourf(getX.(xy₁ᴾᴹᴸ), getY.(xy₁ᴾᴹᴸ), reshape(abs.(u1ref₁_pml),size(xy₁ᴾᴹᴸ)...), colormap=:jet, levels=40)
Plots.contourf!(getX.(xy₂ᴾᴹᴸ), getY.(xy₂ᴾᴹᴸ), reshape(abs.(u1ref₂_pml), size(xy₁ᴾᴹᴸ)...), colormap=:jet, levels=40)
if ((σ₀ᵛ > 0) || (σ₀ʰ > 0))
  Plots.vline!([Lᵥ], label="PML Domain", lc=:black, lw=1, ls=:dash)  
else
  Plots.vline!([Lᵥ+δ′], label="ABC", lc=:black, lw=1, ls=:dash)
end
Plots.plot!(getX.(cᵢ.(LinRange(0,1,N₂))), getY.(cᵢ.(LinRange(0,1,N₂))), label="Interface", lc=:red, lw=2, size=(400,500))
xlims!((0,cᵢ_pml(1.0)[1]))
ylims!((c₀²_pml(0.0)[2], c₀¹_pml(1.0)[2]))
xlabel!("\$x\$")
ylabel!("\$y\$")

##### ##### ##### ##### ##### #####
# Plot the PML reference solution
##### ##### ##### ##### ##### #####
plt4 = Plots.contourf(getX.(xy₁), getY.(xy₁), reshape(abs.(u1ref₁),size(xy₁)...), colormap=:jet, levels=40, cbar=:none)
Plots.contourf!(getX.(xy₂), getY.(xy₂), reshape(abs.(u1ref₂), size(xy₂)...), colormap=:jet, levels=40)
Plots.plot!(getX.(cᵢ.(LinRange(0,1,N₁))), getY.(cᵢ.(LinRange(0,1,N₁))), label="Interface", lc=:red, lw=2, size=(400,500))
xlims!((cᵢ(0)[1],cᵢ(1.0)[1]))
ylims!((c₀²(0.0)[2], c₀¹(1.0)[2]))
if ((σ₀ᵛ > 0) || (σ₀ʰ > 0))
  Plots.plot!([Lᵥ+δ′,Lᵥ+δ′], [-Lₕ-δ′, Lₕ+δ′], label="PML", lc=:black, lw=1, ls=:dash)  
end
Plots.plot!([Lᵥ,Lᵥ], [-Lₕ-δ′, Lₕ+δ′], label="Truncated Region", lc=:green, lw=1, ls=:solid)
xlabel!("\$x\$")
ylabel!("\$y\$")

##### ##### ##### ##### ##### ##### #####
# Plot the maximum norm error with time.
##### ##### ##### ##### ##### ##### #####
# plt5 = Plots.plot()
if (δ > 0)
  Plots.plot!(plt5, LinRange(0,tf, ntime), max_abs_error, yaxis=:log10, label="PML", color=:red, lw=2)
else
  Plots.plot!(plt5, LinRange(0,tf, ntime), max_abs_error, yaxis=:log10, label="ABC", color=:blue, lw=1, legendfontsize=10, ls=:dash)
end
ylims!(plt5, (10^-8, 1))
xlabel!(plt5, "Time")
ylabel!(plt5, "Maximum Error")