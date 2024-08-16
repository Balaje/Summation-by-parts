using SummationByPartsPML
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

include("elastic_wave_operators.jl");
include("plotting_functions.jl");
include("pml_stiffness_mass_matrices.jl");
include("time_discretization.jl");

##### ##### ##### ##### 
# Define the domain
##### ##### ##### ##### 
cᵢ(q) = @SVector [4.4π*q, 0.8π*exp(-40π*(q-0.5)^2)]
c₀¹(r) = @SVector [0.0, 4π*r]
c₁¹(q) = cᵢ(q)
c₂¹(r) = @SVector [4.4π, 4π*r]
c₃¹(q) = @SVector [4.4π*q, 4π]
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)
c₀²(r) = @SVector [0.0, 4π*r - 4π]
c₁²(q) = @SVector [4.4π*q, -4π]
c₂²(r) = @SVector [4.4π, 4π*r-4π]
c₃²(q) = cᵢ(q)
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)


##### ##### ##### ##### ##### ##### 
# EXAMPLE OF AN ANISOTROPIC DOMAIN
##### ##### ##### ##### ##### ##### 
# """
# Material properties coefficients of an anisotropic material
# """
# c₁₁¹(x) = 4.0
# c₂₂¹(x) = 20.0
# c₃₃¹(x) = 2.0
# c₁₂¹(x) = 3.8

# c₁₁²(x) = 4*c₁₁¹(x)
# c₂₂²(x) = 4*c₂₂¹(x)
# c₃₃²(x) = 4*c₃₃¹(x)
# c₁₂²(x) = 4*c₁₂¹(x)

# ρ₁(x) = 1.0
# ρ₂(x) = 0.25

##### ##### ##### ##### ##### ##### 
# EXAMPLE OF AN ISOTROPIC DOMAIN
##### ##### ##### ##### ##### ##### 
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
const δ = 0.1*Lᵥ
const σ₀ᵛ = 4*((max(cp₁, cp₂)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*((max(cs₁, cs₂)))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ Lᵥ) || x[1] > Lᵥ)
    return σ₀ᵛ*((x[1] - Lᵥ)/δ)^3  
  else
    return 0.0
  end
end

"""
Horizontal PML strip
"""
function σₕ(x)
  if((x[2] ≈ Lₕ) || (x[2] > Lₕ))
    return σ₀ʰ*((x[2] - Lₕ)/δ)^3  
  elseif( (x[2] ≈ -Lₕ) || (x[2] < -Lₕ) )
    return σ₀ʰ*abs((x[2] + Lₕ)/δ)^3  
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
Material velocity tensors
"""
Z₁¹(x) = @SMatrix [√(c₁₁¹(x)*ρ₁(x))  0;  0 √(c₃₃¹(x)*ρ₁(x))]
Z₂¹(x) = @SMatrix [√(c₃₃¹(x)*ρ₁(x))  0;  0 √(c₂₂¹(x)*ρ₁(x))]

Z₁²(x) = @SMatrix [√(c₁₁²(x)*ρ₂(x))  0;  0 √(c₃₃²(x)*ρ₂(x))]
Z₂²(x) = @SMatrix [√(c₃₃²(x)*ρ₂(x))  0;  0 √(c₂₂²(x)*ρ₂(x))]

"""
Initial conditions
"""
𝐔(x) = @SVector [exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2)), exp(-20*((x[1]-2π)^2 + (x[2]-1.6π)^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Discretize the domain using a mapping to the reference grid [0,1]^2   
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
N = 101;
Ω₁(qr) = transfinite_interpolation(qr, domain₁);
Ω₂(qr) = transfinite_interpolation(qr, domain₂);
qr₁ = reference_grid_2d((round(Int64, 1.1*N - 0.1),N));
qr₂ = reference_grid_2d((round(Int64, 1.1*N - 0.1),N));
xy₁ = Ω₁.(qr₁);
xy₂ = Ω₂.(qr₂);
n₁, m₁ = size(qr₁); n₂, m₂ = size(qr₂);

##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the stiffness and mass matrices
##### ##### ##### ##### ##### ##### ##### ##### 
𝒫 = 𝒫₁, 𝒫₂
𝒫ᴾᴹᴸ = 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ
Z₁₂ = (Z₁¹, Z₂¹), (Z₁², Z₂²)
σₕσᵥ = σₕ, σᵥ
ρ = ρ₁, ρ₂
stima = two_layer_elasticity_pml_stiffness_matrix((domain₁,domain₂), (qr₁,qr₂), (𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α));
massma = two_layer_elasticity_pml_mass_matrix((domain₁,domain₂), (qr₁,qr₂), (ρ₁, ρ₂));

##### ##### ##### ##### ##### ##### ##### ##### 
# Define the time stepping parameters
##### ##### ##### ##### ##### ##### ##### ##### 
Δt = 0.2*norm(xy₁[1,1] - xy₁[1,2])/sqrt(max(cp₁, cp₂)^2 + max(cs₁,cs₂)^2)
tf = 40.0
ntime = ceil(Int, tf/Δt)
l2norm = zeros(Float64, ntime)

plt3 = Vector{Plots.Plot}(undef,3);

##### ##### ##### ##### 
# Begin time loop
##### ##### ##### ##### 
let
  t = 0.0
  X₀¹ = vcat(eltocols(vec(𝐔.(xy₁))), eltocols(vec(𝐏.(xy₁))), eltocols(vec(𝐕.(xy₁))), eltocols(vec(𝐖.(xy₁))), eltocols(vec(𝐐.(xy₁))), eltocols(vec(𝐑.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔.(xy₂))), eltocols(vec(𝐏.(xy₂))), eltocols(vec(𝐕.(xy₂))), eltocols(vec(𝐖.(xy₂))), eltocols(vec(𝐐.(xy₂))), eltocols(vec(𝐑.(xy₂))));
  X₀ = vcat(X₀¹, X₀²)
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  M = massma*stima
  count = 1;
  # @gif for i=1:ntime
  Hq = SBP4_1D(round(Int64,1.1*N - 0.1)).norm;
  Hr = SBP4_1D(N).norm;
  Hqr = Hq ⊗ Hr
  # @gif for i=1:ntime
  for i=1:ntime
    sol = X₀, k₁, k₂, k₃, k₄
    X₀ = RK4_1!(M, sol, Δt)    
    t += Δt    
    (i%30==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    #  Extract the displacement field from the raw solution vector
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    u1ref₁,u2ref₁ = split_solution(X₀[1:12*(m₁*n₁)], (n₁,m₁), 12);
    u1ref₂,u2ref₂ = split_solution(X₀[12*(m₁*n₁)+1:12*(m₁*n₁ + m₂*n₂)], (n₂,m₂), 12);
    U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
    U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)
    
    if((i==ceil(Int64, 1/Δt)) || (i == ceil(Int64, 2/Δt)) || (i == ceil(Int64, 5/Δt)))
      plt3[count] = Plots.plot()
      plot_displacement_field!(plt3[count], (xy₁,xy₂), (U1,U2), (0.0,Lᵥ), (-Lₕ,Lₕ), (0.0,δ), (0.0,0.0), cᵢ)
      count += 1
    end

    ##### ##### ##### ##### ##### ##### ##### ##### 
    # Uncomment for producing GIFs.
    # Also uncomment the @gif macro near for loop
    ##### ##### ##### ##### ##### ##### ##### ##### 
    # plt3_gif = Plots.plot();
    # plot_displacement_field!(plt3_gif, (xy₁,xy₂), (U1,U2), (Lₕ,Lᵥ,δ), cᵢ)

    ##### ##### ##### ##### ##### ##### 
    # Compute the discrete L²-norm
    ##### ##### ##### ##### ##### ##### 
    l2norm[i] = sqrt(u1ref₁'*Hqr*u1ref₁ + u2ref₁'*Hqr*u2ref₁ + u1ref₂'*Hqr*u1ref₂ + u2ref₂'*Hqr*u2ref₂)
  end
  # end every 15
  global X₁ = X₀
end  

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Extract the displacement field from the raw solution vector
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
u1ref₁,u2ref₁ = split_solution(X₁[1:12*(m₁*n₁)], (n₁,m₁), 12);
u1ref₂,u2ref₂ = split_solution(X₁[12*(m₁*n₁)+1:12*(m₁*n₁ + m₂*n₂)], (n₂,m₂), 12);
U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)

##### ##### ##### ##### ##### ##### 
# Plot the displacement field.
##### ##### ##### ##### ##### ##### 
plt3_1 = Plots.plot();
plot_displacement_field!(plt3_1, (xy₁,xy₂), (U1,U2), (0.0,Lᵥ), (-Lₕ,Lₕ), (0.0,δ), (0.0,0.0), cᵢ);

##### ##### ##### ##### ##### ##### #####
# Plot the discretized physical domain
##### ##### ##### ##### ##### ##### ##### 
plt4 = Plots.plot();
plot_discretization!(plt4, (xy₁,xy₂), (0.0,Lᵥ), (-Lₕ,Lₕ), (0.0,δ), (0.0,0.0), cᵢ)

##### ##### ##### ##### ##### ##### #####
# Plot the norm of the solution vs time
##### ##### ##### ##### ##### ##### #####
plt5 = Plots.plot(LinRange(0,tf,ntime), l2norm, label="", lw=2, yaxis=:log10)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| \\bf{u} \\|_{H} \$")
Plots.xlims!(plt5, (0,tf))