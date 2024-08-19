using SummationByPartsPML
using StaticArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff
# Install pyplot for this to work ....
using PyPlot
using Plots
pyplot()
using LaTeXStrings

PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

using SplitApplyCombine

using Interpolations

using MAT

vars1 = matread("./examples/Marmousi2/marmousi2_downsampled_10.mat");

###########################################################
# CONSTRUCT THE GEOMETRY OF THE DOMAIN USING THE RAW DATA #
###########################################################

# Extract the coordinates of elastic domain
X₂ = vars1["X_e"]/1000; Z₂ = vars1["Z_e"]/1000
n₂, m₂ = size(X₂);
x₂ = X₂[1,:]; z₂ = Z₂[:,1]

# Interface separating Layers 1 & 2
function cᵢ(q)
  x = x₂[1] + (x₂[end]-x₂[1])*q
  y = (x - x₂[1])/(x₂[end]-x₂[1])*(-2.47 + 3.34) - 3.34
  @SVector [x,y]
end

# Layer 1 - Elastic domain
c₀¹(r) = @SVector [x₂[1], cᵢ(0)[2] + (z₂[end]- cᵢ(0)[2])*r] # Left boundary 
c₁¹(q) = cᵢ(q) # Botom boundary
c₂¹(r) = @SVector [x₂[end], cᵢ(1)[2] + (z₂[end] - cᵢ(1)[2])*r] # Right boundary
c₃¹(q) = @SVector [x₂[1] + (x₂[end]-x₂[1])*q, z₂[end]] # Top boundary
domain₁ = domain_2d(c₀¹, c₁¹, c₂¹, c₃¹)

# Layer 2 - Elastic domain
c₀²(r) = @SVector [x₂[1], z₂[1] + (cᵢ(0)[2] - z₂[1])*r] # Left boundary 
c₁²(q) = @SVector [x₂[1] + (x₂[end]-x₂[1])*q, z₂[1]] # Bottom boundary
c₂²(r) = @SVector [x₂[end], z₂[1] + (cᵢ(1)[2] - z₂[1])*r] # Right boundary
c₃²(q) = cᵢ(q) # Top boundary
domain₂ = domain_2d(c₀², c₁², c₂², c₃²)

M₁, N₁ = 601, 101; # 1201 × 201 (in paper) 
M₂, N₂ = 601, 21; # 1201 × 41 (in paper)

Ω₁(qr) = transfinite_interpolation(qr, domain₁);
Ω₂(qr) = transfinite_interpolation(qr, domain₂);

# Reference grids on the two layers
qr₁ = reference_grid_2d((M₁,N₁));
qr₂ = reference_grid_2d((M₂,N₂));

XZ₁ = Ω₁.(qr₁);
XZ₂ = Ω₂.(qr₂);

###################################################
# CONSTRUCT THE MATERIAL PROPERTIES ON THE DOMAIN #
###################################################

# Get the material properties of Layer 2 and 3
vp₂ = vars1["vp_e"]/1000
vs₂ = vars1["vs_e"]/1000
rho₂ = vars1["rho_e"]/1000

# Function to interpolate the wave-speeds and density and obtatin the elastic constants
function get_elastic_constants(VP, VS, RHO, XZ)
  M, N = size(XZ)
  VPᵢ = [VP(XZ[i,j][1], XZ[i,j][2]) for i=1:M, j=1:N]
  VSᵢ = [VS(XZ[i,j][1], XZ[i,j][2]) for i=1:M, j=1:N]
  RHOᵢ = [RHO(XZ[i,j][1], XZ[i,j][2]) for i=1:M, j=1:N]
  MU = (VSᵢ.^2).*RHOᵢ;
  LAMBDA = (VPᵢ.^2).*RHOᵢ - 2*MU
  C₁₁ = C₂₂ = 2*MU + LAMBDA
  C₃₃ = MU
  C₁₂ = LAMBDA
  P = [@SMatrix [C₁₁[i,j] 0 0 C₁₂[i,j]; 0 C₃₃[i,j] C₃₃[i,j] 0; 0 C₃₃[i,j] C₃₃[i,j] 0; C₁₂[i,j] 0  0 C₂₂[i,j]] for i=1:M, j=1:N]
  Z₁ = [@SMatrix [sqrt(C₁₁[i,j]*RHOᵢ[i,j]) 0; 0 sqrt(C₃₃[i,j]*RHOᵢ[i,j])] for i=1:M, j=1:N]
  Z₂ = [@SMatrix [sqrt(C₃₃[i,j]*RHOᵢ[i,j]) 0; 0 sqrt(C₂₂[i,j]*RHOᵢ[i,j])] for i=1:M, j=1:N]
  (C₁₁, C₂₂, C₁₂, C₃₃), P, (Z₁, Z₂), RHOᵢ
end

# Perform interpolations using the raw data
VP2 = LinearInterpolation((x₂,z₂), vp₂', extrapolation_bc=Line());
VS2 = LinearInterpolation((x₂,z₂), vs₂', extrapolation_bc=Line());
RHO2 = LinearInterpolation((x₂,z₂), rho₂', extrapolation_bc=Line());

(C₁₁¹, C₂₂¹, C₁₂¹, C₃₃¹), P₁, (𝐙₁¹, 𝐙₂¹), RHO₁ = get_elastic_constants(VP2, VS2, RHO2, XZ₁)
(C₁₁², C₂₂², C₁₂², C₃₃²), P₂, (𝐙₁², 𝐙₂²), RHO₂ = get_elastic_constants(VP2, VS2, RHO2, XZ₂)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Build a dictionary with the material properties and Physical Coordinates
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
material_properties_on_layer_1 = Dict(XZ₁ .=> P₁);
material_properties_on_layer_2 = Dict(XZ₂ .=> P₂);
𝒫₁(x) = material_properties_on_layer_1[x]
𝒫₂(x) = material_properties_on_layer_2[x]

impedance_matrix_on_layer_1₁ = Dict(XZ₁ .=> 𝐙₁¹)
impedance_matrix_on_layer_1₂ = Dict(XZ₁ .=> 𝐙₂¹)
impedance_matrix_on_layer_2₁ = Dict(XZ₂ .=> 𝐙₁²)
impedance_matrix_on_layer_2₂ = Dict(XZ₂ .=> 𝐙₂²)
Z₁¹(x) = impedance_matrix_on_layer_1₁[x]
Z₂¹(x) = impedance_matrix_on_layer_1₂[x]
Z₁²(x) = impedance_matrix_on_layer_2₁[x]
Z₂²(x) = impedance_matrix_on_layer_2₂[x]

density_on_layer_1 = Dict(XZ₁ .=> RHO₁)
density_on_layer_2 = Dict(XZ₂ .=> RHO₂)
ρ₁(x) = density_on_layer_1[x]
ρ₂(x) = density_on_layer_2[x]