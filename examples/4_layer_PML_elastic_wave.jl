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

"""
Initial conditions
"""
𝐔(x) = @SVector [0.0, 0.0]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Discretize the domain using a mapping to the reference grid [0,1]^2   
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
h = 0.1
Nx = ceil(Int64, 48/h) + 1
Ny = ceil(Int64, 10/h) + 1
Ny1 = ceil(Int64, 14/h) + 1
Ω₁(qr) = transfinite_interpolation(qr, domain₁);
Ω₂(qr) = transfinite_interpolation(qr, domain₂);
Ω₃(qr) = transfinite_interpolation(qr, domain₃);
Ω₄(qr) = transfinite_interpolation(qr, domain₄);
qr₁ = reference_grid_2d((Nx,Ny));
qr₂ = reference_grid_2d((Nx,Ny));
qr₃ = reference_grid_2d((Nx,Ny));
qr₄ = reference_grid_2d((Nx,Ny1));
xy₁ = Ω₁.(qr₁);
xy₂ = Ω₂.(qr₂);
xy₃ = Ω₂.(qr₃);
xy₄ = Ω₂.(qr₄);

##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the stiffness and mass matrices
##### ##### ##### ##### ##### ##### ##### ##### 
𝒫 = 𝒫₁, 𝒫₂, 𝒫₃, 𝒫₄
𝒫ᴾᴹᴸ = 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ, 𝒫₃ᴾᴹᴸ, 𝒫₄ᴾᴹᴸ
Z₁₂ = (Z₁¹, Z₂¹), (Z₁², Z₂²), (Z₁³, Z₂³), (Z₁⁴, Z₂⁴)
σₕσᵥ = τ, σ
ρ = ρ₁, ρ₂, ρ₃, ρ₄
stima = four_layer_elasticity_pml_stiffness_matrix((domain₁,domain₂,domain₃,domain₄), (qr₁,qr₂,qr₃,qr₄), (𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α));
massma = four_layer_elasticity_pml_mass_matrix((domain₁,domain₂,domain₃,domain₄), (qr₁,qr₂,qr₃,qr₄), (ρ₁, ρ₂, ρ₃, ρ₄));

#=
"""
Right hand side function. 
  In this example, we drive the problem using an explosive moment tensor point source.
"""
function f(t::Float64, x::SVector{2,Float64}, params)
  s₁, s₂, M₀ = params
  @SVector[-1/(2π*√(s₁*s₂))*exp(-(x[1]-20)^2/(2s₁) - (x[2]+15)^2/(2s₂))*(x[1]-20)/s₁*exp(-(t-0.215)^2/0.15)*M₀,
           -1/(2π*√(s₁*s₂))*exp(-(x[1]-20)^2/(2s₁) - (x[2]+15)^2/(2s₂))*(x[2]+15)/s₂*exp(-(t-0.215)^2/0.15)*M₀]
end

##### ##### ##### ##### ##### ##### ##### ##### 
# Define the time stepping parameters
##### ##### ##### ##### ##### ##### ##### ##### 
Δt = 0.2*h/sqrt(max((cp₁^2+cs₁^2), (cp₂^2+cs₂^2), (cp₃^2+cs₃^2), (cp₄^2+cs₄^2)));
tf = 5.0
ntime = ceil(Int, tf/Δt)
Δt = tf/ntime;
l2norm = zeros(Float64, ntime);

plt3 = Vector{Plots.Plot}(undef,3+ceil(Int64, tf/10));

# Begin time loop
let
  t = 0.0
  X₀¹ = vcat(eltocols(vec(𝐔.(xy₁))), eltocols(vec(𝐏.(xy₁))), eltocols(vec(𝐕.(xy₁))), eltocols(vec(𝐖.(xy₁))), eltocols(vec(𝐐.(xy₁))), eltocols(vec(𝐑.(xy₁))));
  X₀² = vcat(eltocols(vec(𝐔.(xy₂))), eltocols(vec(𝐏.(xy₂))), eltocols(vec(𝐕.(xy₂))), eltocols(vec(𝐖.(xy₂))), eltocols(vec(𝐐.(xy₂))), eltocols(vec(𝐑.(xy₂))));
  X₀³ = vcat(eltocols(vec(𝐔.(xy₃))), eltocols(vec(𝐏.(xy₃))), eltocols(vec(𝐕.(xy₃))), eltocols(vec(𝐖.(xy₃))), eltocols(vec(𝐐.(xy₃))), eltocols(vec(𝐑.(xy₃))));
  X₀⁴ = vcat(eltocols(vec(𝐔.(xy₄))), eltocols(vec(𝐏.(xy₄))), eltocols(vec(𝐕.(xy₄))), eltocols(vec(𝐖.(xy₄))), eltocols(vec(𝐐.(xy₄))), eltocols(vec(𝐑.(xy₄))));

  X₀ = vcat(X₀¹, X₀², X₀³, X₀⁴)
  k₁ = zeros(Float64, length(X₀))
  k₂ = zeros(Float64, length(X₀))
  k₃ = zeros(Float64, length(X₀))
  k₄ = zeros(Float64, length(X₀)) 
  M = massma*stima
  count = 1;
  # @gif for i=1:ntime
  Hq = SBP4_1D(Nx).norm;
  Hr = SBP4_1D(Ny).norm;
  Hr1 = SBP4_1D(Ny1).norm;
  Hqr = Hq ⊗ Hr
  Hqr1 = Hq ⊗ Hr1
  function 𝐅(t, xy, Z2) 
    Z, Z1 = Z2
    xy₁, xy₂, xy₃, xy₄ = xy    
    [Z; eltocols(f.(Ref(t), vec(xy₁), Ref((0.5*h, 0.5*h, 1000)))); Z; Z; Z; Z;
     Z; eltocols(f.(Ref(t), vec(xy₂), Ref((0.5*h, 0.5*h, 1000)))); Z; Z; Z; Z;
     Z; eltocols(f.(Ref(t), vec(xy₃), Ref((0.5*h, 0.5*h, 1000)))); Z; Z; Z; Z;
     Z1; eltocols(f.(Ref(t), vec(xy₄), Ref((0.5*h, 0.5*h, 1000)))); Z1; Z1; Z1; Z1]
  end
  xys =  xy₁, xy₂, xy₃, xy₄
  Z = zeros(2*length(xy₁))
  Z1 = zeros(2*length(xy₄))
  for i=1:ntime    
    # # This block is for the moment-source function
    Fs = (𝐅((i-1)*Δt, xys, (Z,Z1)), 𝐅((i-0.5)Δt, xys, (Z,Z1)), 𝐅(i*Δt, xys, (Z,Z1)))
    X₀ = RK4_1!(M, (X₀, k₁, k₂, k₃, k₄), Δt, Fs, massma)  
    t += Δt    
    (i%ceil(Int64,ntime/20)==0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(X₀)))

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    #  Extract the displacement field from the raw solution vector
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    u1ref₁,u2ref₁ = split_solution(X₀[1:12*(Nx*Ny)], (Nx,Ny), 12);
    u1ref₂,u2ref₂ = split_solution(X₀[12*(Nx*Ny)+1:12*(Nx*Ny + Nx*Ny)], (Nx,Ny), 12);
    u1ref₃,u2ref₃ = split_solution(X₀[12*(Nx*Ny + Nx*Ny)+1:12*(Nx*Ny + Nx*Ny + Nx*Ny)], (Nx,Ny), 12);
    u1ref₄,u2ref₄ = split_solution(X₀[12*(Nx*Ny + Nx*Ny + Nx*Ny)+1:12*(Nx*Ny + Nx*Ny + Nx*Ny + Nx*Ny1)], (Nx,Ny1), 12);
    
    U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)
    U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)
    U3 = sqrt.(u1ref₃.^2 + u2ref₃.^2)
    U4 = sqrt.(u1ref₄.^2 + u2ref₄.^2)
    
    if((i==ceil(Int64, 3/Δt)) || (i == ceil(Int64, 5/Δt)) || (i == ceil(Int64, 9/Δt)) || ((i*Δt)%10 ≈ 0.0))
      plt3[count] = Plots.contourf(getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₂), getY.(xy₂), reshape(U2,size(xy₂)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₃), getY.(xy₃), reshape(U3,size(xy₃)...), colormap=:jet)
      Plots.contourf!(plt3[count], getX.(xy₄), getY.(xy₄), reshape(U4,size(xy₄)...), colormap=:jet)
      Plots.vline!(plt3[count], [L], label="\$ x \\ge "*string(round(L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.vline!(plt3[count], [0], label="\$ x \\ge "*string(round(0, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.hline!(plt3[count], [-L], label="\$ y \\ge "*string(round(-L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
      Plots.plot!(plt3[count], getX.(interface₁.(LinRange(0,1,100))), getY.(interface₁.(LinRange(0,1,100))), label="Interface 1", lc=:red, lw=2, legend=:none)
      Plots.plot!(plt3[count], getX.(interface₂.(LinRange(0,1,100))), getY.(interface₂.(LinRange(0,1,100))), label="Interface 2", lc=:red, lw=2, legend=:none)
      Plots.plot!(plt3[count], getX.(interface₃.(LinRange(0,1,100))), getY.(interface₃.(LinRange(0,1,100))), label="Interface 3", lc=:red, lw=2,  aspect_ratio=1.09, legend=:none)
      xlims!(plt3[count], (0-δ,L+δ))
      ylims!(plt3[count], (-L-δ,0))
      xlabel!(plt3[count], "\$x\$")
      ylabel!(plt3[count], "\$y\$")
      count += 1
    end

    l2norm[i] = sqrt(u1ref₁'*Hqr*u1ref₁ + u2ref₁'*Hqr*u2ref₁ +
                      u1ref₂'*Hqr*u1ref₂ + u2ref₂'*Hqr*u2ref₂ + 
                      u1ref₃'*Hqr*u1ref₃ + u2ref₃'*Hqr*u2ref₃ + 
                      u1ref₄'*Hqr1*u1ref₄ + u2ref₄'*Hqr1*u2ref₄)
  end
  # end  every 10  
  global Xref = X₀
end;

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#  Extract the displacement field from the raw solution vector
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
u1ref₁,u2ref₁ = split_solution(Xref[1:12*(Nx*Ny)], (Nx,Ny), 12);
u1ref₂,u2ref₂ = split_solution(Xref[12*(Nx*Ny)+1:12*(Nx*Ny + Nx*Ny)], (Nx,Ny), 12);
u1ref₃,u2ref₃ = split_solution(Xref[12*(Nx*Ny + Nx*Ny)+1:12*(Nx*Ny + Nx*Ny + Nx*Ny)], (Nx,Ny), 12);
u1ref₄,u2ref₄ = split_solution(Xref[12*(Nx*Ny + Nx*Ny + Nx*Ny)+1:12*(Nx*Ny + Nx*Ny + Nx*Ny + Nx*Ny1)], (Nx,Ny1), 12);

U1 = sqrt.(u1ref₁.^2 + u2ref₁.^2)*sqrt(0.5)
U2 = sqrt.(u1ref₂.^2 + u2ref₂.^2)*sqrt(0.5)
U3 = sqrt.(u1ref₃.^2 + u2ref₃.^2)*sqrt(0.5)
U4 = sqrt.(u1ref₄.^2 + u2ref₄.^2)*sqrt(0.5)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Plot the absolute value of the displacement fields
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
plt3_1 = Plots.plot();
Plots.contourf!(plt3_1, getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₂), getY.(xy₂), reshape(U2, size(xy₂)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₃), getY.(xy₃), reshape(U3,size(xy₃)...), colormap=:jet)
Plots.contourf!(plt3_1, getX.(xy₄), getY.(xy₄), reshape(U4,size(xy₄)...), colormap=:jet)
Plots.vline!(plt3_1, [L], label="\$ x \\ge "*string(round(L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.vline!(plt3_1, [0], label="\$ x \\ge "*string(round(0, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.hline!(plt3_1, [-L], label="\$ y \\ge "*string(round(-L, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
Plots.plot!(plt3_1, getX.(interface₁.(LinRange(0,1,100))), getY.(interface₁.(LinRange(0,1,100))), label="Interface 1", lc=:red, lw=2, legend=:none)
Plots.plot!(plt3_1, getX.(interface₂.(LinRange(0,1,100))), getY.(interface₂.(LinRange(0,1,100))), label="Interface 2", lc=:red, lw=2, legend=:none)
Plots.plot!(plt3_1, getX.(interface₃.(LinRange(0,1,100))), getY.(interface₃.(LinRange(0,1,100))), label="Interface 3", lc=:red, lw=2, legend=:none, aspect_ratio=1.09)
xlims!(plt3_1, (0-δ,L+δ))
ylims!(plt3_1, (-L-δ,0.0))
xlabel!(plt3_1, "\$x\$")
ylabel!(plt3_1, "\$y\$")
# c_ticks = (LinRange(2.5e-6,1.0e-5,5), string.(round.(LinRange(1.01,7.01,5), digits=4)).*"\$ \\times 10^{-7}\$");
# Plots.plot!(plt3_1, colorbar_ticks=c_ticks)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Plot the l2norm of the displacement as a function of time
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
plt5 = Plots.plot(LinRange(0,tf,ntime), l2norm, label="", lw=1, yaxis=:log10)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| \\bf{u} \\|_{H} \$")
# Plots.xlims!(plt5, (0,1000))
=#