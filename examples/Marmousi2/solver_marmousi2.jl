include("./geometry_properties.jl");
# include("./2_layer_PML_SBP_functions.jl");
include("../pml_stiffness_mass_matrices.jl");
include("../elastic_wave_operators.jl");

cp₂ = maximum(vp₂); cs₂ = maximum(vs₂);

##### ##### ##### ##### ##### ##### ##### 
# Parameters for PML damping
##### ##### ##### ##### ##### ##### ##### 
const Lₕ = x₂[end] - x₂[1]
const δ = 0.1*(Lₕ)
const σ₀ᵛ = 4*(cp₂)/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ (0.9*Lₕ)) || x[1] > (0.9*Lₕ))
    return σ₀ᵛ*((x[1] - 0.9*Lₕ)/δ)^3      
  elseif((x[1] ≈ (0.1*Lₕ)) || x[1] < (0.1*Lₕ))
    return σ₀ᵛ*((0.1*Lₕ - x[1])/δ)^3      
  else
    return 0.0
  end
end

"""
Horizontal PML strip
"""
function σₕ(x)
  0.0
end

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the PML component of the material properties
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
Pᴾᴹᴸ₁ = [@SMatrix [C₁₁¹[i,j]*(σₕ(Ω₁(qr₁[i,j])) - σᵥ(Ω₁(qr₁[i,j]))) 0 0 0; 
                   0 C₃₃¹[i,j]*(σₕ(Ω₁(qr₁[i,j])) - σᵥ(Ω₁(qr₁[i,j]))) 0 0; 
                   0 0 C₃₃¹[i,j]*(σᵥ(Ω₁(qr₁[i,j])) - σₕ(Ω₁(qr₁[i,j]))) 0; 
                   0 0 0 C₂₂¹[i,j]*(σᵥ(Ω₁(qr₁[i,j])) - σₕ(Ω₁(qr₁[i,j])))] 
                   for i=1:N₁, j=1:M₁];
Pᴾᴹᴸ₂ = [@SMatrix [C₁₁²[i,j]*(σₕ(Ω₂(qr₂[i,j])) - σᵥ(Ω₂(qr₂[i,j]))) 0 0 0;
                   0 C₃₃²[i,j]*(σₕ(Ω₂(qr₂[i,j])) - σᵥ(Ω₂(qr₂[i,j]))) 0 0; 
                   0 0 C₃₃²[i,j]*(σᵥ(Ω₂(qr₂[i,j])) - σₕ(Ω₂(qr₂[i,j]))) 0; 
                   0 0 0 C₂₂²[i,j]*(σᵥ(Ω₂(qr₂[i,j])) - σₕ(Ω₂(qr₂[i,j])))] 
                   for i=1:N₂, j=1:M₂];
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Build a dictionary to express the properties as a function to build the linear system
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
pml_material_properties_on_layer_1 = Dict(XZ₁ .=> Pᴾᴹᴸ₁);
pml_material_properties_on_layer_2 = Dict(XZ₂ .=> Pᴾᴹᴸ₂);
𝒫₁ᴾᴹᴸ(x) = pml_material_properties_on_layer_1[x]
𝒫₂ᴾᴹᴸ(x) = pml_material_properties_on_layer_2[x]                

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the PML stiffness and mass matrices
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
𝒫 = 𝒫₁, 𝒫₂
𝒫ᴾᴹᴸ = 𝒫₁ᴾᴹᴸ, 𝒫₂ᴾᴹᴸ
Z₁₂ = (Z₁¹, Z₂¹), (Z₁², Z₂²)
σₕσᵥ = σₕ, σᵥ
ρ = ρ₁, ρ₂
h = norm(XZ₂[end,1]-XZ₂[end-1,1])
stima = marmousi_two_layer_elasticity_pml_stiffness_matrix((domain₁,domain₂), (qr₁,qr₂), (𝒫, 𝒫ᴾᴹᴸ, Z₁₂, σₕσᵥ, ρ, α), 300/h);
massma = two_layer_elasticity_pml_mass_matrix((domain₁,domain₂), (qr₁,qr₂), (ρ₁, ρ₂));

##### ##### ##### ##### ##### ##### ##### 
# Zero initial conditions
##### ##### ##### ##### ##### ##### ##### 
𝐔(x) = @SVector [0.0, 0.0]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

"""
Explosive moment tensor point source
"""
function f(t::Float64, x::SVector{2,Float64}, params)
  s₁, s₂, M₀, pos_x, pos_y = params
  @assert length(pos_x) == length(pos_y)
  res = @SVector [0.0, 0.0]
  for i=1:lastindex(pos_x)
    res += @SVector[-1/(2π*√(s₁*s₂))*exp(-(x[1]-pos_x[i]*(16.9864))^2/(2s₁) - (x[2]-(pos_y[i])*(-3.4972))^2/(2s₂))*(x[1]-pos_x[i]*(16.9864))/s₁*exp(-(t-0.215)^2/0.15)*M₀,
                    -1/(2π*√(s₁*s₂))*exp(-(x[1]-pos_x[i]*(16.9864))^2/(2s₁) - (x[2]-(pos_y[i])*(-3.4972))^2/(2s₂))*(x[2]-pos_y[i]*(-3.4972))/s₂*exp(-(t-0.215)^2/0.15)*M₀]
  end
  res
end

##### ##### ##### ##### ##### ##### 
# Time stepping parameters
##### ##### ##### ##### ##### ##### 
Δt = 0.2*h/sqrt(cp₂^2+cs₂^2);
tf = 1.0
ntime = ceil(Int, tf/Δt)
params = (0.5*norm(XZ₁[1,1] - XZ₁[1,2]), 0.5*norm(XZ₁[1,1] - XZ₁[2,1]), 1000, (0.15, 0.5, 0.85), (0.3, 0.3, 0.3))

##### ##### ##### #####
# Plotting handles
##### ##### ##### #####
nplots = 20
ntime_plot = ceil(Int64, ntime/nplots);
plt3 = Vector{Plots.Plot}(undef,nplots-1);

##### ##### ##### ##### 
# Begin time stepping
##### ##### ##### ##### 
let
  t = 0.0
  W₀ = vcat(eltocols(vec(𝐔.(XZ₁))), eltocols(vec(𝐏.(XZ₁))), eltocols(vec(𝐕.(XZ₁))), eltocols(vec(𝐖.(XZ₁))), eltocols(vec(𝐐.(XZ₁))), eltocols(vec(𝐑.(XZ₁))))
  X₀ = vcat(eltocols(vec(𝐔.(XZ₂))), eltocols(vec(𝐏.(XZ₂))), eltocols(vec(𝐕.(XZ₂))), eltocols(vec(𝐖.(XZ₂))), eltocols(vec(𝐐.(XZ₂))), eltocols(vec(𝐑.(XZ₂))))  
  global Z₀ = vcat(W₀, X₀)
  global l2norm = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma*stima
  count = 1  
  function 𝐅(t, xy, Z)  
    xy₁, xy₂ = xy    
    Z₁, Z₂ = Z
    [Z₁; eltocols(f.(Ref(t), vec(xy₁), Ref(params))); Z₁; Z₁; Z₁; Z₁;
     Z₂; eltocols(f.(Ref(t), vec(xy₂), Ref(params))); Z₂; Z₂; Z₂; Z₂]
  end
  # @gif for i=1:ntime
  xys =  XZ₁, XZ₂
  XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
  XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 
  Z = zeros(2*length(XZ₁)), zeros(2*length(XZ₂))
  Hq₁ = SBP4_1D(N₁).norm;  Hr₁ = SBP4_1D(M₁).norm;
  Hq₂ = SBP4_1D(N₂).norm;  Hr₂ = SBP4_1D(M₂).norm;  
  Hqr₁ = Hq₁ ⊗ Hr₁; 
  Hqr₂ = Hq₂ ⊗ Hr₂  
  for i=1:ntime    
    ##### ##### ##### ##### 
    # RK4 time stepping
    ##### ##### ##### ##### 
    Z₀ = RK4_1!(M, (Z₀, k₁, k₂, k₃, k₄), Δt, (𝐅(t, xys, Z), 𝐅(t+0.5Δt, xys, Z), 𝐅(t+Δt, xys, Z)), massma)            
    t += Δt        
    (i%ntime_plot == 0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    ##### ##### ##### ##### ##### ##### ##### ##### 
    # Extract the displacements from the raw vector
    ##### ##### ##### ##### ##### ##### ##### ##### 
    u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(M₁*N₁)], (M₁,N₁), 12);
    u1ref₂,u2ref₂ = split_solution(Z₀[12*(M₁*N₁)+1:12*(M₁*N₁ + M₂*N₂)], (M₂,N₂), 12);    
    absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2));
    absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2));    

    ##### ##### ##### ##### ##### ##### 
    # Plot at every t=T/20 intervals
    ##### ##### ##### ##### ##### ##### 
    if(ceil(i%ntime_plot) == 0.0)      
      plt3[count] = Plots.plot()        
      Plots.contourf!(plt3[count], XC₁, ZC₁, reshape((absu1), size(XC₁)...), label="", colormap=:jet)
      Plots.contourf!(plt3[count], XC₂, ZC₂, reshape((absu2), size(XC₂)...), label="", colormap=:jet)      
      Plots.plot!(plt3[count], [0,x₂[end]],[-3.34,-2.47], lw=2, lc=:white, label="")
      Plots.plot!(plt3[count], [0,x₂[end]],[z₂[1],z₂[1]], lw=2, lc=:white, label="")
      Plots.vline!(plt3[count], [(x₂[1]+0.9*Lₕ)], lw=1, lc=:white, ls=:dash, label="")
      Plots.vline!(plt3[count], [(x₂[1]+0.1*Lₕ)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(900,200))            
      Plots.xlims!(plt3[count], (0.0,x₂[end]))
      Plots.ylims!(plt3[count], (z₂[1],z₂[end]))
      Plots.xlabel!(plt3[count], "\$x\$")
      Plots.ylabel!(plt3[count], "\$y\$")      
      count += 1
    end

    l2norm[i] = sqrt(u1ref₁'*Hqr₁*u1ref₁ + u2ref₁'*Hqr₁*u2ref₁ +
                      u1ref₂'*Hqr₂*u1ref₂ + u2ref₂'*Hqr₂*u2ref₂)
  end
  # end every 10
end  

u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(M₁*N₁)], (M₁,N₁), 12);
u1ref₂,u2ref₂ = split_solution(Z₀[12*(M₁*N₁)+1:12*(M₁*N₁ + M₂*N₂)], (M₂,N₂), 12);    
absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;

# Get the x-and-y coordinates separately
XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 

# 
plt3_1 = Plots.plot()
Plots.contourf!(plt3_1, XC₁, ZC₁, reshape((absu1), size(XC₁)...), label="", colormap=:jet)
Plots.contourf!(plt3_1, XC₂, ZC₂, reshape((absu2), size(XC₂)...), label="", colormap=:jet)      
Plots.plot!(plt3_1, [0,x₂[end]],[-3.34,-2.47], lw=2, lc=:white, label="")
Plots.plot!(plt3_1, [0,x₂[end]],[z₂[1],z₂[1]], lw=2, lc=:white, label="")
Plots.vline!(plt3_1, [(x₂[1]+0.9*Lₕ)], lw=1, lc=:white, ls=:dash, label="")
Plots.vline!(plt3_1, [(x₂[1]+0.1*Lₕ)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(900,200))            
Plots.xlims!(plt3_1, (0.0,x₂[end]))
Plots.ylims!(plt3_1, (z₂[1],z₂[end]))
Plots.xlabel!(plt3_1, "\$x\$")
Plots.ylabel!(plt3_1, "\$y\$") 

plt4 = Plots.contourf(X₂, Z₂, vp₂, label="", colormap=:jet)
# Plots.contourf!(plt4, X₂, Z₂, vp₂, label="", colormap=:jet)
Plots.xlims!(plt4, (x₂[1],x₂[end]))
Plots.ylims!(plt4, (z₂[1],z₂[end]))
Plots.xlabel!(plt4, "\$x\$")
Plots.ylabel!(plt4, "\$y\$")

# scalefontsizes(3)
plt5 = Plots.plot(LinRange(0,tf,ntime), maxvals, label="", lw=2)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| \\mathbf{u} \\|_{\\mathbf{H}} \$")