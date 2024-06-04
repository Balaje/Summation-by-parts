include("./2_layer_geometry_properties.jl");
include("./2_layer_PML_SBP_functions.jl");

cp₂ = maximum(vp₂); cs₂ = maximum(vs₂);

"""
The PML damping
"""
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

Pᴾᴹᴸ₁ = [@SMatrix [C₁₁¹[i,j]*(σₕ(Ω₁(𝐪𝐫₁[i,j])) - σᵥ(Ω₁(𝐪𝐫₁[i,j]))) 0 0 0; 
                   0 C₃₃¹[i,j]*(σₕ(Ω₁(𝐪𝐫₁[i,j])) - σᵥ(Ω₁(𝐪𝐫₁[i,j]))) 0 0; 
                   0 0 C₃₃¹[i,j]*(σᵥ(Ω₁(𝐪𝐫₁[i,j])) - σₕ(Ω₁(𝐪𝐫₁[i,j]))) 0; 
                   0 0 0 C₂₂¹[i,j]*(σᵥ(Ω₁(𝐪𝐫₁[i,j])) - σₕ(Ω₁(𝐪𝐫₁[i,j])))] 
                   for i=1:M₁, j=1:N₁]
Pᴾᴹᴸ₂ = [@SMatrix [C₁₁²[i,j]*(σₕ(Ω₂(𝐪𝐫₂[i,j])) - σᵥ(Ω₂(𝐪𝐫₂[i,j]))) 0 0 0; 
                   0 C₃₃²[i,j]*(σₕ(Ω₂(𝐪𝐫₂[i,j])) - σᵥ(Ω₂(𝐪𝐫₂[i,j]))) 0 0; 
                   0 0 C₃₃²[i,j]*(σᵥ(Ω₂(𝐪𝐫₂[i,j])) - σₕ(Ω₂(𝐪𝐫₂[i,j]))) 0; 
                   0 0 0 C₂₂²[i,j]*(σᵥ(Ω₂(𝐪𝐫₂[i,j])) - σₕ(Ω₂(𝐪𝐫₂[i,j])))] 
                   for i=1:M₂, j=1:N₂]
                  
ℙ₁ = [Pt(P₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:M₁, j=1:N₁];
ℙ₂ = [Pt(P₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:M₂, j=1:N₂];

ℙᴾᴹᴸ₁ = [Ptᴾᴹᴸ(Pᴾᴹᴸ₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:M₁, j=1:N₁];
ℙᴾᴹᴸ₂ = [Ptᴾᴹᴸ(Pᴾᴹᴸ₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:M₂, j=1:N₂];

stima = 𝐊2ₚₘₗ((ℙ₁,ℙ₂), (ℙᴾᴹᴸ₁, ℙᴾᴹᴸ₂), ((Z₁¹, Z₂¹), (Z₁², Z₂²)), (RHO₁, RHO₂), (𝛀₁,𝛀₂), (𝐪𝐫₁,𝐪𝐫₂));
massma =  𝐌2⁻¹ₚₘₗ((𝛀₁, 𝛀₂), (𝐪𝐫₁, 𝐪𝐫₂), (RHO₁, RHO₂));

# 𝐔(x) = @SVector [exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.15+x₁[1]*0.85))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.5+x₁[1]*0.5))^2 + (x[2]-(-1.5))^2)), 
#                  exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.15+x₁[1]*0.85))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.5+x₁[1]*0.5))^2 + (x[2]-(-1.5))^2))]
𝐔(x) = @SVector [0.0, 0.0]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)a
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

h = norm(XZ₂[end,1] - XZ₂[end-1,1]);
Δt = 0.2*h/sqrt(cp₂^2+cs₂^2);
tf = 10
ntime = ceil(Int, tf/Δt)
params = (0.5*norm(XZ₁[1,1] - XZ₁[1,2]), 0.5*norm(XZ₁[1,1] - XZ₁[2,1]), 1000, (0.15, 0.5, 0.85), (0.3, 0.3, 0.3))
nplots = 20
ntime_plot = ceil(Int64, ntime/nplots);

plt3 = Vector{Plots.Plot}(undef,nplots-1);

# scalefontsizes()
let
  t = 0.0
  W₀ = vcat(eltocols(vec(𝐔.(XZ₁))), eltocols(vec(𝐏.(XZ₁))), eltocols(vec(𝐕.(XZ₁))), eltocols(vec(𝐖.(XZ₁))), eltocols(vec(𝐐.(XZ₁))), eltocols(vec(𝐑.(XZ₁))))
  X₀ = vcat(eltocols(vec(𝐔.(XZ₂))), eltocols(vec(𝐏.(XZ₂))), eltocols(vec(𝐕.(XZ₂))), eltocols(vec(𝐖.(XZ₂))), eltocols(vec(𝐐.(XZ₂))), eltocols(vec(𝐑.(XZ₂))))  
  global Z₀ = vcat(W₀, X₀)
  # t = tf
  # global Z₀ = Z₀
  global maxvals = zeros(Float64, ntime)
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
  Z = zeros(2*length(XZ₁)), zeros(2*length(XZ₂))
  Hq₁ = SBP_1_2_CONSTANT_0_1(N₁).norm;  Hr₁ = SBP_1_2_CONSTANT_0_1(M₁).norm;
  Hq₂ = SBP_1_2_CONSTANT_0_1(N₂).norm;  Hr₂ = SBP_1_2_CONSTANT_0_1(M₂).norm;  
  Hqr₁ = Hq₁ ⊗ Hr₁; Hqr₂ = Hq₂ ⊗ Hr₂  
  XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
  XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄    
    Fs = (𝐅(t, xys, Z), 𝐅(t+0.5Δt, xys, Z), 𝐅(t+Δt, xys, Z))
    Z₀ = RK4_1!(M, sol, Δt, Fs, massma)            
    t += Δt        
    (i%ntime_plot == 0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
    u1ref₂,u2ref₂ = split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);    
    absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2));
    absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2));    

    # Add code to plot to generate the GIFs    
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
      count+=1
    end

    maxvals[i] = sqrt(u1ref₁'*Hqr₁*u1ref₁ + u2ref₁'*Hqr₁*u2ref₁ +
                      u1ref₂'*Hqr₂*u1ref₂ + u2ref₂'*Hqr₂*u2ref₂)
  end
  # end every 10
end  

u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;

# Get the x-and-y coordinates separately
XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 

# scalefontsizes()

plt3_1 = Plots.plot();
# 
Plots.contourf!(plt3_1, XC₁, ZC₁, reshape((absu1), size(XC₁)...), label="", colormap=:jet)
Plots.contourf!(plt3_1, XC₂, ZC₂, reshape((absu2), size(XC₂)...), label="", colormap=:jet)
Plots.plot!(plt3_1, [0,x₂[end]],[-3.34,-2.47], lw=2, lc=:white, label="")
Plots.plot!(plt3_1, [0,x₂[end]],[z₂[1],z₂[1]], lw=2, lc=:white, label="")
Plots.vline!(plt3_1, [(x₂[1]+0.9*Lₕ)], lw=1, lc=:white, ls=:dash, label="")
Plots.vline!(plt3_1, [(x₂[1]+0.1*Lₕ)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(600,200))
# Plots.vspan!(plt3_1, [(x₁[1]+0.9*Lₕ),x₁[end]], fillalpha=0.5, fillcolor=:orange, label="")
Plots.xlims!(plt3_1, (x₂[1],x₂[end]))
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

plt6 = Plots.plot();
Plots.contour!(plt6, XC₁, ZC₁, σᵥ.(XZ₁), label="", colormap=:jet)
Plots.contour!(plt6, XC₂, ZC₂, σᵥ.(XZ₂), label="", colormap=:jet)
Plots.annotate!(plt6, 10, -0.2, ("Layer 1", 15, :black))
Plots.annotate!(plt6, 10, -1.8, ("Layer 2", 15, :black))
Plots.annotate!(plt6, 14, -3.2, ("Layer 3", 15, :black))
Plots.annotate!(plt6, 16.2, -2, ("PML", 15, :black, :bold))
Plots.plot!(plt6, [0,x₂[end]],[-3.34,-2.47], lw=2, lc=:black, label="")
Plots.plot!(plt6, [0,x₂[end]],[z₂[1],z₂[1]], lw=2, lc=:black, label="")
Plots.vline!(plt6, [(x₂[1]+0.9*Lₕ)], lw=1, lc=:black, ls=:dash, label="")
Plots.vline!(plt6, [(x₂[1]+0.1*Lₕ)], lw=1, lc=:black, ls=:dash, label="", legend=:topleft, size=(900,300))
# Plots.vspan!(plt3, [(x₁[1]+0.9*Lₕ),x₁[end]], fillalpha=0.5, fillcolor=:orange, label="")
Plots.xlims!(plt6, (x₂[1],x₂[end]))
Plots.ylims!(plt6, (z₂[1],z₂[end]))
Plots.xlabel!(plt6, "\$x\$")
Plots.ylabel!(plt6, "\$y\$")