include("./3_layer_geometry_properties.jl");
include("./3_layer_PML_SBP_functions.jl");

cp₁ = maximum(vp₁); cs₁ = maximum(vs₁);
cp₂ = maximum(vp₂); cs₂ = maximum(vs₂);

"""
The PML damping
"""
const Lᵥ = abs(z₂[1]-z₁[end])
const Lₕ = x₁[end] - x₁[1]
const δ = 0.1*(Lₕ)
const σ₀ᵛ = 0*(max(cp₁, cp₂))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*(max(cp₁, cp₂))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ (x₁[1]+0.9*Lₕ)) || x[1] > (x₁[1]+0.9*Lₕ))
    return σ₀ᵛ*((x[1] - x₁[1] - 0.9*Lₕ)/δ)^3  
    # return 0.5*σ₀ᵛ*(1 + tanh((x[1] - x₁[1] - 0.9*Lₕ)))
  elseif((x[1] ≈ (x₁[1]+0.1*Lₕ)) || x[1] < (x₁[1]+0.1*Lₕ))
    return σ₀ᵛ*((x₁[1] + 0.1*Lₕ - x[1])/δ)^3  
    # return 0.5*σ₀ᵛ*(1 + tanh((x₁[1] + 0.1*Lₕ - x[1])))
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
Pᴾᴹᴸ₃ = [@SMatrix [C₁₁³[i,j]*(σₕ(Ω₃(𝐪𝐫₃[i,j])) - σᵥ(Ω₃(𝐪𝐫₃[i,j]))) 0 0 0; 
                   0 C₃₃³[i,j]*(σₕ(Ω₃(𝐪𝐫₃[i,j])) - σᵥ(Ω₃(𝐪𝐫₃[i,j]))) 0 0; 
                   0 0 C₃₃³[i,j]*(σᵥ(Ω₃(𝐪𝐫₃[i,j])) - σₕ(Ω₃(𝐪𝐫₃[i,j]))) 0; 
                   0 0 0 C₂₂³[i,j]*(σᵥ(Ω₃(𝐪𝐫₃[i,j])) - σₕ(Ω₃(𝐪𝐫₃[i,j])))] 
                   for i=1:M₃, j=1:N₃]

                  
ℙ₁ = [Pt(P₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:M₁, j=1:N₁];
ℙ₂ = [Pt(P₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:M₂, j=1:N₂];
ℙ₃ = [Pt(P₃[i,j], Ω₃, 𝐪𝐫₃[i,j]) for i=1:M₃, j=1:N₃];
ℙᴾᴹᴸ₁ = [Ptᴾᴹᴸ(Pᴾᴹᴸ₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:M₁, j=1:N₁];
ℙᴾᴹᴸ₂ = [Ptᴾᴹᴸ(Pᴾᴹᴸ₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:M₂, j=1:N₂];
ℙᴾᴹᴸ₃ = [Ptᴾᴹᴸ(Pᴾᴹᴸ₃[i,j], Ω₃, 𝐪𝐫₃[i,j]) for i=1:M₃, j=1:N₃];

stima = 𝐊3ₚₘₗ((ℙ₁,ℙ₂,ℙ₃), (ℙᴾᴹᴸ₁, ℙᴾᴹᴸ₂, ℙᴾᴹᴸ₃), ((Z₁¹, Z₂¹), (Z₁², Z₂²), (Z₁³, Z₂³)), (RHO₁, RHO₂, RHO₃), (𝛀₁,𝛀₂,𝛀₃), (𝐪𝐫₁,𝐪𝐫₂,𝐪𝐫₃));
massma =  𝐌3⁻¹ₚₘₗ((𝛀₁, 𝛀₂, 𝛀₃), (𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃), (RHO₁, RHO₂, RHO₃));

# 𝐔(x) = @SVector [exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.15+x₁[1]*0.85))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.5+x₁[1]*0.5))^2 + (x[2]-(-1.5))^2)), 
#                  exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.15+x₁[1]*0.85))^2 + (x[2]-(-1.5))^2)) + exp(-20*((x[1]-(x₁[end]*0.5+x₁[1]*0.5))^2 + (x[2]-(-1.5))^2))]
𝐔(x) = @SVector [0.0, 0.0]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)a
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

const h = norm(XZ₃[end,1] - XZ₃[end-1,1]);
const Δt = 0.2*h/sqrt(max((cp₁^2+cs₁^2), (cp₂^2+cs₂^2)));
tf = 5.0
ntime = ceil(Int, tf/Δt)
params = (0.05*norm(XZ₂[1,1] - XZ₂[1,2]), 0.05*norm(XZ₂[1,1] - XZ₂[2,1]), 10, (0.15, 0.5, 0.85), (0.3, 0.3, 0.3))
ntime_plot = ceil(Int64, ntime/10);

plt3 = Vector{Plots.Plot}(undef,9);

# scalefontsizes()
let
  t = 0.0
  W₀ = vcat(eltocols(vec(𝐔.(XZ₁))), eltocols(vec(𝐏.(XZ₁))), eltocols(vec(𝐕.(XZ₁))), eltocols(vec(𝐖.(XZ₁))), eltocols(vec(𝐐.(XZ₁))), eltocols(vec(𝐑.(XZ₁))))
  X₀ = vcat(eltocols(vec(𝐔.(XZ₂))), eltocols(vec(𝐏.(XZ₂))), eltocols(vec(𝐕.(XZ₂))), eltocols(vec(𝐖.(XZ₂))), eltocols(vec(𝐐.(XZ₂))), eltocols(vec(𝐑.(XZ₂))))
  Y₀ = vcat(eltocols(vec(𝐔.(XZ₃))), eltocols(vec(𝐏.(XZ₃))), eltocols(vec(𝐕.(XZ₃))), eltocols(vec(𝐖.(XZ₃))), eltocols(vec(𝐐.(XZ₃))), eltocols(vec(𝐑.(XZ₃))))
  global Z₀ = vcat(W₀, X₀, Y₀)
  global maxvals = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma*stima
  count = 1  
  function 𝐅(t, xy, Z)  
    xy₁, xy₂, xy₃ = xy    
    Z₁, Z₂, Z₃ = Z
    [Z₁; eltocols(f.(Ref(t), vec(xy₁), Ref(params))); Z₁; Z₁; Z₁; Z₁;
     Z₂; eltocols(f.(Ref(t), vec(xy₂), Ref(params))); Z₂; Z₂; Z₂; Z₂;
     Z₃; eltocols(f.(Ref(t), vec(xy₃), Ref(params))); Z₃; Z₃; Z₃; Z₃]
  end
  # @gif for i=1:ntime
  xys =  XZ₁, XZ₂, XZ₃
  Z = zeros(2*length(XZ₁)),zeros(2*length(XZ₂)),zeros(2*length(XZ₃))
  Hq₁ = SBP_1_2_CONSTANT_0_1(N₁).norm;  Hr₁ = SBP_1_2_CONSTANT_0_1(M₁).norm;
  Hq₂ = SBP_1_2_CONSTANT_0_1(N₂).norm;  Hr₂ = SBP_1_2_CONSTANT_0_1(M₂).norm;
  Hq₃ = SBP_1_2_CONSTANT_0_1(N₃).norm;  Hr₃ = SBP_1_2_CONSTANT_0_1(M₃).norm;
  Hqr₁ = Hq₁ ⊗ Hr₁
  Hqr₂ = Hq₂ ⊗ Hr₂
  Hqr₃ = Hq₃ ⊗ Hr₃
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    # Z₀ = RK4_1!(Δt, M, sol)    
    Fs = (𝐅(t, xys, Z), 𝐅(t+0.5Δt, xys, Z), 𝐅(t+Δt, xys, Z))
    Z₀ = RK4_1!(M, sol, Δt, Fs, massma)        
    #Z₀ = massma*Z₀
    t += Δt        
    (i%10 == 0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
    u1ref₂,u2ref₂ = split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
    u1ref₃,u2ref₃ = split_solution(Z₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
    absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
    absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;
    absu3 = sqrt.((u1ref₃.^2) + (u2ref₃.^2)) ;

    # Add code to plot to generate the GIFs
    # if((i == ceil(Int64, 0.1/Δt)) || (i == ceil(Int64, 0.2/Δt)) || (i == ceil(Int64, 0.3/Δt)) ||  (i == ceil(Int64, 0.4/Δt)) ||  (i == ceil(Int64, 0.5/Δt)) ||  (i == ceil(Int64, 0.6/Δt)) ||  (i == ceil(Int64, 0.7/Δt)) ||  (i == ceil(Int64, 0.8/Δt))) 
    if(ceil(i%ntime_plot) == 0.0)
      XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
      XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 
      XC₃ = getX.(XZ₃); ZC₃ = getY.(XZ₃)
      plt3[count] = Plots.plot()
      Plots.contourf!(plt3[count], XC₁, ZC₁, reshape((absu1), size(XC₁)...), label="", colormap=:jet)
      Plots.contourf!(plt3[count], XC₂, ZC₂, reshape((absu2), size(XC₂)...), label="", colormap=:jet)
      Plots.contourf!(plt3[count], XC₃, ZC₃, reshape((absu3), size(XC₃)...), label="", colormap=:jet)
      # Plots.annotate!(plt3[count], 10, -0.2, ("Layer 1", 10, :white))
      # Plots.annotate!(plt3[count], 10, -1.8, ("Layer 2", 10, :white))
      # Plots.annotate!(plt3[count], 14, -3.2, ("Layer 3", 10, :white))
      # Plots.annotate!(plt3[count], 16.2, -2, ("\$ \\sigma_0^v = 8\$", 10, :white))
      Plots.plot!(plt3[count], [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:white, label="")
      Plots.plot!(plt3[count], [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:white, label="")
      Plots.vline!(plt3[count], [(x₁[1]+0.9*Lₕ)], lw=1, lc=:white, ls=:dash, label="")
      Plots.vline!(plt3[count], [(x₁[1]+0.1*Lₕ)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(900,200))      
      Plots.xlims!(plt3[count], (0.0,x₁[end]))
      Plots.ylims!(plt3[count], (z₂[1],z₁[end]))
      Plots.xlabel!(plt3[count], "\$x\$ (in km)")
      Plots.ylabel!(plt3[count], "\$z\$ (in km)")
      count+=1
    end

    maxvals[i] = sqrt(u1ref₁'*Hqr₁*u1ref₁ + u2ref₁'*Hqr₁*u2ref₁ +
                      u1ref₂'*Hqr₂*u1ref₂ + u2ref₂'*Hqr₂*u2ref₂ + 
                      u1ref₃'*Hqr₃*u1ref₃ + u2ref₃'*Hqr₃*u2ref₃)
  end
  # end every 10
end  

u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
u1ref₃,u2ref₃ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;
absu3 = sqrt.((u1ref₃.^2) + (u2ref₃.^2)) ;

# Get the x-and-y coordinates separately
XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 
XC₃ = getX.(XZ₃); ZC₃ = getY.(XZ₃)

# scalefontsizes()

plt3_1 = Plots.plot();
# 
Plots.contourf!(plt3_1, XC₁, ZC₁, reshape((absu1), size(XC₁)...), label="", colormap=:jet)
Plots.contourf!(plt3_1, XC₂, ZC₂, reshape((absu2), size(XC₂)...), label="", colormap=:jet)
Plots.contourf!(plt3_1, XC₃, ZC₃, reshape((absu3), size(XC₃)...), label="", colormap=:jet)
Plots.plot!(plt3_1, [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:white, label="")
Plots.plot!(plt3_1, [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:white, label="")
Plots.vline!(plt3_1, [(x₁[1]+0.9*Lₕ)], lw=1, lc=:white, ls=:dash, label="")
Plots.vline!(plt3_1, [(x₁[1]+0.1*Lₕ)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(600,200))
# Plots.vspan!(plt3_1, [(x₁[1]+0.9*Lₕ),x₁[end]], fillalpha=0.5, fillcolor=:orange, label="")
Plots.xlims!(plt3_1, (x₁[1],x₁[end]))
Plots.ylims!(plt3_1, (z₂[1],z₁[end]))
Plots.xlabel!(plt3_1, "\$x\$ (in km)")
Plots.ylabel!(plt3_1, "\$z\$ (in km)")

plt4 = Plots.contourf(X₂, Z₂, vp₂, label="", colormap=:jet)
Plots.contourf!(plt4, X₁, Z₁, vp₁, label="", colormap=:jet, size=(600,200))
Plots.xlims!(plt4, (x₁[1],x₁[end]))
Plots.ylims!(plt4, (z₂[1],z₁[end]))
Plots.xlabel!(plt4, "\$x\$ (in km)")
Plots.ylabel!(plt4, "\$z\$ (in km)")

# scalefontsizes(3)
plt5 = Plots.plot(LinRange(0,tf,ntime), maxvals, label="", lw=2)
Plots.xlabel!(plt5, "Time \$t\$")
Plots.ylabel!(plt5, "\$ \\| u \\|_{\\mathbf{H}} \$")

plt6 = Plots.contour(XC₂, ZC₂, σᵥ.(XZ₂), label="", colormap=:jet)
Plots.contour!(plt6, XC₁, ZC₁, σᵥ.(XZ₁), label="", colormap=:jet)
Plots.contour!(plt6, XC₃, ZC₃, σᵥ.(XZ₃), label="", colormap=:jet)
Plots.annotate!(plt6, 10, -0.2, ("Layer 1", 15, :black))
Plots.annotate!(plt6, 10, -1.8, ("Layer 2", 15, :black))
Plots.annotate!(plt6, 14, -3.2, ("Layer 3", 15, :black))
Plots.annotate!(plt6, 16.2, -2, ("PML", 15, :black, :bold))
Plots.plot!(plt6, [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:black, label="")
Plots.plot!(plt6, [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:black, label="")
Plots.vline!(plt6, [(x₁[1]+0.9*Lₕ)], lw=1, lc=:black, ls=:dash, label="")
Plots.vline!(plt6, [(x₁[1]+0.1*Lₕ)], lw=1, lc=:black, ls=:dash, label="", legend=:topleft, size=(900,300))
# Plots.vspan!(plt3, [(x₁[1]+0.9*Lₕ),x₁[end]], fillalpha=0.5, fillcolor=:orange, label="")
Plots.xlims!(plt6, (x₁[1],x₁[end]))
Plots.ylims!(plt6, (z₂[1],z₁[end]))
Plots.xlabel!(plt6, "\$x\$ (in km)")
Plots.ylabel!(plt6, "\$z\$ (in km)")