include("./3_layer_geometry_properties.jl")
include("./3_layer_PML_SBP_functions.jl")

"""
The PML damping
"""
const Lᵥ = abs(z₂[1]-z₁[end])
const Lₕ = x₁[end] - x₁[1]
const δ = 0.1*(Lₕ)
const σ₀ᵛ = 4*(√(max(maximum(vp₁), maximum(vp₂))))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const σ₀ʰ = 0*(√(max(maximum(vp₁), maximum(vp₂))))/(2*δ)*log(10^4) #cₚ,max = 4, ρ = 1, Ref = 10^-4
const α = σ₀ᵛ*0.05; # The frequency shift parameter

"""
Vertical PML strip
"""
function σᵥ(x)
  if((x[1] ≈ (x₁[1]+0.9*Lₕ)) || x[1] > (x₁[1]+0.9*Lₕ))
    # return σ₀ᵛ*((x[1] - x₁[1] - 0.9*Lₕ)/δ)^3  
    return 0.5*σ₀ᵛ*(1 + tanh((x[1] - x₁[1] - 0.9*Lₕ)))
  elseif((x[1] ≈ (x₁[1]+0.1*Lₕ)) || x[1] < (x₁[1]+0.1*Lₕ))
    # return σ₀ᵛ*((x₁[1] + 0.1*Lₕ - x[1])/δ)^3  
    0.5*σ₀ᵛ*(1 + tanh((x₁[1] + 0.1*Lₕ - x[1])))
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
ℙᴾᴹᴸ₁ = [Pt(Pᴾᴹᴸ₁[i,j], Ω₁, 𝐪𝐫₁[i,j]) for i=1:M₁, j=1:N₁];
ℙᴾᴹᴸ₂ = [Pt(Pᴾᴹᴸ₂[i,j], Ω₂, 𝐪𝐫₂[i,j]) for i=1:M₂, j=1:N₂];
ℙᴾᴹᴸ₃ = [Pt(Pᴾᴹᴸ₃[i,j], Ω₃, 𝐪𝐫₃[i,j]) for i=1:M₃, j=1:N₃];

stima = 𝐊3ₚₘₗ((ℙ₁,ℙ₂,ℙ₃), (ℙᴾᴹᴸ₁, ℙᴾᴹᴸ₂, ℙᴾᴹᴸ₃), ((Z₁¹, Z₂¹), (Z₁², Z₂²), (Z₁³, Z₂³)), (RHO₁, RHO₂, RHO₃), (𝛀₁,𝛀₂,𝛀₃), (𝐪𝐫₁,𝐪𝐫₂,𝐪𝐫₃));
massma =  𝐌3⁻¹ₚₘₗ((𝛀₁, 𝛀₂, 𝛀₃), (𝐪𝐫₁, 𝐪𝐫₂, 𝐪𝐫₃), (RHO₁, RHO₂, RHO₃));

𝐔(x) = @SVector [20*exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2)), 
                 20*exp(-20*((x[1]-(x₁[end]*0.85+x₁[1]*0.15))^2 + (x[2]-(-1.5))^2))]
𝐏(x) = @SVector [0.0, 0.0] # = 𝐔ₜ(x)
𝐕(x) = @SVector [0.0, 0.0]
𝐖(x) = @SVector [0.0, 0.0]
𝐐(x) = @SVector [0.0, 0.0]
𝐑(x) = @SVector [0.0, 0.0]

const Δt = 1e-4
tf = 1.0
ntime = ceil(Int, tf/Δt)

let
  t = 0.0
  W₀ = vcat(eltocols(vec(𝐔.(XZ₁))), eltocols(vec(𝐏.(XZ₁))), eltocols(vec(𝐕.(XZ₁))), eltocols(vec(𝐖.(XZ₁))), eltocols(vec(𝐐.(XZ₁))), eltocols(vec(𝐑.(XZ₁))))
  X₀ = vcat(eltocols(vec(𝐔.(XZ₂))), eltocols(vec(𝐏.(XZ₂))), eltocols(vec(𝐕.(XZ₂))), eltocols(vec(𝐖.(XZ₂))), eltocols(vec(𝐐.(XZ₂))), eltocols(vec(𝐑.(XZ₂))))
  Y₀ = vcat(eltocols(vec(𝐔.(XZ₃))), eltocols(vec(𝐏.(XZ₃))), eltocols(vec(𝐕.(XZ₃))), eltocols(vec(𝐖.(XZ₃))), eltocols(vec(𝐐.(XZ₃))), eltocols(vec(𝐑.(XZ₃))))
  global Z₀ = vcat(W₀, X₀, Y₀)
  global maxvals₁ = zeros(Float64, ntime)
  global maxvals₂ = zeros(Float64, ntime)
  k₁ = zeros(Float64, length(Z₀))
  k₂ = zeros(Float64, length(Z₀))
  k₃ = zeros(Float64, length(Z₀))
  k₄ = zeros(Float64, length(Z₀)) 
  M = massma*stima
  # @gif for i=1:ntime
  for i=1:ntime
    sol = Z₀, k₁, k₂, k₃, k₄
    Z₀ = RK4_1!(M, sol)    
    t += Δt        
    (i%100 == 0) && println("Done t = "*string(t)*"\t max(sol) = "*string(maximum(Z₀)))

    # Plotting part for 
    u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
    u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
    u1ref₃,u2ref₃ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
    absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
    absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;
    absu3 = sqrt.((u1ref₃.^2) + (u2ref₃.^2)) ;

    # plt3 = scatter(Tuple.(XZ₁ |> vec), zcolor=vec(absu1), colormap=:tempo, markersize=5, msw=0, label="", markershape=:rect); 
    # scatter!(plt3, Tuple.(XZ₂ |> vec), zcolor=vec(absu2), colormap=:tempo, markersize=5, msw=0, label="", markershape=:rect);    
    # scatter!(plt3, Tuple.(XZ₃ |> vec), zcolor=vec(absu3), colormap=:tempo, markersize=5, msw=0, label="", markershape=:rect)
    # hline!(plt3, [z₁[1]], lc=:black, lw=2, label="Interface")
    # scatter!(plt3, Tuple.(cᵢ.(LinRange(0,1,100))), mc=:black, msw=0.0, ms=4)
    # vline!(plt3, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
    # vline!(plt3, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
    # xlims!(plt3, (x₁[1], x₁[end]))
    # ylims!(plt3, (z₂[1], z₁[end]))
    # title!(plt3, "\$|u(x,y)|\$ at Time t="*string(round(t,digits=4)));

    # plt4 = scatter(Tuple.(XZ₁ |> vec), zcolor=vec(RHO₁), markersize=5, msw=0, label="", markershape=:rect); 
    # scatter!(plt4, Tuple.(XZ₂ |> vec), zcolor=vec(RHO₂), markersize=5, msw=0, label="", markershape=:rect);    
    # scatter!(plt4, Tuple.(XZ₃ |> vec), zcolor=vec(RHO₃), markersize=5, msw=0, label="", markershape=:rect)
    # hline!(plt4, [z₁[1]], lc=:black, lw=2, label="Interface")
    # scatter!(plt4, Tuple.(cᵢ.(LinRange(0,1,100))), mc=:black, msw=0.0, ms=4)
    # vline!(plt4, [(x₁[1]+0.9*Lₕ)], lc=:darkgreen, lw=2, label="x ≥ Lₓ (PML)")
    # vline!(plt4, [(x₁[1]+0.1*Lₕ)], lc=:darkgreen, lw=2, label="x ≤ Lₓ (PML)")
    # xlims!(plt4, (x₁[1], x₁[end]))
    # ylims!(plt4, (z₂[1], z₁[end]))
    # title!(plt4, "p-wave speed of the material")

    # plot(plt3, plt4, layout=(2,1), size=(1200,800))

    maxvals₁[i] = sqrt(norm(u1ref₁,2)^2 + norm(u2ref₁)^2)
    maxvals₂[i] = sqrt(norm(u1ref₂,2)^2 + norm(u2ref₂)^2)
  end
  # end every 1000
end  

u1ref₁,u2ref₁ = split_solution(Z₀[1:12*(prod(𝛀₁.mn))], 𝛀₁.mn, 12);
u1ref₂,u2ref₂ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))], 𝛀₂.mn, 12);
u1ref₃,u2ref₃ =  split_solution(Z₀[12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+1:12*(prod(𝛀₁.mn))+12*(prod(𝛀₂.mn))+12*(prod(𝛀₃.mn))], 𝛀₃.mn, 12);
absu1 = sqrt.((u1ref₁.^2) + (u2ref₁.^2)) ;
absu2 = sqrt.((u1ref₂.^2) + (u2ref₂.^2)) ;
absu3 = sqrt.((u1ref₃.^2) + (u2ref₃.^2)) ;

getX(ARR) = ARR[1]
getY(ARR) = ARR[2]

XC₁ = getX.(XZ₁); ZC₁ = getY.(XZ₁) 
XC₂ = getX.(XZ₂); ZC₂ = getY.(XZ₂) 
XC₃ = getX.(XZ₃); ZC₃ = getY.(XZ₃)

pyplot()

scalefontsizes(1/1.8)

plt3 = Plots.contourf(XC₁, ZC₁, reshape(absu1, size(XC₁)...), colormap=:turbo,clims=(1,5))
Plots.contourf!(plt3, XC₂, ZC₂, reshape(absu2, size(XC₂)...), label="", colormap=:turbo,clims=(1,5))
Plots.contourf!(plt3, XC₃, ZC₃, reshape(absu3, size(XC₃)...), label="", colormap=:turbo,clims=(1,5))
Plots.plot!(plt3, [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:pink, label="Interface 1")
Plots.plot!(plt3, [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:pink, label="Interface 2")
Plots.vline!(plt3, [(x₁[1]+0.9*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≥ Lₓ (PML)")
Plots.vline!(plt3, [(x₁[1]+0.1*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≤ Lₓ (PML)", legend=:topleft, size=(1600,600), 
             topmargin=1*Plots.cm, bottommargin=1*Plots.cm, leftmargin=1*Plots.cm)
Plots.xlims!(plt3, (x₁[1],x₁[end]))
Plots.ylims!(plt3, (z₂[1],z₁[end]))

plt4 = Plots.contourf(XC₂, ZC₂, reshape(C₃₃², size(XC₂)...), label="", colormap=:turbo)
Plots.contourf!(plt4, XC₃, ZC₃, reshape(C₃₃³, size(XC₃)...), label="", colormap=:turbo)
Plots.contourf!(plt4, XC₁, ZC₁, reshape(C₃₃¹, size(XC₁)...), colormap=:turbo, label="",clims=(0.01,15))
Plots.plot!(plt4, [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:pink, label="Interface 1")
Plots.plot!(plt4, [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:pink, label="Interface 2")
Plots.vline!(plt4, [(x₁[1]+0.9*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≥ Lₓ (PML)")
Plots.vline!(plt4, [(x₁[1]+0.1*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≤ Lₓ (PML)", legend=:topleft, size=(1600,600), 
             topmargin=1*Plots.cm, bottommargin=1*Plots.cm, leftmargin=1*Plots.cm)
Plots.xlims!(plt4, (x₁[1],x₁[end]))
Plots.ylims!(plt4, (z₂[1],z₁[end]))

plt5 = Plots.contourf(XC₂, ZC₂, reshape(σᵥ.(Ω₂.(𝐪𝐫₂)), size(XC₂)...), label="", colormap=:turbo)
Plots.contourf!(plt5, XC₃, ZC₃, reshape(σᵥ.(Ω₃.(𝐪𝐫₃)), size(XC₃)...), label="", colormap=:turbo)
Plots.contourf!(plt5, XC₁, ZC₁, reshape(σᵥ.(Ω₁.(𝐪𝐫₁)), size(XC₁)...), colormap=:turbo, label="")
Plots.plot!(plt5, [0,x₁[end]],[-3.34,-2.47], lw=2, lc=:pink, label="Interface 1")
Plots.plot!(plt5, [0,x₁[end]],[z₁[1],z₁[1]], lw=2, lc=:pink, label="Interface 2")
Plots.vline!(plt5, [(x₁[1]+0.9*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≥ Lₓ (PML)")
Plots.vline!(plt5, [(x₁[1]+0.1*Lₕ)], lw=1, lc=:pink, ls=:dash, label="x ≤ Lₓ (PML)", legend=:topleft, size=(1600,600), 
             topmargin=1*Plots.cm, bottommargin=1*Plots.cm, leftmargin=1*Plots.cm)
Plots.xlims!(plt5, (x₁[1],x₁[end]))
Plots.ylims!(plt5, (z₂[1],z₁[end]))

# Plots.plot(plt3, plt4, layout=(2,1), size=(1600,1200))