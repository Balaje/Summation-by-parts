"""
x-and-y coordinates from the SVector{2,Float64}
"""
getX(C) = C[1]; 
getY(C) = C[2];


"""
Function to plot the displacement field along with the geometry of the domain
"""
function plot_displacement_field!(plt::Plots.Plot, xy::NTuple{2,AbstractMatrix{SVector{2,Float64}}}, U::NTuple{2,AbstractVector{Float64}}, 
  xlimits::NTuple{2,Float64}, ylimits::NTuple{2,Float64}, 
  δx::NTuple{2,Float64}, δy::NTuple{2,Float64}, cᵢ::Function)

  xy₁, xy₂ = xy
  U1, U2 = U
  x₀, x₁ = xlimits
  y₀, y₁ = ylimits
  δx₀, δx₁ = δx 
  δy₀, δy₁ = δy
  Plots.contourf!(plt, getX.(xy₁), getY.(xy₁), reshape(U1,size(xy₁)...), colormap=:jet)
  Plots.contourf!(plt, getX.(xy₂), getY.(xy₂), reshape(U2,size(xy₂)...), colormap=:jet)
  Plots.vline!(plt, [Lᵥ], label="\$ x \\ge "*string(round(Lᵥ, digits=3))*"\$ (PML)", lc=:black, lw=1, ls=:dash)
  Plots.plot!(plt, getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="Interface", lc=:red, lw=2, size=(400,500), legend=:none)
  xlims!(plt, (x₀-δx₀,x₁+δx₁))
  ylims!(plt, (y₀-δy₀,y₁+δy₁))
  xlabel!(plt, "\$x\$")
  ylabel!(plt, "\$y\$")
  plt
end

"""
Plot the discretization
"""
function plot_discretization!(plt4::Plots.Plot, xy::NTuple{2,AbstractMatrix{SVector{2,Float64}}},
  xlimits::NTuple{2,Float64}, ylimits::NTuple{2,Float64}, 
  δx::NTuple{2,Float64}, δy::NTuple{2,Float64}, cᵢ::Function)

  xy₁, xy₂ = xy
  x₀, x₁ = xlimits
  y₀, y₁ = ylimits
  δx₀, δx₁ = δx 
  δy₀, δy₁ = δy 
  Plots.scatter!(plt4, vec(Tuple.(xy₁)), mc=:red, msw=0.01, ms=4, label="")
  Plots.scatter!(vec(Tuple.(xy₂)), mc=:blue, msw=0.01, ms=4, label="", size=(400,500))
  Plots.plot!(getX.(cᵢ.(LinRange(0,1,100))), getY.(cᵢ.(LinRange(0,1,100))), label="", lc=:green, lw=1, size=(400,500))
  xlims!(plt4, (x₀-0.1*(x₁-x₀)-δx₀,x₁+0.1*(x₁-x₀)+δx₁))
  ylims!(plt4, (y₀-0.1*(y₁-y₀)-δy₀,y₁+0.1*(y₁-y₀)+δy₁))
  xlabel!(plt4, "\$ x \$")
  ylabel!(plt4, "\$ y \$")
end