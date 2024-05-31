# Install pyplot for this to work ....
using PyPlot
using Plots
pyplot()
using LaTeXStrings

PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

using DelimitedFiles

function get_data_from_plot(plt)
  (plt.series_list[1].plotattributes[:x].surf, plt.series_list[1].plotattributes[:y].surf, plt.series_list[1].plotattributes[:z].surf), 
  (plt.series_list[2].plotattributes[:x].surf, plt.series_list[2].plotattributes[:y].surf, plt.series_list[2].plotattributes[:z].surf)
end

function plot_solution_log_scale(data1, data2, clims, n, nlevels)
  XC₁, ZC₁, absu1 = data1
  XC₂, ZC₂, absu2 = data2
  plt3_1 = Plots.plot();

  if(clims==Nothing)
    Plots.contourf!(plt3_1, XC₁, ZC₁, (absu1), label="", colormap=:jet)
    Plots.contourf!(plt3_1, XC₂, ZC₂, (absu2), label="", colormap=:jet)
  else  
    Plots.contourf!(plt3_1, XC₁, ZC₁, (absu1), label="", colormap=:jet, levels=nlevels)
    Plots.contourf!(plt3_1, XC₂, ZC₂, (absu2), label="", colormap=:jet, clims=clims, levels=nlevels)
    c_ticks = (LinRange(clims...,n), string.(round.(LinRange(clims...,n), digits=2)));
    Plots.plot!(plt3_1, colorbar_ticks=c_ticks)
  end  
  Plots.plot!(plt3_1, [0,16.9864],[-3.34,-2.47], lw=2, lc=:white, label="")
  Plots.plot!(plt3_1, [0,16.9864],[-3.4972,-3.4972], lw=2, lc=:white, label="")
  Plots.vline!(plt3_1, [(0.9*16.9864)], lw=1, lc=:white, ls=:dash, label="")
  Plots.vline!(plt3_1, [(0.1*16.9864)], lw=1, lc=:white, ls=:dash, label="", legend=:topleft, size=(600,200))  
  Plots.xlims!(plt3_1, (0,16.9864))
  Plots.ylims!(plt3_1, (-3.4972,-0.44964))
  Plots.xlabel!(plt3_1, "\$x\$")
  Plots.ylabel!(plt3_1, "\$z\$")
  plt3_1
end

function write_data_to_disk(filename, plt)
  data1, data2 = get_data_from_plot(plt)
  XC₁, ZC₁, absu1 = data1
  XC₂, ZC₂, absu2 = data2
  open("./marmousi-paper-images/DATA/"*filename*"_layer_1.txt", "w") do io
    writedlm(io, hcat(vec(XC₁), vec(ZC₁), vec(absu1)))
  end
  open("./marmousi-paper-images/DATA/"*filename*"_layer_2.txt", "w") do io
    writedlm(io, hcat(vec(XC₂), vec(ZC₂), vec(absu2)))
  end
end

time_t = 4.5;
XYZ₁ = readdlm("./marmousi-paper-images/DATA/marmousi_sol_"*string(time_t)*"_layer_1.txt");
XYZ₂ = readdlm("./marmousi-paper-images/DATA/marmousi_sol_"*string(time_t)*"_layer_2.txt");
X₁ = XYZ₁[:,1]; Y₁ = XYZ₁[:,2]; Z₁ = XYZ₁[:,3];
X₂ = XYZ₂[:,1]; Y₂ = XYZ₂[:,2]; Z₂ = XYZ₂[:,3];
MN = readdlm("./marmousi-paper-images/DATA/grid.txt")
M₁ = Int64(MN[1,1]); N₁ = Int64(MN[1,2]);
M₂ = Int64(MN[2,1]); N₂ = Int64(MN[2,2]);

clims = (0.0, 0.5)
plot_ms = Plots.contour(reshape(X₁,(M₁,N₁)), reshape(Y₁,(M₁,N₁)), reshape(Z₁,(M₁,N₁)), colormap=:jet, size=(600,200), clims=clims, levels=50);
Plots.contour!(plot_ms, reshape(X₂, (M₂,N₂)), reshape(Y₂,(M₂,N₂)),  reshape(Z₂,(M₂,N₂)), colormap=:jet, levels=50);
c_ticks = (LinRange(clims...,3), string.(round.(LinRange(clims...,3), digits=2)));
Plots.plot!(plot_ms, colorbar_ticks=c_ticks)
Plots.plot!(plot_ms, [0,16.9864],[-3.34,-2.47], lw=1, lc=:red, label="")
Plots.plot!(plot_ms, [0,16.9864],[-3.4972,-3.4972], lw=1, lc=:red, label="")
Plots.vline!(plot_ms, [(0.9*16.9864)], lw=1, lc=:red, ls=:dash, label="")
Plots.vline!(plot_ms, [(0.1*16.9864)], lw=1, lc=:red, ls=:dash, label="", legend=:topleft, size=(600,200))  
Plots.xlims!(plot_ms, (0,16.9864))
Plots.ylims!(plot_ms, (-3.4972,-0.44964))
Plots.xlabel!(plot_ms, "\$x\$")
Plots.ylabel!(plot_ms, "\$z\$")