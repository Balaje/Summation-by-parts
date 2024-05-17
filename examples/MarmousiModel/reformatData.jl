################################################
# Function to reformat the Marmousi Model data #
################################################

using MAT
using Plots

function downSample(X,Z,rho,vp,vs,d)
  X[1:d:end,1:d:end], Z[1:d:end,1:d:end], rho[1:d:end,1:d:end], vp[1:d:end,1:d:end], vs[1:d:end,1:d:end]
end

vars = matread("./examples/MarmousiModel/marmousi2.mat");
s = vars["marmousi2"];

rho = s["rho"]*1000;
vp = s["vp"]*1000;
vs = s["vs"]*1000;
dx = s["dx"]*1000;
dz = s["dz"]*1000;

Nz, Nx = size(rho);
Lx = (Nx-1)*dx;
Lz = (Nz-1)*dz;
x = 0:dx:Lx;
z = -(0:dz:Lz);

X = x' .* ones(length(z));
Z = ones(length(x))' .* z;

# First 361 layers are water, interface at z=-0.40496 km
N_water = 361;
rho_water = rho[1:N_water, :];
vp_water = vp[1:N_water, :];
vs_water = vs[1:N_water, :];
X_water = X[1:N_water, :];
Z_water = Z[1:N_water, :];

# Layers 362:end are elastic, ut we add one layer to get back to interface coord.
X_el = X[N_water:end, :];
Z_el = Z[N_water:end, :];
rho_el = rho[N_water+1:end, :];
rho_el = reduce(hcat, (2*rho_el[1,:] - rho_el[2,:], rho_el'))';
vp_el = vp[N_water+1:end, :];
vp_el = reduce(hcat, (2*vp_el[1,:] - vp_el[2,:], vp_el'))';
vs_el = vs[N_water+1:end, :];
vs_el = reduce(hcat, (2*vs_el[1,:] - vs_el[2,:], vs_el'))';

Z_el = reverse(Z_el, dims=1)
rho_el = reverse(rho_el, dims=1)
vp_el = reverse(vp_el, dims=1)
vs_el = reverse(vs_el, dims=1)
Z_water = reverse(Z_water, dims=1)
rho_water = reverse(rho_water, dims=1)
vp_water = reverse(vp_water, dims=1)
vs_water = reverse(vs_water, dims=1)
Z = reverse(Z, dims=1)
rho = reverse(rho, dims=1)
vp = reverse(vp, dims=1)
vs = reverse(vs, dims=1)

downSamplings = [10]
for i = 1:lastindex(downSamplings)
  global d = downSamplings[i];
  X_e,Z_e,rho_e,vp_e,vs_e = downSample(X_el, Z_el, rho_el, vp_el, vs_el, d);
  X_w,Z_w,rho_w,vp_w,vs_w = downSample(X_water, Z_water, rho_water, vp_water, vs_water, d);
  XX,ZZ,RHO,VP,VS = downSample(X, Z, rho, vp, vs, d);

  filename = "./examples/MarmousiModel/marmousi2_downsampled_"*string(d)*".mat";
  ds_data = Dict("X_e"=>X_e, 
           "Z_e"=>Z_e,
           "rho_e"=>rho_e,
           "vp_e"=>vp_e,
           "vs_e"=>vs_e,
           "X_w"=>X_w,
           "Z_w"=>Z_w,
           "rho_w"=>rho_w,
           "vp_w"=>vp_w,
           "vs_w"=>vs_w,
           "X"=>XX,
           "Z"=>ZZ,
           "rho"=>RHO,
           "vp"=>VP,
           "vs"=>VS);
  matwrite(filename, ds_data)
  
end

############### ##################
# Visualize the downsampled data #
############### ################## 
vars = matread("./examples/MarmousiModel/marmousi2_downsampled_"*string(d)*".mat");
vp = vars["vp"]
vs = vars["vs"]
X = vars["X"]
Z = vars["Z"]
rho = vars["rho"]
plt1 = Plots.contourf(X[1,:]/1000, Z[:,1]/1000, vp/1000, lw=0.1, size=(800,400), bottom_margin=8*Plots.mm, xtickfontsize=12, ytickfontsize=12, top_margin=5*Plots.mm, xguidefontsize=12, yguidefontsize=12, left_margin=8*Plots.mm);
xlabel!(plt1, "Distance (km)");
ylabel!(plt1, "Depth (km)");
title!(plt1, "Cₚ (km/s)")

plt2 = Plots.contourf(X[1,:]/1000, Z[:,1]/1000, vs/1000, lw=0.1, size=(800,400), bottom_margin=8*Plots.mm, xtickfontsize=12, ytickfontsize=12, top_margin=5*Plots.mm, xguidefontsize=12, yguidefontsize=12, left_margin=8*Plots.mm);
xlabel!(plt2, "Distance (km)");
ylabel!(plt2, "Depth (km)");
title!(plt2, "Cₛ (km/s)")

plt3 = Plots.contourf(X[1,:]/1000, Z[:,1]/1000, rho/1000, lw=0.1, size=(800,400), bottom_margin=8*Plots.mm, xtickfontsize=12, ytickfontsize=12, top_margin=5*Plots.mm, xguidefontsize=12, yguidefontsize=12, left_margin=8*Plots.mm);
xlabel!(plt3, "Distance (km)");
ylabel!(plt3, "Depth (km)");
title!(plt3, "ρ (kg/m³)")