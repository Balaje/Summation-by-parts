include("geometry.jl");
include("material_props.jl");
include("SBP.jl");

# Reference Domain (0,1)×(0,1)
domain = (0.0,1.0,0.0,1.0);
M = 101; # No of points along the axes

q = LinRange(0,1,M);
r = LinRange(0,1,M);
QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);

# Compute the material properties on the Grid
𝐀 = ToGrid(Aₜ, QR);
𝐁 = ToGrid(Bₜ, QR);
𝐂 = ToGrid(Cₜ, QR);

# Collect all the necessary matrices from the SBP method
METHOD = SBP(M);
HHinv, D1, D2s, S, Ids = METHOD;
H, Hinv = HHinv;
D2, D2c = D2s;
E₀, Eₙ, e₀, eₙ, Id = Ids;

# Finite difference operators along the (q,r) direction
Dq = D1; Dr = D1
Sq = S; Sr = S;
Hq = H; Hr = H;

𝐃𝐪 = (I(2) ⊗ Dq ⊗ I(M));
𝐃𝐫 = (I(2) ⊗ I(M) ⊗ Dr);
𝐒𝐪 = (I(2) ⊗ Sq ⊗ I(M));
𝐒𝐫 = (I(2) ⊗ I(M) ⊗ Sr);