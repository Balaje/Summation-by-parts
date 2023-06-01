include("geometry.jl");
include("material_props.jl");
include("SBP.jl");
include("../time-stepping.jl");

using Plots

"""
The material property tensor in the physical coordinates
  𝒫(x) = [A(x) C(x); 
          C(x)' B(x)]
where A(x), B(x) and C(x) are the material coefficient matrices in the phyiscal domain (Defined in material_props.jl)
"""
𝒫(x) = @SMatrix [A₁₁(x) A₁₂(x) C₁₁(x) C₁₂(x); 
                 A₂₁(x) A₂₂(x) C₂₁(x) C₂₂(x); 
                 C₁₁(x) C₂₁(x) B₁₁(x) B₁₂(x);
                 C₁₂(x) C₂₂(x) B₂₁(x) B₂₂(x)];

"""
Function to return the material tensor in the reference coordinates (0,1)×(0,1). Returns 
  𝒫' = S*𝒫*S'
where S is the transformation matrix
"""
function t(S, r)  
  invJ = J⁻¹(S, r)    
  Pr = 𝒫(r)
  S = invJ ⊗ I(2)
  S*Pr*S'
end

"""
The material coefficient matrices in the reference coordinates (0,1)×(0,1).
  A(x) -> Aₜ(r)
  B(x) -> Bₜ(r)
  C(x) -> Cₜ(r) 
"""
Aₜ(r) = t(𝒮,r)[1:2, 1:2];
Bₜ(r) = t(𝒮,r)[3:4, 3:4];
Cₜ(r) = t(𝒮,r)[1:2, 3:4];

"""
The tensor form of variable SBP finite difference for the elasticity problem. 
This approximates the ∂q(A*∂q) operator, where A is the function evaluated at the grid points.
Must pass the OnGrid operation on the 
"""
function 𝐃𝐪𝐪(𝐀::OnGrid)
  X = 𝐀.X
  M = size(X[1], 1)
  dropzeros([SBP_VARIABLE_4(M, X[1])[2] SBP_VARIABLE_4(M, X[2])[2];
   SBP_VARIABLE_4(M, X[3])[2] SBP_VARIABLE_4(M, X[4])[2]])
end

"""
Sparsify the OnGrid function. Does the operation
  A = OnGrid(f)
      # Here A.x = (Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64})
  sparsify(A)
      # Does [spdiagm(A.x[1]) spdiagm(A.x[2]); 
              spdiagm(A.x[3]) spdiagm(A.x[4])]
"""
function sparsify(A::OnGrid)
  X = A.X
  dropzeros([spdiagm(X[1]) spdiagm(X[2]); 
   spdiagm(X[3]) spdiagm(X[4])])
end


#################################
# Now begin solving the problem #
#################################

domain = (0.0,1.0,0.0,1.0);
M = 21; # No of points along the axes
q = LinRange(0,1,M);
r = LinRange(0,1,M);
QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);

# Compute the material properties on the Grid.
# (Using the OnGrid interface defined in geometry.jl)
𝐀ₜ = OnGrid(Aₜ, QR);
𝐁ₜ = OnGrid(Bₜ, QR);
𝐂ₜ = OnGrid(Cₜ, QR);

# Get the stencils for computing the first derivatives.
METHOD = SBP(M);

# Penalty terms for applying the boundary conditions using the SAT method
τ₀ = -1;
τ₁ = 1;
τ₂ = -1;
τ₃ = 1;
pterms = (τ₀, τ₁, τ₂, τ₃)


"""
The stiffness term (K) in the elastic wave equation
  Ü + KU = f
"""
function K(stencil)
  METHOD, pterms, coeffs = stencil
  𝐀ₜ, 𝐁ₜ, 𝐂ₜ = coeffs

  # Collect all the necessary finite difference matrices from the method
  # NOTE: Here D2s, H are not needed. 
  #       The D2s matrix is not needed since we use the variable SBP operator
  #       H because Hinv is precomputed
  HHinv, D1, D2s, S, Ids = METHOD;
  H, Hinv = HHinv;
  E₀, Eₙ, e₀, eₙ, Id = Ids; # Needed for non-zero boundary conditions

  # Finite difference operators along the (q,r) direction
  Dq = D1; Dr = D1
  Sq = S; Sr = S;
  Hq = H; Hr = H;
  Hqinv = Hinv; Hrinv = Hinv;
  τ₀, τ₁, τ₂, τ₃ = pterms

  # Discrete Operators in 2D
  𝐃𝐪 = (I(2) ⊗ Dq ⊗ I(M));
  𝐃𝐫 = (I(2) ⊗ I(M) ⊗ Dr);
  𝐒𝐪 = (I(2) ⊗ Sq ⊗ I(M));
  𝐒𝐫 = (I(2) ⊗ I(M) ⊗ Sr);  
  
  𝐇𝐪₀⁻¹ = (I(2) ⊗ (Hqinv*E₀) ⊗ I(M)); # q (x) = 0
  𝐇𝐫₀⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*E₀)); # r (y) = 0
  𝐇𝐪ₙ⁻¹ = (I(2) ⊗ (Hqinv*Eₙ) ⊗ I(M)); # q (x) = 1 
  𝐇𝐫ₙ⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*Eₙ)); # r (y) = 1 

  # The variable SBP operator
  𝐃𝐪𝐪ᴬ = 𝐃𝐪𝐪(𝐀ₜ);
  𝐃𝐫𝐫ᴮ = 𝐃𝐪𝐪(𝐁ₜ);

  # Get the matrix version of the material properties
  𝐀 = sparsify(𝐀ₜ);
  𝐁 = sparsify(𝐁ₜ);
  𝐂 = sparsify(𝐂ₜ);

  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪*𝐂*𝐃𝐫 + 𝐃𝐫*𝐂'*𝐃𝐪); # The Elastic wave-equation operator
  𝐓𝐪 = (𝐀*𝐒𝐪 + 𝐂*𝐃𝐫); # The horizontal traction operator
  𝐓𝐫 = (𝐁*𝐒𝐫 + 𝐂'*𝐃𝐪); # The vertical traction operator

  (𝐏 - (τ₀*𝐇𝐫₀⁻¹*𝐓𝐫 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫 + τ₂*𝐇𝐪₀⁻¹*𝐓𝐪 + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪)) # The "stiffness term"  
end

# Assume an initial condition and load vector.
U₀(x) = (@SVector [sin(π*x[1])*sin(π*x[2]), 0.0])';
Uₜ₀(x) = (@SVector [0.0, 0.0])';
F(x,t) = (@SVector [0.0, 1.0])';

# Begin solving the problem
# Temporal Discretization parameters
tf = 1.0
Δt = 1e-3
ntime = ceil(Int64,tf/Δt)
# Plots
plt = plot()
plt1 = plot()

args = METHOD, pterms, (𝐀ₜ, 𝐁ₜ, 𝐂ₜ)

# The SBP matrices
HHinv, D1, D2s, S, Ids = METHOD;
H, Hinv = HHinv;
E₀, Eₙ, e₀, eₙ, Id = Ids;
Dq = D1; Dr = D1
Sq = S; Sr = S;
Hq = H; Hr = H;
Hqinv = Hinv; Hrinv = Hinv;
τ₀, τ₁, τ₂, τ₃ = pterms

stima = K(args)
massma = ρ*spdiagm(ones(size(stima,1)))
let
  u₀ = vec(reduce(vcat, U₀.(QR)));
  v₀ = vec(reduce(vcat, Uₜ₀.(QR)));  
  global u₁ = zero(u₀)  
  global v₁ = zero(v₀)  
  t = 0.0
  for i=1:ntime    
    Fvec = vec(reduce(vcat, F.(QR,t) + F.(QR,t+Δt)))    
    fargs = Δt, u₀, v₀, Fvec
    u₁,v₁ = CN(stima, massma, fargs)
    t = t+Δt
    u₀ = u₁
    v₀ = v₁
    (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(sum(u₀)))
  end  
end