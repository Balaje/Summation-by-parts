include("geometry.jl");
# include("material_props.jl");
include("SBP.jl");
include("../time-stepping.jl");

using Plots

"""
Function to return the material tensor in the reference coordinates (0,1)×(0,1). Returns 
  𝒫' = S*𝒫*S'
where S is the transformation matrix
"""
function t(S, r)  
  invJ = J⁻¹(S, r)      
  S = invJ ⊗ I(2)
  S*𝒫*S'
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

#################################
# Now begin solving the problem #
#################################

domain = (0.0,1.0,0.0,1.0);
M = 11; # No of points along the axes
q = LinRange(0,1,M);
r = LinRange(0,1,M);
QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);

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
  METHOD, pterms = stencil

  # Collect all the necessary finite difference matrices from the method
  # NOTE: Here D2s, H are not needed. 
  #       The D2s matrix is not needed since we use the variable SBP operator
  #       H because Hinv is precomputed
  HHinv, D1, D2s, S, Ids = METHOD;
  H, Hinv = HHinv;
  E₀, Eₙ, e₀, eₙ, Id = Ids; # Needed for non-zero boundary conditions

  # Finite difference operators along the (q,r) direction
  Dq = D1; Dr = D1
  Dqq = D2s[1]; Drr = D2s[1];
  Sq = S; Sr = S;
  # Hq = H; Hr = H;
  Hqinv = Hinv; Hrinv = Hinv;
  τ₀, τ₁, τ₂, τ₃ = pterms

  # Discrete Operators in 2D
  𝐃𝐪 = Dq ⊗ I(M);
  𝐃𝐫 = I(M) ⊗ Dr;
  𝐒𝐪 = Sq ⊗ I(M);
  𝐒𝐫 = I(M) ⊗ Sr;  
  
  𝐇𝐪₀⁻¹ = (I(2) ⊗ (Hqinv*E₀) ⊗ I(M)); # q (x) = 0
  𝐇𝐫₀⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*E₀)); # r (y) = 0
  𝐇𝐪ₙ⁻¹ = (I(2) ⊗ (Hqinv*Eₙ) ⊗ I(M)); # q (x) = 1 
  𝐇𝐫ₙ⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*Eₙ)); # r (y) = 1 

  # The second derivative SBP operator
  𝐃𝐪𝐪ᴬ = A ⊗ (Dqq ⊗ I(M))
  𝐃𝐫𝐫ᴮ = B ⊗ (I(M) ⊗ Drr)
  𝐃𝐪C𝐃𝐫 = C ⊗ (𝐃𝐪 * 𝐃𝐫)
  𝐃𝐫Cᵗ𝐃𝐪 = Cᵀ ⊗ (𝐃𝐫 * 𝐃𝐪)

  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪); # The Elastic wave-equation operator
  𝐓𝐪 = (A ⊗ 𝐒𝐪 + C ⊗ 𝐃𝐫); # The horizontal traction operator
  𝐓𝐫 = (Cᵀ ⊗ 𝐃𝐪 + B ⊗ 𝐒𝐫); # The vertical traction operator

  -𝐏 + (τ₀*𝐇𝐫₀⁻¹*𝐓𝐫 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫 + τ₂*𝐇𝐪₀⁻¹*𝐓𝐪 + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪) # The "stiffness term"  
end

"""
The boundary contribution terms. Applied into the load vector during time stepping
"""
function BC(t::Float64, stencil)
  METHOD, pterms, QR = stencil
  Hinv = METHOD[1][2]
  Ids = METHOD[5];
  Hqinv = Hinv; Hrinv = Hinv
  τ₀, τ₁, τ₂, τ₃ = pterms
  E₀, Eₙ, _, _, _ = Ids

  𝐇𝐪₀⁻¹ = (I(2) ⊗ (Hqinv*E₀) ⊗ I(M)); # q (x) = 0
  𝐇𝐫₀⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*E₀)); # r (y) = 0
  𝐇𝐪ₙ⁻¹ = (I(2) ⊗ (Hqinv*Eₙ) ⊗ I(M)); # q (x) = 1 
  𝐇𝐫ₙ⁻¹ = (I(2) ⊗ I(M) ⊗ (Hrinv*Eₙ)); # r (y) = 1  

  bq₀ = (I(2) ⊗ E₀ ⊗ I(M))*vec(reduce(vcat, g₃.(QR,t))') # q (x) = 0
  br₀ = (I(2) ⊗ I(M) ⊗ E₀)*vec(reduce(vcat, g₀.(QR,t))') # r (y) = 0
  bqₙ = (I(2) ⊗ Eₙ ⊗ I(M))*vec(reduce(vcat, g₁.(QR,t))') # q (x) = 1
  brₙ = (I(2) ⊗ I(M) ⊗ Eₙ)*vec(reduce(vcat, g₂.(QR,t))') # r (y) = 1

  -(τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ + τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end



# Assume an exact solution and compute the intitial condition and load vector
U(x,t) = (@SVector [sin(π*x[1])*sin(π*x[2])*sin(π*t), sin(2π*x[1])*sin(2π*x[2])*sin(π*t)]);
# Compute the right hand side using the exact solution
Uₜ(x,t) = ForwardDiff.derivative(τ->U(x,τ), t)
Uₜₜ(x,t) = ForwardDiff.derivative(τ->Uₜ(x,τ), t)
# Compute the initial data from the exact solution
U₀(x) = U(x,0);
Uₜ₀(x) = Uₜ(x,0);
"""
The right-hand side function
"""
function F(x,t) 
  V(x) = U(x,t)
  Uₜₜ(x,t) - divσ(V, x);
end
"""
The Neumann boundary conditions (σ⋅n)
"""
function 𝐠(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y)...);
  n = @SMatrix [0 1 0 -1; -1 0 1 0]
  𝛔(x)*n   
end
g₀(x,t) = 𝐠(x,t)[:,1]
g₁(x,t) = 𝐠(x,t)[:,2]
g₂(x,t) = 𝐠(x,t)[:,3]
g₃(x,t) = 𝐠(x,t)[:,4]


# Begin solving the problem
# Temporal Discretization parameters
tf = 0.25
Δt = 1e-3
ntime = ceil(Int64,tf/Δt)
# Plots
plt = plot()
plt1 = plot()

args = METHOD, pterms;

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
  u₀ = vec(reduce(vcat, U₀.(QR))');
  v₀ = vec(reduce(vcat, Uₜ₀.(QR)));  
  global u₁ = zero(u₀)  
  global v₁ = zero(v₀)  
  t = 0.0
  for i=1:ntime    
    rhs = vec(reduce(vcat,F.(QR,t))') + vec(reduce(vcat,F.(QR,t+Δt))) + BC(t, (METHOD,pterms,QR)) + BC(t+Δt, (METHOD,pterms,QR))    
    fargs = Δt, u₀, v₀, -rhs
    u₁,v₁ = CN(stima, massma, fargs)
    t = t+Δt
    u₀ = u₁
    v₀ = v₁
    (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(maximum(abs.(u₀))))
  end  
end

function UV(sol)
  _2M² = length(sol)
  M² = Int(_2M²/2)
  M = Int(sqrt((length(sol))/2))
  (reshape(reshape(sol,(2,M²))[1,:],(M,M)), reshape(reshape(sol,(2,M²))[2,:],(M,M)))
end