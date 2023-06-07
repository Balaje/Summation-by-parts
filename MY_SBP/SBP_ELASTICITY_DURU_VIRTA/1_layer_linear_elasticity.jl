include("geometry.jl");
include("material_props.jl");
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
M = 21; # No of points along the axes
q = LinRange(0,1,M);
r = LinRange(0,1,M);
QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);

"""
Flatten the 2d function as a single vector for the time iterations
"""
function flatten_grid_function(f, QR, t; P=I(M^2))
  F = f.(QR,t)
  collect(vec(reduce(hcat, P*F)'))
end

# Penalty terms for applying the boundary conditions using the SAT method
τ₀ = τ₁ = τ₂ = τ₃ = 1;
pterms = (τ₀, τ₁, τ₂, τ₃)

function SBP_2d(SBP_1d)
  # Collect all the necessary finite difference matrices from the method
  # NOTE: Here D2s, H are not needed. 
  #       The D2s matrix is not needed since we use the variable SBP operator
  #       H because Hinv is precomputed
  HHinv, D1, D2s, S, Ids = SBP_1d;
  _, Hinv = HHinv;
  E₀, Eₙ, _, _, Id = Ids; # Needed for non-zero boundary conditions

  # Finite difference operators along the (q,r) direction
  Dq = D1; Dr = D1
  Dqq = D2s[1]; Drr = D2s[1];
  Sq = S; Sr = S;  
  Hqinv = Hinv; Hrinv = Hinv;

  # Discrete Operators in 2D
  𝐃𝐪 = Dq ⊗ Id;
  𝐃𝐫 = Id ⊗ Dr;
  𝐒𝐪 = Sq ⊗ Id;
  𝐒𝐫 = Id ⊗ Sr;  
  
  𝐇𝐪₀⁻¹ = (I(2) ⊗ (Hqinv*E₀) ⊗ Id); # q (x) = 0
  𝐇𝐫₀⁻¹ = (I(2) ⊗ Id ⊗ (Hrinv*E₀)); # r (y) = 0
  𝐇𝐪ₙ⁻¹ = (I(2) ⊗ (Hqinv*Eₙ) ⊗ Id); # q (x) = 1 
  𝐇𝐫ₙ⁻¹ = (I(2) ⊗ Id ⊗ (Hrinv*Eₙ)); # r (y) = 1 

  # The second derivative SBP operator
  𝐃𝐪𝐪ᴬ = A ⊗ (Dqq ⊗ Id);
  𝐃𝐫𝐫ᴮ = B ⊗ (Id ⊗ Drr);
  𝐃𝐪C𝐃𝐫 = (I(2) ⊗ 𝐃𝐪) * (C ⊗ 𝐃𝐫);
  𝐃𝐫Cᵗ𝐃𝐪 = (I(2) ⊗ 𝐃𝐫) * (Cᵀ ⊗ 𝐃𝐪);

  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪); # The Elastic wave-equation operator
  𝐓𝐪 = (A ⊗ 𝐒𝐪 + C ⊗ 𝐃𝐫); # The horizontal traction operator
  𝐓𝐫 = (Cᵀ ⊗ 𝐃𝐪 + B ⊗ 𝐒𝐫); # The vertical traction operator

  𝐈q₀ = E₀ ⊗ Id
  𝐈qₙ = Eₙ ⊗ Id
  𝐈r₀ = Id ⊗ E₀
  𝐈rₙ = Id ⊗ Eₙ

  (𝐏, 𝐓𝐪, 𝐓𝐫), (𝐃𝐪𝐪ᴬ, 𝐃𝐫𝐫ᴮ, 𝐃𝐪C𝐃𝐫, 𝐃𝐫Cᵗ𝐃𝐪), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), (𝐈q₀, 𝐈r₀, 𝐈qₙ, 𝐈rₙ)
end


"""
The stiffness term (K) in the elastic wave equation
  Ü + KU = f
"""
function K(sbp, pterms)
  (𝐏, 𝐓𝐪, 𝐓𝐫), _, (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), _ = sbp
  τ₀, τ₁, τ₂, τ₃ = pterms
  -𝐏 + (-τ₀*𝐇𝐫₀⁻¹*𝐓𝐫 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫 - τ₂*𝐇𝐪₀⁻¹*𝐓𝐪 + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪) # The "stiffness term"  
end

"""
The boundary contribution terms. Applied into the load vector during time stepping
"""
function BC(t::Float64, sbp_2d, pterms)
  _, _, (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), (𝐈q₀, 𝐈r₀, 𝐈qₙ, 𝐈rₙ) = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms

  bq₀ = flatten_grid_function(g₃, QR, t; P=𝐈q₀) # q (x) = 0  
  br₀ = flatten_grid_function(g₀, QR, t; P=𝐈r₀) # r (y) = 0
  bqₙ = flatten_grid_function(g₁, QR, t; P=𝐈qₙ) # q (x) = 1
  brₙ = flatten_grid_function(g₂, QR, t; P=𝐈rₙ) # r (y) = 1

  -(-τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ - τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end



# Assume an exact solution and compute the intitial condition and load vector
U(x,t) = (@SVector [sin(π*x[1])*sin(π*x[2])*t^3, sin(2π*x[1])*sin(2π*x[2])*t^3]);
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
  SMatrix{2,4,Float64}(𝛔(x)*n)
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

sbp_1d = SBP(M);
sbp_2d = SBP_2d(sbp_1d);

stima = K(sbp_2d, pterms)
massma = ρ*spdiagm(ones(size(stima,1)))
massma⁻¹ = (1/ρ)*ones(Float64, 2*M^2)
let
  u₀ = flatten_grid_function(U, QR, 0)
  v₀ = flatten_grid_function(Uₜ, QR, 0)
 #=  # Leapfrog method
  t = 0.0
  fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
  u₁ = LF1(stima, massma⁻¹, (Δt, u₀, fₙ, v₀))
  u₀ = u₁
  t += Δt
  global u₂ = zero(u₀)
  for i=2:ntime
    fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
    u₂ = LF(stima, massma⁻¹, (Δt, u₁, u₀, fₙ))
    u₀ = u₁
    u₁ = u₂
    t += Δt    
    (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(maximum(abs.(u₀))))
  end
  global sol = u₂ =#
  
  # Crank Nicolson Method
  global u₁ = zero(u₀)  
  global v₁ = zero(v₀)  
  t = 0.0
  for i=1:ntime   
    Fₙ = flatten_grid_function(F, QR, t)
    Fₙ₊₁ = flatten_grid_function(F, QR, t+Δt)
    gₙ = BC(t, sbp_2d, pterms)
    gₙ₊₁ = BC(t+Δt, sbp_2d, pterms)

    rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
    fargs = Δt, u₀, v₀, rhs
    u₁,v₁ = CN(stima, massma, fargs)
    t = t+Δt
    u₀ = u₁
    v₀ = v₁
    # (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(maximum(abs.(u₀))))
  end   
  global sol = u₁  
end

# Compute the L²Error
H = sbp_1d[1][1]
𝐇 = I(2) ⊗ H ⊗ H
e = sol - flatten_grid_function(U, QR, tf)
@show sqrt(e'*𝐇*e)

function UV(sol)
  _2M² = length(sol)
  M² = Int(_2M²/2)
  M = Int(sqrt(M²))
  (reshape(sol[1:M²],(M,M)), reshape(sol[M²+1:end], (M,M)))
end

## Visualize the solution
Uap, Vap = UV(sol)
Ue, Ve = UV(reduce(hcat,U.(QR,tf))')
plt1 = contourf(LinRange(0,1,M), LinRange(0,1,M), Uap, title="u₁ Approximate")
plt2 = contourf(LinRange(0,1,M), LinRange(0,1,M), Ue, title="u₁ Exact")
plt3 = contourf(LinRange(0,1,M), LinRange(0,1,M), Vap, title="v₁ Approximate")
plt4 = contourf(LinRange(0,1,M), LinRange(0,1,M), Ve, title="v₁ Exact")
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(400,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(400,800));