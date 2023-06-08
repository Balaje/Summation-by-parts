include("geometry.jl");
include("material_props.jl");
include("SBP.jl");
include("SBP_2d.jl")
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

"""
Flatten the 2d function as a single vector for the time iterations
"""
eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = vec(reshape(reinterpret(Float64, v), dim, :)');


"""
The stiffness term (K) in the elastic wave equation
Ü = -K*U + (f + g)
"""
function stima(sbp_2d, pterms)
  (𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), _ = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms  
  # The second derivative SBP operator
  𝐃𝐪𝐪ᴬ = A ⊗ 𝐃𝐪𝐪
  𝐃𝐫𝐫ᴮ = B ⊗ 𝐃𝐫𝐫
  𝐃𝐪C𝐃𝐫 = (I(2) ⊗ 𝐃𝐪) * (C ⊗ 𝐃𝐫)
  𝐃𝐫Cᵗ𝐃𝐪 = (I(2) ⊗ 𝐃𝐫) * (Cᵀ ⊗ 𝐃𝐪)  
  # The Elastic wave-equation operators
  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪) # The bulk term
  𝐓𝐪₀ = -(A ⊗ 𝐒𝐪 + C ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫₀ = -(Cᵀ ⊗ 𝐃𝐪 + B ⊗ 𝐒𝐫) # The vertical traction operator
  𝐓𝐪ₙ = (A ⊗ 𝐒𝐪 + C ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫ₙ = (Cᵀ ⊗ 𝐃𝐪 + B ⊗ 𝐒𝐫) # The vertical traction operator
  # The "stiffness term"  
  𝐏 - (τ₀*𝐇𝐫₀⁻¹*𝐓𝐫₀ + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫ₙ + τ₂*𝐇𝐪₀⁻¹*𝐓𝐪₀ + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪ₙ) 
end

"""
The boundary contribution terms g
  Ü = -K*U + (f + g)
Applied into the load vector during time stepping
"""
function nbc(t::Float64, XY, sbp_2d, pterms)
  _, _, (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), (𝐈q₀, 𝐈r₀, 𝐈qₙ, 𝐈rₙ) = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms

  bq₀ = eltocols(𝐈q₀*g₀.(XY, t)) # q (x) = 0  
  br₀ = eltocols(𝐈r₀*g₁.(XY, t)) # r (y) = 0
  bqₙ = eltocols(𝐈qₙ*g₂.(XY,t)) # q (x) = 1
  brₙ = eltocols(𝐈rₙ*g₃.(XY,t)) # r (y) = 1

  (τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ + τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end

#################################
# Now begin solving the problem #
#################################

# Assume an exact solution and compute the intitial condition and load vector
U(x,t) = (@SVector [sin(π*x[1])*sin(π*x[2])*sin(π*t), sin(2π*x[1])*sin(2π*x[2])*sin(π*t)]);
# Compute the right hand side using the exact solution
Uₜ(x,t) = ForwardDiff.derivative(τ->U(x,τ), t)
Uₜₜ(x,t) = ForwardDiff.derivative(τ->Uₜ(x,τ), t)
# Compute the initial data from the exact solution
U₀(x) = U(x,0);
Uₜ₀(x) = Uₜ(x,0);
function F(x,t) 
  V(x) = U(x,t)
  Uₜₜ(x,t) - divσ(V, x);
end
function g₀(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(-1) + τ[2]*(0); τ[3]*(-1) + τ[4]*(0)]
end
function g₁(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(0) + τ[2]*(-1); τ[3]*(0) + τ[4]*(-1)]
end
function g₂(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(1) + τ[2]*(0); τ[3]*(1) + τ[4]*(0)]
end
function g₃(x,t)
  V(x) = U(x,t)
  𝛔(y) = σ(∇(V, y),y);  
  τ = 𝛔(x)  
  @SVector [τ[1]*(0) + τ[2]*(1); τ[3]*(0) + τ[4]*(1)]
end

# Discretize the domain
domain = (0.0,1.0,0.0,1.0);
M = 11; # No of points along the axes
q = LinRange(0,1,M);
r = LinRange(0,1,M);
XY = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
# Get the SBP matrices
sbp_1d = SBP(M);
sbp_2d = SBP_2d(sbp_1d);
# Penalty terms for applying the boundary conditions using the SAT method
τ₀ = τ₁ = τ₂ = τ₃ = 1;
pterms = (τ₀, τ₁, τ₂, τ₃)
# Begin solving the problem
# Temporal Discretization parameters
tf = 0.25
Δt = 1e-3
ntime = ceil(Int64,tf/Δt)
# Empty Plots
plt = plot()
plt1 = plot()
𝐊 = stima(sbp_2d, pterms)
𝐌 = ρ*spdiagm(ones(2*M^2))
let
  u₀ = eltocols(U.(XY,0))
  v₀ = eltocols(Uₜ.(XY,0))
  #=  
  # Leapfrog scheme
  t = 0.0
  fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
  u₁ = LF1(𝐊, 𝐌⁻¹, (Δt, u₀, fₙ, v₀))
  u₀ = u₁
  t += Δt
  global u₂ = zero(u₀)
  for i=2:ntime
    fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
    u₂ = LF(𝐊, 𝐌⁻¹, (Δt, u₁, u₀, fₙ))
    u₀ = u₁
    u₁ = u₂
    t += Δt    
    (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(maximum(abs.(u₀))))
  end
  global sol = u₂ 
  =#
  
  # Crank Nicolson Method
  global u₁ = zero(u₀)  
  global v₁ = zero(v₀)  
  t = 0.0
  for i=1:ntime   
    Fₙ = eltocols(F.(XY, t))
    Fₙ₊₁ = eltocols(F.(XY, t+Δt))
    gₙ = nbc(t, XY, sbp_2d, pterms)
    gₙ₊₁ = nbc(t+Δt, XY, sbp_2d, pterms)

    rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
    fargs = Δt, u₀, v₀, rhs
    u₁,v₁ = CN(𝐊, 𝐌, fargs)
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