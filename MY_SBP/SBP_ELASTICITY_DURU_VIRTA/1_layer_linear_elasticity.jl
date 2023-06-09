include("2d_elasticity_problem.jl");

"""
The stiffness term (K) in the elastic wave equation
Ü = -K*U + (f + g)
"""
function stima(sbp_2d, pterms)
  (𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), _ = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms  
  # The second derivative SBP operator
  Ac = collect(A(@SVector[0.0,0.0]))
  Bc = collect(B(@SVector[0.0,0.0]))
  Cc = collect(C(@SVector[0.0,0.0]))
  Cᵀc = collect(Cᵀ(@SVector[0.0,0.0]))
  𝐃𝐪𝐪ᴬ = Ac ⊗ 𝐃𝐪𝐪
  𝐃𝐫𝐫ᴮ = Bc ⊗ 𝐃𝐫𝐫
  𝐃𝐪C𝐃𝐫 = (I(2) ⊗ 𝐃𝐪) * (Cc ⊗ 𝐃𝐫)
  𝐃𝐫Cᵗ𝐃𝐪 = (I(2) ⊗ 𝐃𝐫) * (Cᵀc ⊗ 𝐃𝐪)  
  # The Elastic wave-equation operators
  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪) # The bulk term
  𝐓𝐪₀ = -(Ac ⊗ 𝐒𝐪 + Cc ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫₀ = -(Cᵀc ⊗ 𝐃𝐪 + Bc ⊗ 𝐒𝐫) # The vertical traction operator
  𝐓𝐪ₙ = (Ac ⊗ 𝐒𝐪 + Cc ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫ₙ = (Cᵀc ⊗ 𝐃𝐪 + Bc ⊗ 𝐒𝐫) # The vertical traction operator
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

  bq₀ = sparsevec(eltocols(𝐈q₀*g₀.(XY, t))) # q (x) = 0  
  br₀ = sparsevec(eltocols(𝐈r₀*g₁.(XY, t))) # r (y) = 0
  bqₙ = sparsevec(eltocols(𝐈qₙ*g₂.(XY,t))) # q (x) = 1
  brₙ = sparsevec(eltocols(𝐈rₙ*g₃.(XY,t))) # r (y) = 1

  collect(τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ + τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end

#################################
# Now begin solving the problem #
#################################
# Discretize the domain
domain = (0.0,1.0,0.0,1.0);
M = 101; # No of points along the axes
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
lu𝐊 = factorize(𝐌 - (Δt/2)^2*𝐊)
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
    u₁,v₁ = CN(lu𝐊, 𝐊, 𝐌, fargs)    
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
e = sol - eltocols(U.(XY,tf))
@show sqrt(e'*𝐇*e)

function UV(sol)
  _2M² = length(sol)
  M² = Int(_2M²/2)
  M = Int(sqrt(M²))
  (reshape(sol[1:M²],(M,M)), reshape(sol[M²+1:end], (M,M)))
end

## Visualize the solution
Uap, Vap = UV(sol)
Ue, Ve = UV(reduce(hcat,U.(XY,tf))')
plt1 = contourf(q, r, Uap, title="u₁ Approximate")
plt2 = contourf(q, r, Ue, title="u₁ Exact")
plt3 = contourf(q, r, Vap, title="v₁ Approximate")
plt4 = contourf(q, r, Ve, title="v₁ Exact")
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(400,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(400,800));