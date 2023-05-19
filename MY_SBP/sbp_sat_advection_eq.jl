###### ######  ###### ######  ###### ######  ###### ######  ###### ######  ###### ######  
# Program to solve the hyperbolic scalar equation 
#   uₜ + a(uₓ) = ϵ(uₓₓ), 0 ≤ x ≤ 1, t > 0,
#   α(u(0,t)) + uₓ(0,t) = g₀(t)
#   β(u(1,t)) + uₓ(1,t) = g₁(t)
# Some sample diagonal norms are shown in the paper by Mattsson and Nordstrom (Appendix A)\
# Let us test out some of the SBP operator. 
# The termpral discretization is carried out using the 4th order Runge Kutta scheme
###### ######  ###### ######  ###### ######  ###### ######  ###### ######  ###### ######  

include("include.jl")

# Define the problem
const a = 1.0
const c = 2.0
const ϵ = 0.1
const w = sqrt(c^2 - a^2)/(2ϵ)
const b = (c-a)/(2ϵ)
# Mixed boundary condition on x=0, Neumann boundary cndition on x=1
const α = 1
const β = 0
u(x,t) = sin(w*(x-c*t))*exp(-b*x)
f(x) = u(x,0)
g₀(t) = (b-α)*sin(w*c*t) + w*cos(w*c*t)
g₁(t) = β*sin(w*(1-c*t))*exp(-b) + exp(-b)*w*cos(w*(1-c*t)) - b*exp(-b)*sin(w*(1-c*t))
"""
The RHS function for the Runge Kutta Iteration
"""
function f(t::Float64, v::AbstractVector{T}, kwargs) where T <: Number  
  coeffs, sbp, τ₀₁, ics = kwargs
  a,ϵ,α,β = coeffs
  τ₀, τ₁ = τ₀₁
  g₀, g₁ = ics
  HHinv, D1, D2s, S, unit_vecs = sbp
  _, Hinv = HHinv  
  E₀, Eₙ, e₀, eₙ, Id = unit_vecs
  D2, _ = D2s
  g0 = g₀(t)
  g1 = g₁(t)
  -a*D1*v + ϵ*D2*v - τ₀*(Hinv*(E₀*(α*Id+S)*v - e₀*g0)) - τ₁*(Hinv*(Eₙ*(β*Id+S)*v - eₙ*g1))
end

# Begin solving the problem
# Temporal Discretization parameters
tf = 1.0
Δt = 5e-5
ntime = ceil(Int64,tf/Δt)
# Penalty parameters
const τ₀ = -ϵ
const τ₁ = ϵ
# Plots
plt = plot()
plt1 = plot()
# Spatial Discretization
N = [40, 60, 100, 200, 300]  
L²Error = zeros(Float64, size(N,1))
for (n,i) ∈ zip(N,1:length(N))
  let
    x = LinRange(0,1,n+1)
    sbp = SBP(n+1);
    H = sbp[1][1]; # Norm matrix for the l2error
    args = (a,ϵ,α,β), sbp, (τ₀, τ₁), (g₀, g₁)
    let
      u₀ = f.(x)
      global u₁ = zero(u₀)  
      t = 0.0
      for i=1:ntime
        u₀ = RK4!(u₁, Δt, t, u₀, args)    
        t = t+Δt
      end      
      plot!(plt, x, u₁, lc=:blue, lw=0.5, label="Approx. solution n="*string(n))

      e = u.(x,tf) - u₁
      L²Error[i] = sqrt(e'*H*e)
    end
    println("Done n = "*string(n))
  end
end
plot!(plt, x, u.(x,tf), lc=:red, lw=2, ls=:dash, label="Exact solution")
plot!(plt, x, f.(x), lc=:black, lw=1, ls=:dash, label="Initial condition")
rate = log.(L²Error[2:end]./L²Error[1:end-1])./(log.(N[1:end-1]./N[2:end]))
plot!(plt1, 1 ./(N), L²Error, xscale=:log10, yscale=:log10, label="L²Error")
plot!(plt1, 1 ./(N), 1 ./(N).^4, ls=:dash, lw=0.5, label="h⁴")
### ### ### ### ### ### ### ### 
# Running this gives
# julia> rate
# 4-element Vector{Float64}:
#  4.206668554230709
#  4.184137602101776
#  4.12890189205842
#  4.083294430882183
### ### ### ### ### ### ### ### 