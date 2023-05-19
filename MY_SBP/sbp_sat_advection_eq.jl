###### ######  ###### ######  ###### ######  ###### ######  ###### ######  ###### ######  
# Program to solve the hyperbolic scalar equation 
#   uₜ + a(uₓ) = ϵ(uₓₓ), 0 ≤ x ≤ 1, t > 0,
#   α(u(0,t)) + uₓ(0,t) = g₀(t)
#   β(u(1,t)) + uₓ(1,t) = g₁(t)
# Some sample diagonal norms are shown in the paper by Mattsson and Nordstrom (Appendix A)\
# Let us test out some of the SBP operator. 
# The termpral discretization is carried out using the 4th order Runge Kutta scheme
###### ######  ###### ######  ###### ######  ###### ######  ###### ######  ###### ######  

using SparseArrays
using LinearAlgebra
using StaticArrays
using Plots

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

# Spatial discretization
"""
Gets the scheme with second order accuracy at the boundary.
(From the OneDrive code)
"""
function SBP(m::Int64)  
  # H discrete norm matrix
  h = 1/(m-1)
  # h = 1
  H = spdiagm(ones(m))
  H[1:4, 1:4] = spdiagm([17/48, 59/48, 43/48, 49/48])
  H[m-3:m, m-3:m] = rotr90(spdiagm([17/48, 59/48, 43/48, 49/48]), 2)
  
  # D₁ Matrix (Discrete First derivative operator)
  D1 = spdiagm(-2=>(1/12)*ones(m-2), -1=>(-8/12)*ones(m-1),
        1=>(8/12)*ones(m-1), 2=>(-1/12)*ones(m-2))
  D1[1:4, 1:6] = [-24/17 59/34 -4/17 -3/34 0 0; 
                  -1/2 0 1/2 0 0 0; 
                  4/43 -59/86 0 59/86 -4/43 0; 
                  3/98 0 -59/98 0 32/49 -4/49];                    
  D1[m-3:m, m-5:m] = rotr90(-D1[1:4,1:6], 2)  

  # D₂ Matrix (Discrete second derivative operator)
  # D₂ = H⁻¹(-A + BS)
  D2 = spdiagm(-2=>(-1/12)*ones(m-2), -1=>(16/12)*ones(m-1), 
        0=>-30/12*ones(m), 1=>(16/12)*ones(m-1), 2=>(-1/12)*ones(m-2))
  m55 = -5/2
  s15 = 0  
  D2[1:5,1:7] = 
  [154/17+48/17*m55-48/17*s15 -565/17-192/17*m55+192/17*s15 788/17+288/17*m55-288/17*s15 -497/17-192/17*m55+192/17*s15 120/17+48/17*m55-48/17*s15 0 0; 
   -421/59-192/59*m55 1802/59+768/59*m55 -2821/59-1152/59*m55 1920/59+768/59*m55 -480/59-192/59*m55 0 0; 
   716/43+288/43*m55 -2821/43-1152/43*m55 4210/43+1728/43*m55 -2821/43-1152/43*m55 716/43+288/43*m55 0 0; 
  -481/49-192/49*m55 1920/49+768/49*m55 -403/7-1152/49*m55 1802/49+768/49*m55 -416/49-192/49*m55 -4/49 0; 
  5/2+m55 -10-4*m55 179/12+6*m55 -26/3-4*m55 m55 4/3 -1/12];
  D2[m-4:m, m-6:m] = rotr90(D2[1:5,1:7], 2)  

  # BS Matrix (Operator to apprximate the boundary term in the second derivative discrete operator)
  BS=spzeros(Float64,m,m);
  BS[1,1:5] = -[s15-11/6,-4*s15+3,6*s15-3/2,-4*s15+1/3,s15];
  BS[m,m-4:m] = reverse(-[s15-11/6,-4*s15+3,6*s15-3/2,-4*s15+1/3,s15], dims=1)

  # Scale the operators appropriately with the mesh size
  H = h*H
  BS = BS/h
  D2 = D2/(h^2)
  D1 = D1/h
 
  # Compatible second derivate D2c
  M = BS-H*D2
  S = BS
  S[1,:] = -S[1,:]
  for i=2:size(S,1)-1
    S[i,i] = 0;    
  end
  B = spzeros(Float64,m,m)
  B[1,1] = -1
  B[m,m] = 1
  D2c = H\(-M+B*D1)

  E₀ = spzeros(Float64,m,m)
  Eₙ = spzeros(Float64,m,m)
  E₀[1,1] = 1.0  
  Eₙ[m,m] = 1.0
  e₀ = diag(E₀)
  eₙ = diag(Eₙ)
  Id = I(length(e₀))

  Hinv = H\Id

  (H, Hinv), D1, (D2,D2c), S, (E₀, Eₙ, e₀, eₙ, Id)
end
function SBP(k::Int64, N::Int64, domain::Tuple{Float64,Float64})
  display("To be implemented")
end

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

"""
The Runge Kutta scheme
"""
function RK4!(res::Vector{T}, Δt::Float64, t::Float64, u::Vector{T}, kwargs) where T<:Number
  k₁ = f(t, u, kwargs)
  k₂ = f(t + 0.5*Δt, u + 0.5*Δt*k₁, kwargs)
  k₃ = f(t + 0.5*Δt, u + 0.5*Δt*k₂, kwargs)
  k₄ = f(t + Δt, u + Δt*k₃, kwargs)    
  res .= (u + (Δt)/6*(k₁ + 2k₂ + 2k₃ + k₄))  
end

# Discretization parameters
N = 51
x = LinRange(0,1,N)
sbp = SBP(N);
τ₀ = -ϵ
τ₁ = ϵ
args = (a,ϵ,α,β), sbp, (τ₀, τ₁), (g₀, g₁)
tf = 1.0
Δt = 1e-3
ntime = ceil(Int64,tf/Δt)
plt = plot()
let
  u₀ = f.(x)
  global u₁ = zero(u₀)  
  t = 0.0
  for i=1:ntime
    u₀ = RK4!(u₁, Δt, t, u₀, args)    
    t = t+Δt
  end
  plot!(plt, x, u.(x,tf), lc=:red, lw=2, ls=:dash, label="Exact solution")
  plot!(plt, x, u₁, lc=:blue, lw=0.5, label="Approximate Solution")
end