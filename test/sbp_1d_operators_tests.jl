using SparseArrays
using LinearAlgebra
using StaticArrays
using Test

# Load the package
using SummationByPartsPML

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Test out some 1D SBP operators
# We take a polynomial function f(x) = x^2. 
# This way the 4th order SBP operator is accurate to machine error
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 

f(x) = x^2
f′(x) = 2.0x
f′′(x) = 2.0

SBP4 = SBP4_1D(100)
x = LinRange(0,1,100)
Dₓ = SBP4.D1
Dₓₓ, Dₓₓ′ = SBP4.D2
Sₓ = SBP4.S;
H = SBP4.norm;
𝐈, E₀, Eₙ = SBP4.E

@testset "Testing 1d Identity operator" begin @test 𝐈*f.(x) ≈ f.(x) end;
@testset "Testing 1d First Derivative operator" begin @test Dₓ*f.(x) ≈ f′.(x) end;
@testset "Testing 1d Second Derivative operator" begin @test Dₓₓ*f.(x) ≈ f′′.(x) end;
@testset "Testing 1d Second Derivative operator (Fully compatible)" begin @test Dₓₓ′*f.(x) ≈ f′′.(x) end;
@testset "Testing 1d Boundary Derivative operator" begin @test  Sₓ*f.(x) ≈ (E₀*f′.(x) + Eₙ*f′.(x)) end;

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ######  
# Program to solve the hyperbolic scalar equation 
#   uₜ + a(uₓ) = ϵ(uₓₓ), 0 ≤ x ≤ 1, t > 0,
#     α(u(0,t)) + uₓ(0,t) = g₀(t)
#     β(u(1,t)) + uₓ(1,t) = g₁(t)
# Some sample diagonal norms are shown in Mattsson and Nordstrom, 2004 (Appendix A)
# Let us test out some of the SBP operator. We use the RK4 scheme for the temporal discretization.
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ######  

"""
The 4th order Runge Kutta scheme (for vₜ = f(t,v))
"""
function RK4(g::Function, args::Tuple{T, T, AbstractVector{T}, AbstractVector{T}}, kwargs) where T<:Number
  Δt, t, u, F = args
  k₁ = g(t, u, F, kwargs)
  k₂ = g(t + 0.5*Δt, u + 0.5*Δt*k₁, F, kwargs)
  k₃ = g(t + 0.5*Δt, u + 0.5*Δt*k₂, F, kwargs)
  k₄ = g(t + Δt, u + Δt*k₃, F, kwargs)    
  (u + (Δt)/6*(k₁ + 2k₂ + 2k₃ + k₄))  
end

# Problem parameters
const a = 1.0
const c = 2.0
const ϵ = 0.1
const w = sqrt(c^2 - a^2)/(2ϵ)
const b = (c-a)/(2ϵ)
# Mixed boundary condition on x=0, Neumann boundary cndition on x=1
const α = 1.0
const β = 0.0
u(x,t) = sin(w*(x-c*t))*exp(-b*x)
f(x) = u(x,0)
g₀(t) = (b-α)*sin(w*c*t) + w*cos(w*c*t)
g₁(t) = β*sin(w*(1-c*t))*exp(-b) + exp(-b)*w*cos(w*(1-c*t)) - b*exp(-b)*sin(w*(1-c*t))

"""
The RHS function for the Runge Kutta Iteration
"""
function f(t::Float64, v::AbstractVector{T}, F::AbstractVector{T}, kwargs) where T <: Number  
  coeffs, sbp, τ₀₁, ics = kwargs # Get all the arguments
  a,ϵ,α,β = coeffs # Problem parameters
  τ₀, τ₁ = τ₀₁ # Penalty parameters
  g₀, g₁ = ics # Initial conditions
  # Unpack all the parameters    
  𝐈, E₀, Eₙ = sbp.E; # Identity operators
  H = sbp.norm;    Dₓ = sbp.D1;    Sₓ = sbp.S;    
  Dₓₓ = sbp.D2[1] # Using the non-fully compatible operator for approximating the second derivative 
  H⁻¹ = H\𝐈 # Inverse of the norm    
  e₀ = diag(E₀)
  eₙ = diag(Eₙ)
  𝐠₀ = g₀(t);    𝐠₁ = g₁(t) # Initial conditions
  
  # Paper-like expression with Julia syntax
  -a*Dₓ*v + ϵ*Dₓₓ*v - τ₀*(H⁻¹*(E₀*(α*𝐈+Sₓ)*v - e₀*𝐠₀)) - τ₁*(H⁻¹*(Eₙ*(β*𝐈+Sₓ)*v - eₙ*𝐠₁)) + F
end

# Begin solving the problem
# Temporal Discretization parameters
tf = 1.0
Δt = 5e-5
ntime = ceil(Int64,tf/Δt)
# Penalty parameters
τ₀ = -ϵ
τ₁ = ϵ
# Spatial Discretization

println("");
println("Solving a hyperbolic IBVP (Mattsson and Nordstrom, 2004) with N = [41,81,161,321]...");
println("");

N = [41, 81, 161, 321]  
L²Error = zeros(Float64, length(N)); # Discrete L² Error
for (n,i) ∈ zip(N,1:length(N))
  let
    x = LinRange(0,1,n)
    # Get the SBP operators using the package
    sbp = SBP4_1D(n) 
    # Norm matrix to compute the error
    H = sbp.norm 
    let
      u₀ = f.(x);  t = 0.0
      # Time loop
      for i=1:ntime        
        u₀ = RK4(f, (Δt, t, u₀, zero(x)), ((a,ϵ,α,β), sbp, (τ₀, τ₁), (g₀, g₁)))    
        t = t+Δt
      end                  
      e = u.(x,tf) - u₀
      L²Error[i] = sqrt(e'*H*e)
    end
    println("Done n = "*string(n))
  end
end

println("");
println("L²Error = ", L²Error)
rate = log2.(L²Error[1:end-1]./L²Error[2:end]);
println("Rate = ", rate)
println("");

# Check if the rate is approximately 4
@testset "Check if rate is approximately 4 after solving the 1d hyperbolic problem." begin @test abs(sum(rate)/length(rate) - 4.0) < 0.2 end;