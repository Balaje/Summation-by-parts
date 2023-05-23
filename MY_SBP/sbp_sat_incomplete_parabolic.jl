##### ###### ###### ###### ###### ###### ###### ###### ######
## Refer Mattsson and Nordstrom, 2004
## Consider the incomplete parabolic system
##  vₜ + Cvₓ = Dvₓₓ,  0<x<1, t>0
##    v(x,0) = f(x), 0<x<1, t=0
##    L₀v = g₀(t), x=0, t>0
##    L₁v = g₁(t), x=0, t>0
## Here C = [1 1; 1 -1], D = [0 0; 0 ϵ], v = [v⁽¹⁾, v⁽²⁾]ᵀ
##### ###### ###### ###### ###### ###### ###### ###### ######


include("include.jl")

# Define the problem
const ϵ = 1.0
const Λ = @SMatrix [√(2) 0; 0 -√(2)]
const 𝒟 = ϵ/(2√(2))*(@SMatrix [√(2)-1 1; 1 √(2)+1]) # which is D̃ in the paper, i.e., the diffusion tensor
const α = √(2) - 1
# Some exact solution for testing
const w = 10.0
const b = 1.0
const c = 1.0
v(x,t) =  (@SVector [-sin(w*(x-c*t))*exp(-b*x), sin(w*(x+c*t))*exp(-b*x)])'

"""
Function to compute the boundary data at given time t:
Returns a tuple that corresponds to 
  (g₀(t), g₁(t))
"""
function BoundaryData(t)
  I₀ = @SMatrix [1 0; 0 0];
  I₁ = @SMatrix [0 0; 0 1];
  I₂ = @SMatrix [1 0; 0 1];
  Iᵣ = @SMatrix [0 1; 1 0];
  v₀ = @SVector [sin(w*c*t), sin(w*c*t)]
  ∂ₓv₀ = @SVector [- b*sin(c*t*w) - w*cos(c*t*w), w*cos(c*t*w) - b*sin(c*t*w)]
  v₁ = @SVector [-sin(w*(1-c*t))*exp(-b); sin(w*(1+c*t))*exp(-b)]
  ∂ₓv₁ = @SVector [- b*exp(-b)*sin(w*(c*t - 1)) - w*exp(-b)*cos(w*(c*t - 1)), 
                  w*exp(-b)*cos(w*(c*t + 1)) - b*exp(-b)*sin(w*(c*t + 1))]

  (I₀*Λ*v₀ + (I₁ - I₀)*𝒟*∂ₓv₀, I₁*Λ*v₁ + (I₀ - (I₂+α*Iᵣ)*I₁)*𝒟*∂ₓv₁) # Return
end

"""
The non-zero forcing term in the RHS of the PDE
"""
function F(x,t)
  ∂ₜv = @SVector [c*w*cos(w*(x - c*t))*exp(-b*x), c*w*cos(w*(x + c*t))*exp(-b*x)]
  ∂ₓv = @SVector [b*sin(w*(x - c*t))*exp(-b*x) - w*cos(w*(x - c*t))*exp(-b*x), 
                  w*cos(w*(x + c*t))*exp(-b*x) - b*sin(w*(x + c*t))*exp(-b*x)]
  ∂ₓₓv = @SVector [w^2*sin(w*(x - c*t))*exp(-b*x) - b^2*sin(w*(x - c*t))*exp(-b*x) + 2*b*w*cos(w*(x - c*t))*exp(-b*x), 
                   b^2*sin(w*(x + c*t))*exp(-b*x) - w^2*sin(w*(x + c*t))*exp(-b*x) - 2*b*w*cos(w*(x + c*t))*exp(-b*x)]
  (∂ₜv + Λ*∂ₓv - 𝒟*∂ₓₓv)'
end

"""
Initial condition function
"""
V₀(x) = v(x,0)

"""
Fancy definition of the Kronecker product.
"""
⊗(A, B) = Base.kron(A,B)

# Begin solving the problem
"""
RHS of the discrete time-stepping
"""
function g(t::Float64, v::AbstractVector{T}, F::AbstractVector{T}, kwargs) where T <: Number  
  sbp, pterms = kwargs
  Σ₀, Σ₁ = pterms
  HHinv, D1, D2s, S, unit_vecs = sbp
  _, Hinv = HHinv  
  E₀, Eₙ, e₀, eₙ, _ = unit_vecs
  D2, _ = D2s

  I₀ = @SMatrix [1 0; 0 0];
  I₁ = @SMatrix [0 0; 0 1];
  I₂ = @SMatrix [1 0; 0 1];
  Iᵣ = @SMatrix [0 1; 1 0];
  L₀ = (I₀*Λ) ⊗ E₀ + ((I₁ - I₀)*𝒟) ⊗ (E₀*S)
  L₁ = (I₁*Λ) ⊗ Eₙ + ((I₀ - (I₂ + α*Iᵣ)*I₁)*𝒟) ⊗ (Eₙ*S)

  g₀, g₁ = BoundaryData(t)

  -(Λ ⊗ D1)*v + (𝒟 ⊗ D2)*v + F + (Σ₀ ⊗ Hinv)*(L₀*v - g₀ ⊗ e₀) + (Σ₁ ⊗ Hinv)*(L₁*v - g₁ ⊗ eₙ) # Return
end


# Temporal Discretization parameters
Δt = 1e-5
tf = 1e-1
ntime = ceil(Int64,tf/Δt)
# Spatial discretization parameters
Σ₀ = [-1 0; 0 1];
Σ₁ = [-1 α; 0 1];
pterms = Σ₀, Σ₁;
plt = plot()
plt1 = plot()
N = [30,60,90,120,150]
L²Error = zeros(Float64,length(N))
for (n,i) ∈ zip(N,1:length(N))
  let
    x = LinRange(0,1,n+1)
    sbp = SBP(n+1);
    H = sbp[1][1]; # Norm matrix for the l2error
    args = sbp, pterms
    let
      v₀ = vec(reduce(vcat, V₀.(x)))
      global v₁ = zero(v₀)  
      t = 0.0
      for i=1:ntime
        Fvec = vec(reduce(vcat, F.(x,t)))
        fargs = Δt, t, v₀, Fvec
        v₀ = RK4!(v₁, g, fargs, args)    
        t = t+Δt
        (i % 1000 == 0) && println("Done t="*string(t))
      end  
      vex = vec(reduce(vcat, v.(x, t)))      
      e = vex - v₁
      HI = I(2) ⊗ H
      L²Error[i] = sqrt(e'*HI*e)         
    end      
  end
  println("Done n = "*string(n))
  println(" ")
end

scatter!(plt, LinRange(0,1,N[end]+1), v₁[1:N[end]+1], label="Approx. solution (v⁽¹⁾) n="*string(N[end]))    
scatter!(plt1, LinRange(0,1,N[end]+1), v₁[N[end]+2:end], label="Approx. solution v⁽²⁾ n="*string(N[end])) 
xplot = 0:0.01:1
vex = vec(reduce(vcat, v.(xplot,tf)))
plot!(plt, xplot, vex[1:length(xplot)], lc=:black, lw=2, label="Exact solution (v⁽¹⁾)", ls=:dash)
plot!(plt1, xplot, vex[length(xplot)+1:end], lc=:black, lw=2, label="Exact solution (v⁽²⁾)", ls=:dash)
rate = log.(L²Error[2:end]./L²Error[1:end-1])./(log.(N[1:end-1]./N[2:end]))