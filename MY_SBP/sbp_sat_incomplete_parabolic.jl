##### ###### ###### ###### ###### ###### ###### #
## Example 2 in Mattsson and Nordstrom


include("include.jl")

# Define the problem
const ϵ = 0.1
const Λ = @SMatrix [√(2) 0; 0 -√(2)]
const 𝒟 = ϵ/(2√(2))*(@SMatrix [√(2)-1 1; 1 √(2)+1]) # which is ̃D in the paper, i.e., the diffusion tensor
const α = √(2) - 1
# Some exact solution for testing
const w = 10
const b = 1
const c = 1
v(x,t) =  [-sin(w*(x-c*t))*exp(-b*x), sin(w*(x+c*t))*exp(-b*x)]'

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
  ∂ₓv₀ = @SVector [-w*cos(w*c*t) + b*sin(w*c*t), w*cos(w*c*t) - b*sin(w*c*t)]
  v₁ = @SVector [-sin(w*(1-c*t))*exp(-b); sin(w*(1+c*t))*exp(-b)]
  ∂ₓv₁ = @SVector [-exp(-b)*w*cos(w*(1-c*t)) + b*exp(-b)*sin(w*(1-c*t)), exp(-b)*w*cos(w*(1+c*t)) - b*exp(-b)*sin(w*(1+c*t))]
  (I₀*Λ*v₀ + (I₁ - I₀)*𝒟*∂ₓv₀, I₁*Λ*v₁ + (I₀ - (I₂+α*Iᵣ)*I₁)*𝒟*∂ₓv₁)
end

"""
The non-zero forcing term in the RHS of the PDE
"""
function F(x,t)
  ∂ₜv = @SVector [w*c*cos(w*(x-c*t))*exp(-b*x), w*c*cos(w*(x+c*t))*exp(-b*x)]
  ∂ₓv = @SVector [-exp(-b*x)*w*cos(w*(x-c*t)) + b*exp(-b*x)*sin(w*(x-c*t)), 
                  exp(-b*x)*w*cos(w*(x+c*t)) - b*exp(-b*x)*sin(w*(x+c*t))]
  ∂ₓₓv = @SVector [-exp(-b*x)*w*(-w*sin(w*(x-c*t))) + b*exp(-b*x)*w*cos(w*(x-c*t)) + b*(-b*exp(-b*x)*sin(w*(x-c*t))+exp(-b*x)*w*cos(w*(x-c*t))), 
                   exp(-b*x)*w*(-w*sin(w*(x+c*t))) - b*exp(-b*x)*w*cos(w*(x+c*t)) - b*(-b*exp(-b*x)*sin(w*(x+c*t))+exp(-b*x)*w*cos(w*(x+c*t)))]
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
  L₁ = (I₁*Λ) ⊗ Eₙ + (I₀ - (I₂ + α*Iᵣ)*I₁*𝒟) ⊗ (Eₙ*S)

  g₀, g₁ = BoundaryData(t)

  -(Λ ⊗ D1)*v + (𝒟 ⊗ D2)*v + F + (Σ₀ ⊗ Hinv)*(L₀*v - g₀ ⊗ e₀) + (Σ₁ ⊗ Hinv)*(L₁*v - g₁ ⊗ eₙ)
end


# Temporal Discretization parameters
tf = 0.1
Δt = 5e-4
ntime = ceil(Int64,tf/Δt)
# Spatial discretization parameters
n = 20;
Σ₀ = [-1 0; 0 1];
Σ₁ = [-1 α; 0 1];
pterms = Σ₀, Σ₁;
plt = plot()
plt1 = plot()
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
    plot!(plt, x, v₁[1:n+1], lc=:blue, lw=1, label="Approx. solution (v⁽¹⁾) n="*string(n))    
    plot!(plt1, x, v₁[n+2:end], lc=:blue, lw=1, label="Approx. solution v⁽²⁾ n="*string(n))    
  end  
  vex = vec(reduce(vcat, v.(x,tf)))
  plot!(plt, x, vex[1:n+1], lc=:black, lw=2, label="Exact solution (v⁽¹⁾)", ls=:dash)
  plot!(plt1, x, vex[n+2:end], lc=:black, lw=2, label="Exact solution (v⁽²⁾)", ls=:dash)
end