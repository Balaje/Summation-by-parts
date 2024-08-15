"""
The fourth-order Runge-Kutta scheme
"""
function RK4_1!(M, sol, Δt)  
  X₀, k₁, k₂, k₃, k₄ = sol  
  k₁ .= M*(X₀)
  k₂ .= M*(X₀+0.5*Δt*k₁)
  k₃ .= M*(X₀+0.5*Δt*k₂)
  k₄ .= M*(X₀+Δt*k₃)
  X₀ .+= (Δt/6)*(k₁ + 2*k₂ + 2*k₃ + k₄)
end