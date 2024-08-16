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

"""
The fourth-order Runge-Kutta scheme with an external forcing
"""
function RK4_1!(MK, sol, Δt, F, M)  
  X₀, k₁, k₂, k₃, k₄ = sol
  F₁, F₂, F₄ = F
  k₁ .= MK*(X₀) + M*F₁
  k₂ .= MK*(X₀+0.5*Δt*k₁) + M*F₂
  k₃ .= MK*(X₀+0.5*Δt*k₂) + M*F₂
  k₄ .= MK*(X₀+Δt*k₃) + M*F₄
  X₀ .+= (Δt/6)*(k₁ + 2*k₂ + 2*k₃ + k₄)
end