"""
The 4th order Runge Kutta scheme
"""
function RK4!(res::Vector{T}, Δt::Float64, t::Float64, u::Vector{T}, kwargs) where T<:Number
  k₁ = f(t, u, kwargs)
  k₂ = f(t + 0.5*Δt, u + 0.5*Δt*k₁, kwargs)
  k₃ = f(t + 0.5*Δt, u + 0.5*Δt*k₂, kwargs)
  k₄ = f(t + Δt, u + Δt*k₃, kwargs)    
  res .= (u + (Δt)/6*(k₁ + 2k₂ + 2k₃ + k₄))  
end