"""
The 4th order Runge Kutta scheme
"""
function RK4!(res::Vector{T}, g::Function, args::Tuple{T, T, Vector{T}, Vector{T}}, kwargs) where T<:Number
  Δt, t, u, F = args
  k₁ = g(t, u, F, kwargs)
  k₂ = g(t + 0.5*Δt, u + 0.5*Δt*k₁, F, kwargs)
  k₃ = g(t + 0.5*Δt, u + 0.5*Δt*k₂, F, kwargs)
  k₄ = g(t + Δt, u + Δt*k₃, F, kwargs)    
  res .= (u + (Δt)/6*(k₁ + 2k₂ + 2k₃ + k₄))  
end