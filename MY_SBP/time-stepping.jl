"""
The 4th order Runge Kutta scheme (for vₜ = f(t,v))
"""
function RK4!(res::Vector{T}, g::Function, args::Tuple{T, T, Vector{T}, Vector{T}}, kwargs) where T<:Number
  Δt, t, u, F = args
  k₁ = g(t, u, F, kwargs)
  k₂ = g(t + 0.5*Δt, u + 0.5*Δt*k₁, F, kwargs)
  k₃ = g(t + 0.5*Δt, u + 0.5*Δt*k₂, F, kwargs)
  k₄ = g(t + Δt, u + Δt*k₃, F, kwargs)    
  res .= (u + (Δt)/6*(k₁ + 2k₂ + 2k₃ + k₄))  
end

"""
The Crank-Nicolson scheme (vₜₜ = f(t,v))
"""
function CN(K::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64, Int64}, args::Tuple{T, Vector{T}, Vector{T}, Vector{T}}) where T<:Number  
  Δt, u, v, F = args  
  M⁺ = (M + (Δt/2)^2*K)
  M⁻ = (M - (Δt/2)^2*K)  
  u₁ = M⁺\(M⁻*u + Δt*M*v + (Δt)^2/4*F)
  v₁ = -v + (2/Δt)*(u₁ - u)
  (u₁, v₁)
end