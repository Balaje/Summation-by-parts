include("2d_elasticity_problem.jl");

"""
The stiffness term (K) in the elastic wave equation
Ü = -K*U + (f + g)
"""
function stima(sbp_2d, pterms)
  (𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), _, _ = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms  
  # The second derivative SBP operator
  Ac = collect(A(@SVector[0.0,0.0]))
  Bc = collect(B(@SVector[0.0,0.0]))
  Cc = collect(C(@SVector[0.0,0.0]))
  Cᵀc = collect(Cᵀ(@SVector[0.0,0.0]))
  𝐃𝐪𝐪ᴬ = Ac ⊗ 𝐃𝐪𝐪
  𝐃𝐫𝐫ᴮ = Bc ⊗ 𝐃𝐫𝐫
  𝐃𝐪C𝐃𝐫 = (I(2) ⊗ 𝐃𝐪) * (Cc ⊗ 𝐃𝐫)
  𝐃𝐫Cᵗ𝐃𝐪 = (I(2) ⊗ 𝐃𝐫) * (Cᵀc ⊗ 𝐃𝐪)  
  # The Elastic wave-equation operators
  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪) # The bulk term
  𝐓𝐪₀ = -(Ac ⊗ 𝐒𝐪 + Cc ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫₀ = -(Cᵀc ⊗ 𝐃𝐪 + Bc ⊗ 𝐒𝐫) # The vertical traction operator
  𝐓𝐪ₙ = (Ac ⊗ 𝐒𝐪 + Cc ⊗ 𝐃𝐫) # The horizontal traction operator
  𝐓𝐫ₙ = (Cᵀc ⊗ 𝐃𝐪 + Bc ⊗ 𝐒𝐫) # The vertical traction operator
  # The "stiffness term"  
  𝐏 - (τ₀*𝐇𝐫₀⁻¹*𝐓𝐫₀ + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫ₙ + τ₂*𝐇𝐪₀⁻¹*𝐓𝐪₀ + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪ₙ) 
end

"""
The boundary contribution terms g
  Ü = -K*U + (f + g)
Applied into the load vector during time stepping
"""
function nbc(t::Float64, sbp_2d, pterms)
  _, _, (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), (𝐈q₀a, 𝐈r₀a, 𝐈qₙa, 𝐈rₙa), (XYq₀, XYr₀, XYqₙ, XYrₙ) = sbp_2d
  τ₀, τ₁, τ₂, τ₃ = pterms

  M = Int(sqrt(size(𝐇𝐪₀⁻¹,1)/2))

  bvals_q₀ = reduce(hcat, g₀.(XYq₀, t)) # q (x) = 0  
  bvals_r₀ = reduce(hcat, g₁.(XYr₀, t)) # r (y) = 0
  bvals_qₙ = reduce(hcat, g₂.(XYqₙ, t)) # q (x) = 1
  bvals_rₙ = reduce(hcat, g₃.(XYrₙ, t))  # r (y) = 1  
  
  bq₀ = vec(hcat(sparsevec(𝐈q₀a, bvals_q₀[1,:], M^2), sparsevec(𝐈q₀a, bvals_q₀[2,:], M^2)))
  br₀ = vec(hcat(sparsevec(𝐈r₀a, bvals_r₀[1,:], M^2), sparsevec(𝐈r₀a, bvals_r₀[2,:], M^2)))
  bqₙ = vec(hcat(sparsevec(𝐈qₙa, bvals_qₙ[1,:], M^2), sparsevec(𝐈qₙa, bvals_qₙ[2,:], M^2)))
  brₙ = vec(hcat(sparsevec(𝐈rₙa, bvals_rₙ[1,:], M^2), sparsevec(𝐈rₙa, bvals_rₙ[2,:], M^2)))

  collect(τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ + τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end

#################################
# Now begin solving the problem #
#################################
# Discretize the domain
domain = (0.0,1.0,0.0,1.0);
𝒩 = [11,21,31,41,51]
h = 1 ./(𝒩 .- 1)
L²Error = zeros(Float64,length(𝒩))

for (M,i) in zip(𝒩,1:length(𝒩)) 
  let
    global q = LinRange(0,1,M);
    global r = LinRange(0,1,M);  
    global XY = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
    # Get the SBP matrices
    sbp_1d = SBP(M);
    sbp_2d = SBP_2d(XY, sbp_1d);
    # Penalty terms for applying the boundary conditions using the SAT method
    τ₀ = τ₁ = τ₂ = τ₃ = 1;
    pterms = (τ₀, τ₁, τ₂, τ₃)
    # Begin solving the problem
    # Temporal Discretization parameters
    global tf = 1.25
    Δt = 1e-3
    ntime = ceil(Int64,tf/Δt)
    # Empty Plots
    plt = plot()
    plt1 = plot()

    # Compute the stiffness, mass matrices
    𝐊 = stima(sbp_2d, pterms)
    𝐌 = ρ*spdiagm(ones(2*M^2))
    𝐌⁻ = (𝐌 + (Δt/2)^2*𝐊);
    lu𝐊 = factorize(𝐌 - (Δt/2)^2*𝐊);

    let
      u₀ = eltocols(U.(XY,0))
      v₀ = eltocols(Uₜ.(XY,0))
      #=  
      # Leapfrog scheme
      t = 0.0
      fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
      u₁ = LF1(𝐊, 𝐌⁻¹, (Δt, u₀, fₙ, v₀))
      u₀ = u₁
      t += Δt
      global u₂ = zero(u₀)
      for i=2:ntime
        fₙ = flatten_grid_function(F, QR, t) + BC(t, sbp_2d, pterms)
        u₂ = LF(𝐊, 𝐌⁻¹, (Δt, u₁, u₀, fₙ))
        u₀ = u₁
        u₁ = u₂
        t += Δt    
        (i % 10 == 0) && println("Done t="*string(t)*"\t sum(u₀) = "*string(maximum(abs.(u₀))))
      end
      global sol = u₂ 
      =#
      
      # Crank Nicolson Method
      global u₁ = zero(u₀)  
      global v₁ = zero(v₀) 
      t = 0.0
      for i=1:ntime   
        Fₙ = eltocols(F.(XY, t))
        Fₙ₊₁ = eltocols(F.(XY, t+Δt))
        gₙ = nbc(t, sbp_2d, pterms)
        gₙ₊₁ = nbc(t+Δt, sbp_2d, pterms)

        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(lu𝐊, 𝐌⁻, 𝐌, fargs)    
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
        (i % 100 == 0) && println("Done t="*string(t))
      end   
      global sol = u₁  
    end;

    # Compute the L²Error
    H = sbp_1d[1][1]
    𝐇 = I(2) ⊗ H ⊗ H
    e = sol - eltocols(U.(XY,tf))
    L²Error[i] = sqrt(e'*𝐇*e)
    println("Done N = "*string(M))
  end
end
function UV(sol)
  _2M² = length(sol)
  M² = Int(_2M²/2)
  M = Int(sqrt(M²))
  (reshape(sol[1:M²],(M,M)), reshape(sol[M²+1:end], (M,M)))
end

## Compute the rate and visualize the solution
rate = log.(L²Error[2:end]./L²Error[1:end-1])./log.(h[2:end]./h[1:end-1])
@show L²Error
@show rate

Uap, Vap = UV(sol)
Ue, Ve = UV(reduce(hcat,U.(XY,tf))')
plt1 = contourf(q, r, Uap, title="u₁ Approximate")
plt2 = contourf(q, r, Ue, title="u₁ Exact")
plt3 = contourf(q, r, Vap, title="v₁ Approximate")
plt4 = contourf(q, r, Ve, title="v₁ Exact")
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));

plt5 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
plot!(plt5, h, h.^4, label="O(h⁴)", lw=1);