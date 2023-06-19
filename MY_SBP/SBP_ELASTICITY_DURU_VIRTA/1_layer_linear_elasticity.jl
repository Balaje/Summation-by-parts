# include("tests.jl");
include("2d_elasticity_problem.jl");

"""
The stiffness term (K) in the elastic wave equation
Ü = -K*U + (f + g)
"""
function stima(q, r, sbp_2d, pterms)
  QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
  𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹ = sbp_2d[3]  
  τ₀, τ₁, τ₂, τ₃ = pterms   
  
  # The second derivative SBP operator
  𝐃𝐪𝐪ᴬ = 𝐃𝐪𝐪2d(Aₜ, QR)
  𝐃𝐫𝐫ᴮ = 𝐃𝐫𝐫2d(Bₜ, QR)
  𝐃𝐪C𝐃𝐫, 𝐃𝐫Cᵗ𝐃𝐪 = 𝐃𝐪𝐫𝐃𝐫𝐪2d(Cₜ, QR, sbp_2d)  
  𝐓𝐪, 𝐓𝐫 = 𝐓𝐪𝐓𝐫2d(Aₜ, Bₜ, Cₜ, QR, sbp_2d) # The unsigned traction operator
  # The Elastic wave-equation operators
  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪) # The bulk term  

  # The "stiffness term"  
  𝐏 - (-τ₀*𝐇𝐫₀⁻¹*𝐓𝐫 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫 - τ₂*𝐇𝐪₀⁻¹*𝐓𝐪 + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪) 
end

"""
The boundary contribution terms g
  Ü = -K*U + (f + g)
Applied into the load vector during time stepping
"""
function nbc(t::Float64, q, r, pterms, sbp_2d)  
  τ₀, τ₁, τ₂, τ₃ = pterms

  𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹ = sbp_2d[3]

  M = length(q)

  @inline function E1(i,M)
    res = spzeros(M)
    res[i] = 1.0
    res
  end

  bvals_q₀ = reduce(hcat, [J⁻¹s(𝒮, @SVector[0.0, rᵢ], @SVector[-1.0,0.0])*g(t, c₀, rᵢ, 1) for rᵢ in r]) # q = 0  
  bvals_r₀ = reduce(hcat, [J⁻¹s(𝒮, @SVector[qᵢ, 0.0], @SVector[0.0,-1.0])*g(t, c₁, qᵢ, -1) for qᵢ in q]) # r = 0
  bvals_qₙ = reduce(hcat, [J⁻¹s(𝒮, @SVector[1.0, rᵢ], @SVector[1.0,0.0])*g(t, c₂, rᵢ, -1) for rᵢ in r]) # q = 1
  bvals_rₙ = reduce(hcat, [J⁻¹s(𝒮, @SVector[qᵢ, 1.0], @SVector[0.0,1.0])*g(t, c₃, qᵢ, 1) for qᵢ in q])  # r = 1  
  
  bq₀ = (E1(1,2) ⊗ E1(1,M) ⊗ (bvals_q₀[1,:])) + (E1(2,2) ⊗ E1(1,M) ⊗ (bvals_q₀[2,:]))
  br₀ = (E1(1,2) ⊗ (bvals_r₀[1,:]) ⊗ E1(1,M)) + (E1(2,2) ⊗ (bvals_r₀[2,:]) ⊗ E1(1,M))
  bqₙ = (E1(1,2) ⊗ E1(M,M) ⊗ (bvals_qₙ[1,:])) + (E1(2,2) ⊗ E1(M,M) ⊗ (bvals_qₙ[2,:]))
  brₙ = (E1(1,2) ⊗ (bvals_rₙ[1,:]) ⊗ E1(M,M)) + (E1(2,2) ⊗ (bvals_rₙ[2,:]) ⊗ E1(M,M))

  collect(τ₀*𝐇𝐫₀⁻¹*br₀ + τ₁*𝐇𝐫ₙ⁻¹*brₙ + τ₂*𝐇𝐪₀⁻¹*bq₀ + τ₃*𝐇𝐪ₙ⁻¹*bqₙ)
end

#################################
# Now begin solving the problem #
#################################
# Discretize the domain
domain = (0.0,1.0,0.0,1.0);
𝒩 = [21,31]
h = 1 ./(𝒩 .- 1)
L²Error = zeros(Float64,length(𝒩))

for (M,i) in zip(𝒩,1:length(𝒩)) 
  let
    global q = LinRange(0,1,M);
    global r = LinRange(0,1,M);  
    global 𝐐𝐑 = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
    global XY = 𝒮.(𝐐𝐑)
    detJ = (det∘J).(𝒮, 𝐐𝐑)  
    # Get the SBP matrices
    global sbp_1d = SBP(M);
    global sbp_2d = SBP_2d(sbp_1d);
    # Penalty terms for applying the boundary conditions using the SAT method
    τ₀ = τ₁ = τ₂ = τ₃ = 1.0;
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
    𝐊 = stima(q, r, sbp_2d, pterms)
    Jᵢρᵢ = detJ*ρ
    𝐌 = I(2) ⊗ spdiagm(Jᵢρᵢ)
    𝐌⁻ = (𝐌 + (Δt/2)^2*𝐊);
    lu𝐊 = factorize(𝐌 - (Δt/2)^2*𝐊);

    let
      u₀ = eltocols(U.(XY,0))
      v₀ = eltocols(Uₜ.(XY,0))
      
      # Crank Nicolson Method
      global u₁ = zero(u₀)  
      global v₁ = zero(v₀) 
      t = 0.0
      for i=1:ntime   
        Fₙ = eltocols(detJ .* F.(XY, t))
        Fₙ₊₁ = eltocols(detJ .* F.(XY, t+Δt))
        gₙ = nbc(t, q, r, pterms, sbp_2d)
        gₙ₊₁ = nbc(t+Δt, q, r, pterms, sbp_2d)

        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(lu𝐊, 𝐌⁻, 𝐌, fargs)    
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
        #(i % 100 == 0) && println("Done t="*string(t))
      end   
      global sol = u₁  
    end;

    # Compute the L²Error
    H = sbp_1d[1][1]
    𝐇 = (I(2) ⊗ H ⊗ H)*(I(2) ⊗ spdiagm(detJ))
    e = sol - eltocols(U.(XY,tf))
    L²Error[i] = sqrt(e'*𝐇*e)
    println("Done N = "*string(M)*", L²Error = "*string(L²Error[i]))
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