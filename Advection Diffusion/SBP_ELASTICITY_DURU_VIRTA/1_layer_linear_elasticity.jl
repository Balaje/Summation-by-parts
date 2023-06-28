# include("tests.jl");
include("2d_elasticity_problem.jl");

"""
The stiffness term (K) in the elastic wave equation
ρÜ = -K*U + (f(t) + g(t))
"""
function 𝐊(q, r, sbp_2d, pterms)
  QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
  𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹ = sbp_2d[3]  
  τ₀, τ₁, τ₂, τ₃ = pterms   
  
  # The second derivative SBP operator
  𝐃𝐪𝐪ᴬ = 𝐃𝐪𝐪2d(Aₜ, QR, 𝒮)
  𝐃𝐫𝐫ᴮ = 𝐃𝐫𝐫2d(Bₜ, QR, 𝒮)
  𝐃𝐪C𝐃𝐫, 𝐃𝐫Cᵗ𝐃𝐪 = 𝐃𝐪𝐫𝐃𝐫𝐪2d(Cₜ, QR, sbp_2d, 𝒮)  
  𝐓𝐪, 𝐓𝐫 = 𝐓𝐪𝐓𝐫2d(Aₜ, Bₜ, Cₜ, QR, sbp_2d) # The "unsigned" traction operator
  # The Elastic wave-equation operators
  𝐏 = (𝐃𝐪𝐪ᴬ + 𝐃𝐫𝐫ᴮ + 𝐃𝐪C𝐃𝐫 + 𝐃𝐫Cᵗ𝐃𝐪) # The bulk term  

  # The "stiffness term"  
  detJ = [1,1] ⊗ (det∘J).(𝒮,QR) # The determinant of the transformation
  # The signed version of the traction.
  spdiagm(detJ.^-1)*𝐏 - (-τ₀*𝐇𝐫₀⁻¹*𝐓𝐫 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫 - τ₂*𝐇𝐪₀⁻¹*𝐓𝐪 + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪) 
end

"""
The Neumann boundary contribution terms g(t)
  ρÜ = -K*U + (f(t) + g(t))
Applied into the load vector during time stepping
"""
function 𝐠(t::Float64, q, r, pterms, sbp_2d)  
  τ₀, τ₁, τ₂, τ₃ = pterms
  
  𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹ = sbp_2d[3]

  M = length(q)

  @inline function E1(i,M)
    res = spzeros(M)
    res[i] = 1.0
    res
  end

  # Get the boundary condition vectors. 
  # The function (Defined in geometry.jl)
  #     J⁻¹s(𝒮, (q,r), nᵣ) 
  # gives the surface Jacobian which will be multiplied with the Neumann data. 
  # Here nᵣ is the normal in the reference domain.
  # 
  # The function
  #     g(t, c ∈ {c₀, c₁, c₂, c₃}, t, [1 or -1])
  # gives the value of (σ⋅n) on the curve c. 
  # The 1 or -1 depends on how the surface normal is oriented. I traverse counter-clockwise, so the normal is always 
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
𝒩 = [21]
h = 1 ./(𝒩 .- 1)
L²Error = zeros(Float64,length(𝒩))


# Temporal Discretization parameters
tf = 0.5
Δt = 1e-3
ntime = ceil(Int64,tf/Δt)

for (M,i) in zip(𝒩,1:length(𝒩)) 
  let
    # Define the reference grid
    global q = LinRange(0,1,M);
    global r = LinRange(0,1,M);  
    QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
    XY = 𝒮.(QR)
    detJ = (det∘J).(𝒮, QR)  

    # Get the SBP matrices
    sbp_1d = SBP(M);
    global sbp_2d = SBP_2d(sbp_1d);
    # Penalty terms for applying the boundary conditions using the SAT method
    τ₀ = τ₁ = τ₂ = τ₃ = 1.0;
    pterms = (τ₀, τ₁, τ₂, τ₃)
    
    # Compute the stiffness matrix
    𝐃 = 𝐊(q, r, sbp_2d, pterms)

    # Density (Mass) matrix evaluated at the grid points
    ρᵢ = ρ.(XY)
    𝐌 = I(2) ⊗ spdiagm(ρᵢ)

    # Compute the lu-factoriztion of the LHS for the time stepping (Crank Nicolson)
    𝐌⁻ = (𝐌 + (Δt/2)^2*𝐃)
    𝐌⁺ = (𝐌 - (Δt/2)^2*𝐃)
    lu𝐌⁺ = factorize(𝐌⁺);

    # March in time using Crank Nicolson Scheme
    let
      u₀ = eltocols(U.(XY,0))
      v₀ = eltocols(Uₜ.(XY,0))
      
      # Crank Nicolson Method
      global u₁ = zero(u₀)  
      global v₁ = zero(v₀) 
      t = 0.0
      for i=1:ntime   
        # Compute the nᵗʰ and (n+1)ᵗʰ level of the source term and the boundary terms
        Fₙ = eltocols(F.(XY, t))
        Fₙ₊₁ = eltocols(F.(XY, t+Δt))
        gₙ = 𝐠(t, q, r, pterms, sbp_2d)
        gₙ₊₁ = 𝐠(t+Δt, q, r, pterms, sbp_2d)

        # Combine the RHS
        rhs = Fₙ + Fₙ₊₁ + gₙ + gₙ₊₁

        # Get the solution at the next time step
        fargs = Δt, u₀, v₀, rhs
        u₁,v₁ = CN(lu𝐌⁺, 𝐌⁻, 𝐌, fargs) # Function in "time-stepping.jl"
        t = t+Δt
        u₀ = u₁
        v₀ = v₁
      end           
    end;

    # Compute the L²Error
    H = sbp_1d[1][1]
    𝐇 = (I(2) ⊗ H ⊗ H)*(I(2) ⊗ spdiagm(detJ))
    e = (u₁ - eltocols(U.(XY,tf)))
    L²Error[i] = sqrt(e'*𝐇*e)
    println("Done N = "*string(M)*", L²Error = "*string(L²Error[i]))
  end
end

"""
Function to extract the solution from the long vector
"""
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

Uap, Vap = UV(u₁)
XY = vec(𝒮.([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]))
Ue, Ve = UV(reduce(hcat,U.(XY,tf))')
plt1 = contourf(q, r, Uap, title="u₁ Approximate")
plt2 = contourf(q, r, Ue, title="u₁ Exact")
plt3 = contourf(q, r, Vap, title="v₁ Approximate")
plt4 = contourf(q, r, Ve, title="v₁ Exact")
plt12 = plot(plt1, plt2, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));
plt34 = plot(plt3, plt4, xlabel="x", ylabel="y", layout=(2,1), size=(700,800));

plt5 = plot(h, L²Error, xaxis=:log10, yaxis=:log10, label="L²Error", lw=2);
plot!(plt5, h, h.^4, label="O(h⁴)", lw=1);
plt6_1 = scatter(Tuple.(XY), size=(700,800), markersize=0.5, xlabel="x = x(q,r)", ylabel="y = y(q,r)", label="Physical Domain")
plt6_2 = scatter(Tuple.(vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)])), xlabel="q", ylabel="r", label="Reference Domain", markersize=0.5)
plt6 = plot(plt6_1, plt6_2, layout=(1,2))
plt56 = plot(plt6, plt5, layout=(2,1), size=(700,800))