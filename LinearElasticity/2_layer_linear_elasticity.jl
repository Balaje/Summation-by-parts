include("2d_elasticity_problem.jl")

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = [q, 0.0 + 0.0*sin(2π*q)]; # Bottom boundary. Also the interface
c₂¹(r) = [1.0, r]; # Right boundary
c₃¹(q) = [q, 1.0]; # Top boundayr
# Layer 2 (q,r) ∈ [0,1] × [-1,0]
c₀²(r) = [0.0, r-1]; # Left boundary
c₁²(q) = [q, -1.0]; # Bottom boundary. 
c₂²(r) = [1.0, r-1]; # Right boundary
c₃²(q) = c₁¹(q); # Top boundary. Also the interface
# Compute all the intersection points 
# Layer 1
P₀₁¹ = SVector{2}(P(c₀¹,c₁¹));
P₁₂¹ = SVector{2}(P(c₁¹,c₂¹));
P₂₃¹ = SVector{2}(P(c₂¹,c₃¹));
P₃₀¹ = SVector{2}(P(c₃¹,c₀¹));
# Layer 2
P₀₁² = SVector{2}(P(c₀²,c₁²));
P₁₂² = SVector{2}(P(c₁²,c₂²));
P₂₃² = SVector{2}(P(c₂²,c₃²));
P₃₀² = SVector{2}(P(c₃²,c₀²));
# Use the transfinite interpolation to obtain the physical domai
𝒮¹(qr) = (1-qr[1])*c₀¹(qr[2]) + qr[1]*c₂¹(qr[2]) + (1-qr[2])*c₁¹(qr[1]) + qr[2]*c₃¹(qr[1]) - 
    ((1-qr[2])*(1-qr[1])*P₀₁¹ + qr[2]*qr[1]*P₂₃¹ + qr[2]*(1-qr[1])*P₃₀¹ + (1-qr[2])*qr[1]*P₁₂¹);
𝒮²(qr) = (1-qr[1])*c₀²(qr[2]) + qr[1]*c₂²(qr[2]) + (1-qr[2])*c₁²(qr[1]) + qr[2]*c₃²(qr[1]) - 
    ((1-qr[2])*(1-qr[1])*P₀₁² + qr[2]*qr[1]*P₂₃² + qr[2]*(1-qr[1])*P₃₀² + (1-qr[2])*qr[1]*P₁₂²);
# Check the domain.
M = 21
q¹ = LinRange(0,1,M); r¹ = LinRange(0,1,M);  
QR¹ = vec([@SVector [q¹[j], r¹[i]] for i=1:lastindex(q¹), j=1:lastindex(r¹)]);
q² = LinRange(0,1,M); r² = LinRange(0,1,M);  
QR² = vec([@SVector [q²[j], r²[i]] for i=1:lastindex(q²), j=1:lastindex(r²)]);
plt1 = scatter(Tuple.(𝒮¹.(QR¹)))
scatter!(plt1, Tuple.(𝒮².(QR²)))

# Determinants of the transformation matrix
detJ¹ = (det∘J).(𝒮¹, QR¹);
detJ² = (det∘J).(𝒮², QR²);

# Get the SBP discretization
sbp_1d = SBP(M);
sbp_2d = SBP_2d(sbp_1d);

"""
Function to compute the stiffness matrix using the SBP-SAT method.
On the Neumann boundary, the penalty terms are equal to 1.
On the Interface boundary, the penalty terms are taken from Duru, Virta 2014.
"""
function 𝐊2(q, r, sbp_2d, pterms, H)
  QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
  # The determinants of the transformation
  detJ¹ = [1,1] ⊗ (det∘J).(𝒮¹,QR) 
  detJ² = [1,1] ⊗ (det∘J).(𝒮²,QR) 
  𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹ = sbp_2d[3] 
  #Dq, _, _, Sr = sbp_2d[1] 
  τ₀, τ₁, τ₂, τ₃ = pterms   

  # Get the transformed material properties on the first domain
  Aₜ¹(qr) = t𝒫(𝒮¹,qr)[1:2, 1:2];
  Bₜ¹(qr) = t𝒫(𝒮¹,qr)[3:4, 3:4];
  Cₜ¹(qr) = t𝒫(𝒮¹,qr)[1:2, 3:4];
  # Get the transformed material properties on the second domain
  Aₜ²(qr) = t𝒫(𝒮²,qr)[1:2, 1:2];
  Bₜ²(qr) = t𝒫(𝒮²,qr)[3:4, 3:4];
  Cₜ²(qr) = t𝒫(𝒮²,qr)[1:2, 3:4];
  
  # The second derivative SBP operators on the first domain  
  𝐃𝐪𝐪ᴬ₁ = 𝐃𝐪𝐪2d(Aₜ¹, QR, 𝒮¹)
  𝐃𝐫𝐫ᴮ₁ = 𝐃𝐫𝐫2d(Bₜ¹, QR, 𝒮¹)
  𝐃𝐪C𝐃𝐫₁, 𝐃𝐫Cᵗ𝐃𝐪₁ = 𝐃𝐪𝐫𝐃𝐫𝐪2d(Cₜ¹, QR, sbp_2d, 𝒮¹)  
  global 𝐓𝐪₁, 𝐓𝐫₁ = 𝐓𝐪𝐓𝐫2d(Aₜ¹, Bₜ¹, Cₜ¹, QR, sbp_2d)
  𝐏₁ = spdiagm(detJ¹.^-1)*(𝐃𝐪𝐪ᴬ₁ + 𝐃𝐫𝐫ᴮ₁ + 𝐃𝐪C𝐃𝐫₁ + 𝐃𝐫Cᵗ𝐃𝐪₁) # The bulk term    

  # The second derivative SBP operators on the second domain
  𝐃𝐪𝐪ᴬ₂ = 𝐃𝐪𝐪2d(Aₜ², QR, 𝒮²)
  𝐃𝐫𝐫ᴮ₂ = 𝐃𝐫𝐫2d(Bₜ², QR, 𝒮²)
  𝐃𝐪C𝐃𝐫₂, 𝐃𝐫Cᵗ𝐃𝐪₂ = 𝐃𝐪𝐫𝐃𝐫𝐪2d(Cₜ², QR, sbp_2d, 𝒮²)  
  𝐓𝐪₂, 𝐓𝐫₂ = 𝐓𝐪𝐓𝐫2d(Aₜ², Bₜ², Cₜ², QR, sbp_2d)
  𝐏₂ = spdiagm(detJ².^-1)*(𝐃𝐪𝐪ᴬ₂ + 𝐃𝐫𝐫ᴮ₂ + 𝐃𝐪C𝐃𝐫₂ + 𝐃𝐫Cᵗ𝐃𝐪₂) # The bulk term   

  # The SAT terms for the Neumann boundary
  SATₙ₁ = -(τ₀*𝐇𝐫₀⁻¹*𝐓𝐫₁)*0 + τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫₁ - τ₂*𝐇𝐪₀⁻¹*𝐓𝐪₁ + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪₁ # r=0 (c₁) is the interface
  SATₙ₂ = -τ₀*𝐇𝐫₀⁻¹*𝐓𝐫₂ + (τ₁*𝐇𝐫ₙ⁻¹*𝐓𝐫₂)*0 - τ₂*𝐇𝐪₀⁻¹*𝐓𝐪₂ + τ₃*𝐇𝐪ₙ⁻¹*𝐓𝐪₂ # r=0 (c₃) is the interface

  # The SAT terms for the interface boundary
  M = size(q,1)
  B̃, B̂ = sparse([I(2M^2) -I(2M^2); -I(2M^2) I(2M^2)]), sparse([I(2M^2) I(2M^2); -I(2M^2) -I(2M^2)])

  # Block matrices
  # Hinv = H\I(M) |> sparse
  # 𝐇inv = I(2)⊗I(M)⊗Hinv
  # 𝐇⁻¹ = blockdiag(𝐇inv, 𝐇inv)
  𝐇⁻¹ = blockdiag(𝐇𝐫₀⁻¹, 𝐇𝐫ₙ⁻¹)
  𝐓𝐫 = blockdiag(𝐓𝐫₁, 𝐓𝐫₂)
  𝐁ₕ = B̂
  𝐁ₜ = B̃
  
  # Penalty coefficients
  τₙ = 0.5
  γₙ = 0.5
  ζ₀ = 20*(M-1)
  
  SATᵢ = 𝐇⁻¹*(-τₙ*𝐁ₕ*𝐓𝐫 + γₙ*𝐓𝐫'*𝐁ₕ' + ζ₀*𝐁ₜ) 

  𝒫 = blockdiag(𝐏₁ - SATₙ₁, 𝐏₂ - SATₙ₂)  

  𝒫 - SATᵢ        
end
stima = 𝐊2(q¹, r¹, sbp_2d, (1,1,1,1), sbp_1d[1][1]);
ev = eigvals(collect(stima));