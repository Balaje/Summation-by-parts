include("2d_elasticity_problem.jl")

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = [q, 0.0 + 0.3*sin(2π*q)]; # Bottom boundary. Also the interface
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
M = 41
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
  Dq, _, _, Sr = sbp_2d[1] 
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
  𝐓𝐪₁, 𝐓𝐫₁ = 𝐓𝐪𝐓𝐫2d(Aₜ¹, Bₜ¹, Cₜ¹, QR, sbp_2d)
  𝐏₁ = spdiagm(detJ¹.^-1)*(𝐃𝐪𝐪ᴬ₁ + 𝐃𝐫𝐫ᴮ₁ + 𝐃𝐪C𝐃𝐫₁ + 𝐃𝐫Cᵗ𝐃𝐪₁) # The bulk term    

  # The second derivative SBP operators on the second domain
  𝐃𝐪𝐪ᴬ₂ = 𝐃𝐪𝐪2d(Aₜ², QR, 𝒮²)
  𝐃𝐫𝐫ᴮ₂ = 𝐃𝐫𝐫2d(Bₜ², QR, 𝒮²)
  𝐃𝐪C𝐃𝐫₂, 𝐃𝐫Cᵗ𝐃𝐪₂ = 𝐃𝐪𝐫𝐃𝐫𝐪2d(Cₜ², QR, sbp_2d, 𝒮²)  
  𝐓𝐪₂, 𝐓𝐫₂ = 𝐓𝐪𝐓𝐫2d(Aₜ², Bₜ², Cₜ², QR, sbp_2d)
  𝐏₂ = spdiagm(detJ².^-1)*(𝐃𝐪𝐪ᴬ₂ + 𝐃𝐫𝐫ᴮ₂ + 𝐃𝐪C𝐃𝐫₂ + 𝐃𝐫Cᵗ𝐃𝐪₂) # The bulk term   

  𝐇𝐪₀⁻¹₁ = 𝐇𝐪₀⁻¹₂ = 𝐇𝐪₀⁻¹
  𝐇𝐫₀⁻¹₁ = 𝐇𝐫₀⁻¹₂ = 𝐇𝐫₀⁻¹
  𝐇𝐪ₙ⁻¹₁ = 𝐇𝐪ₙ⁻¹₂ = 𝐇𝐪ₙ⁻¹
  𝐇𝐫ₙ⁻¹₁ = 𝐇𝐫ₙ⁻¹₂ = 𝐇𝐫ₙ⁻¹

  # The SAT terms for the Neumann boundary
  SATₙ₁ = τ₁*𝐇𝐫ₙ⁻¹₁*𝐓𝐫₁ - τ₂*𝐇𝐪₀⁻¹₁*𝐓𝐪₁ + τ₃*𝐇𝐪ₙ⁻¹₁*𝐓𝐪₁ # r=0 (c₁) is the interface
  SATₙ₂ = -τ₀*𝐇𝐫₀⁻¹₂*𝐓𝐫₂ - τ₂*𝐇𝐪₀⁻¹₂*𝐓𝐪₂ + τ₃*𝐇𝐪ₙ⁻¹₂*𝐓𝐪₂ # r=0 (c₃) is the interface


  # The SAT terms for the interface boundary
  M = size(q,1)
  B̃ = spzeros(M^2, M^2); 
  B̃[M, M] = 1; B̃[M+1, M+1] = 1; B̃[M+1, M] = -1; B̃[M, M+1] = -1; 
  B̂ = spzeros(M^2, M^2); 
  B̂[M, M] = 1; B̂[M+1, M+1] = -1; B̃[M+1, M] = 1; B̃[M, M+1] = -1; 
  H⁻¹ = H\(I(M)) |> sparse  

  # Block matrices
  𝐇⁻¹ = blockdiag((I(2) ⊗ H⁻¹ ⊗ H⁻¹), (I(2) ⊗ H⁻¹ ⊗ H⁻¹))
  𝐓𝐫 =  blockdiag(𝐓𝐫₁, 𝐓𝐫₂)
  𝐁ₕ = blockdiag((I(2) ⊗ B̂), (I(2) ⊗ B̂))
  𝐁ₜ = blockdiag((I(2) ⊗ B̃), (I(2) ⊗ B̃))
  # Get the individual contribution from the traction
  b₁₁¹(x) = Bₜ¹(x)[1,1];  b₁₂¹(x) = Bₜ¹(x)[1,2];  b₂₁¹(x) = Bₜ¹(x)[2,1];  b₂₂¹(x) = Bₜ¹(x)[2,2]  
  b₁₁²(x) = Bₜ²(x)[1,1];  b₁₂²(x) = Bₜ²(x)[1,2];  b₂₁²(x) = Bₜ²(x)[2,1];  b₂₂²(x) = Bₜ²(x)[2,2]  
  c₁₁¹(x) = Cₜ¹(x)[1,1];  c₁₂¹(x) = Cₜ¹(x)[1,2];  c₂₁¹(x) = Cₜ¹(x)[2,1];  c₂₂¹(x) = Cₜ¹(x)[2,2]  
  c₁₁²(x) = Cₜ²(x)[1,1];  c₁₂²(x) = Cₜ²(x)[1,2];  c₂₁²(x) = Cₜ²(x)[2,1];  c₂₂²(x) = Cₜ²(x)[2,2]  
  𝐁¹ = [spdiagm(b₁₁¹.(QR)) spdiagm(b₁₂¹.(QR)); spdiagm(b₂₁¹.(QR)) spdiagm(b₂₂¹.(QR))] 
  𝐁² = [spdiagm(b₁₁².(QR)) spdiagm(b₁₂².(QR)); spdiagm(b₂₁².(QR)) spdiagm(b₂₂².(QR))] 
  𝐂¹ = [spdiagm(c₁₁¹.(QR)) spdiagm(c₁₂¹.(QR)); spdiagm(c₂₁¹.(QR)) spdiagm(c₂₂¹.(QR))] 
  𝐂² = [spdiagm(c₁₁².(QR)) spdiagm(c₁₂².(QR)); spdiagm(c₂₁².(QR)) spdiagm(c₂₂².(QR))]   
  𝐃𝐪 = I(2) ⊗ Dq
  𝐒𝐫 = I(2) ⊗ Sr
  𝐃cq = blockdiag(𝐂¹'*𝐃𝐪, 𝐂²'*𝐃𝐪)
  𝐒Br = blockdiag(𝐁¹*𝐒𝐫, 𝐁²*𝐒𝐫)
  # Penalty coefficients
  τₙ = 0.5
  γₙ = -0.5
  τ₀ = 1/(M-1);

  SATᵢ = -τₙ*(𝐇⁻¹*𝐁ₕ*𝐓𝐫) - γₙ*(𝐇⁻¹*𝐒Br'*𝐁ₕ) - γₙ*(𝐇⁻¹*𝐃cq'*𝐁ₕ) - τ₀*(𝐇⁻¹*𝐁ₜ)

  𝒫 = blockdiag(𝐏₁, 𝐏₂)
  SATₙ = blockdiag(SATₙ₁, SATₙ₂)

  𝒫 + SATᵢ - SATₙ
end
