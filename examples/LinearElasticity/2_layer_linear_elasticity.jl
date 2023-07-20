include("2d_elasticity_problem.jl")

"""
Define the geometry of the two layers. 
"""
# Layer 1 (q,r) ∈ [0,1] × [0,1]
c₀¹(r) = [0.0, r]; # Left boundary
c₁¹(q) = [q, 0.0]; # Bottom boundary. Also the interface
c₂¹(r) = [1.0, r]; # Right boundary
c₃¹(q) = [q, 1.0 + 0.0*sin(2π*q)]; # Top boundary
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
q = LinRange(0,1,M); r = LinRange(0,1,M);  
QR = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
plt1 = scatter(Tuple.(𝒮¹.(QR)))
scatter!(plt1, Tuple.(𝒮².(QR)))

 # Get the transformed material properties on the first domain
 Aₜ¹(qr) = t𝒫(𝒮¹,qr)[1:2, 1:2];
 Bₜ¹(qr) = t𝒫(𝒮¹,qr)[3:4, 3:4];
 Cₜ¹(qr) = t𝒫(𝒮¹,qr)[1:2, 3:4];
 # Get the transformed material properties on the second domain
 Aₜ²(qr) = t𝒫(𝒮²,qr)[1:2, 1:2];
 Bₜ²(qr) = t𝒫(𝒮²,qr)[3:4, 3:4];
 Cₜ²(qr) = t𝒫(𝒮²,qr)[1:2, 3:4];

# Get the SBP discretization
sbp_1d = SBP(M);
sbp_2d = SBP_2d(sbp_1d);

function E1(i,M)
  res = spzeros(M,M)
  res[i,i] = 1.0
  res
end

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

  # The SAT terms for the Neumann boundary
  M = size(q,1)
  
  Jsrₙ¹ = I(2) ⊗ I(M) ⊗ (spdiagm([J⁻¹s(𝒮¹, @SVector[qᵢ, 1.0], @SVector[0.0, 1.0]) for qᵢ in q].^-1))  
  Jsq₀¹ = I(2) ⊗ (spdiagm([J⁻¹s(𝒮¹, @SVector[0.0, rᵢ], @SVector[-1.0, 0.0]) for rᵢ in r].^-1)) ⊗ I(M)
  Jsqₙ¹ = I(2) ⊗ (spdiagm([J⁻¹s(𝒮¹, @SVector[1.0, rᵢ], @SVector[1.0, 0.0]) for rᵢ in r].^-1)) ⊗ I(M)

  Jsr₀² = I(2) ⊗ I(M) ⊗ (spdiagm([J⁻¹s(𝒮², @SVector[qᵢ, 0.0], @SVector[0.0, -1.0]) for qᵢ in q].^-1))  
  Jsq₀² = I(2) ⊗ (spdiagm([J⁻¹s(𝒮², @SVector[0.0, rᵢ], @SVector[-1.0, 0.0]) for rᵢ in r].^-1)) ⊗ I(M)
  Jsqₙ² = I(2) ⊗ (spdiagm([J⁻¹s(𝒮², @SVector[1.0, rᵢ], @SVector[1.0, 0.0]) for rᵢ in r].^-1)) ⊗ I(M)

  SATₙ₁ = τ₁*𝐇𝐫ₙ⁻¹*Jsrₙ¹*𝐓𝐫₁ - τ₂*𝐇𝐪₀⁻¹*Jsq₀¹*𝐓𝐪₁ + τ₃*𝐇𝐪ₙ⁻¹*Jsqₙ¹*𝐓𝐪₁ #=-τ₀*𝐇𝐫₀⁻¹*Jsr₀¹*𝐓𝐫₁ +=#  # r=0 (c₁) is the interface
  SATₙ₂ = -τ₀*𝐇𝐫₀⁻¹*Jsr₀²*𝐓𝐫₂ - τ₂*𝐇𝐪₀⁻¹*Jsq₀²*𝐓𝐪₂ + τ₃*𝐇𝐪ₙ⁻¹*Jsqₙ²*𝐓𝐪₂ #=τ₁*𝐇𝐫ₙ⁻¹*Jsrₙ²*𝐓𝐫₂ =#  # r=0 (c₃) is the interface

  # The SAT terms for the interface boundary  
  B̃, B̂ = sparse([I(2M^2) -I(2M^2); -I(2M^2) I(2M^2)]), sparse([I(2M^2) I(2M^2); -I(2M^2) -I(2M^2)])

  # Surface Jacobian on the interfaces
  Jsr₀¹ = I(2) ⊗ I(M) ⊗ (spdiagm([J⁻¹s(𝒮¹, @SVector[qᵢ, 0.0], @SVector[0.0, -1.0]) for qᵢ in q].^-1))
  Jsrₙ² = I(2) ⊗ I(M) ⊗ (spdiagm([J⁻¹s(𝒮², @SVector[qᵢ, 1.0], @SVector[0.0, 1.0]) for qᵢ in q].^-1)) 

  # Block matrices  
  𝐇⁻¹ = blockdiag(𝐇𝐫₀⁻¹, 𝐇𝐫ₙ⁻¹)      
  𝐓𝐫 = blockdiag(𝐓𝐫₁, 𝐓𝐫₂)
  𝐉 = blockdiag(Jsr₀¹, Jsrₙ²)
  𝐁ₕ = B̂
  𝐁ₜ = B̃
  
  # Penalty coefficients
  τₙ = 0.5
  γₙ = 0.5
  ζ₀ = 10*(M-1)
  
  SATᵢ = 𝐇⁻¹*(-τₙ*𝐁ₕ*𝐉*𝐓𝐫 + γₙ*𝐓𝐫'*𝐉'*𝐁ₕ' + ζ₀*𝐁ₜ)

  𝒫 = blockdiag(𝐏₁ - SATₙ₁, 𝐏₂ - SATₙ₂)  

  𝒫 - SATᵢ        
end
stima = 𝐊2(q, r, sbp_2d, (1,1,1,1), sbp_1d[1][1]);
ev = eigvals(collect(stima));