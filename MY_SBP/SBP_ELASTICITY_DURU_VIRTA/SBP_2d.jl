include("SBP.jl")

"""
Function to get the 2d stencil from the 1d version
"""
function SBP_2d(XY, SBP_1d)
  # Collect all the necessary finite difference matrices from the method
  # NOTE: Here D2s, H are not needed. 
  #       The D2s matrix is not needed since we use the variable SBP operator
  #       H because Hinv is precomputed
  HHinv, D1, D2s, S, Ids = SBP_1d;
  _, Hinv = HHinv;
  E₀, Eₙ, _, _, Id = Ids; # Needed for non-zero boundary conditions

  # Finite difference operators along the (q,r) direction
  Dq = D1; Dr = D1
  Dqq = D2s[1]; Drr = D2s[1];
  Sq = S; Sr = S;  
  Hqinv = Hinv; Hrinv = Hinv;

  # Discrete Operators in 2D
  𝐃𝐪 = Dq ⊗ Id;
  𝐃𝐫 = Id ⊗ Dr;
  𝐒𝐪 = Sq ⊗ Id;
  𝐒𝐫 = Id ⊗ Sr;  
  
  𝐇𝐪₀⁻¹ = (I(2) ⊗ (Hqinv*E₀) ⊗ Id); # q (x) = 0
  𝐇𝐫₀⁻¹ = (I(2) ⊗ Id ⊗ (Hrinv*E₀)); # r (y) = 0
  𝐇𝐪ₙ⁻¹ = (I(2) ⊗ (Hqinv*Eₙ) ⊗ Id); # q (x) = 1 
  𝐇𝐫ₙ⁻¹ = (I(2) ⊗ Id ⊗ (Hrinv*Eₙ)); # r (y) = 1 

  𝐃𝐪𝐪 = (Dqq ⊗ Id)
  𝐃𝐫𝐫 = (Id ⊗ Drr) 

  𝐈q₀ = E₀ ⊗ Id
  𝐈qₙ = Eₙ ⊗ Id
  𝐈r₀ = Id ⊗ E₀
  𝐈rₙ = Id ⊗ Eₙ

  𝐈q₀a = findnz(𝐈q₀)[1]; 
  𝐈qₙa = findnz(𝐈qₙ)[1];   
  𝐈r₀a = findnz(𝐈r₀)[1];   
  𝐈rₙa = findnz(𝐈rₙ)[1];   

  XYq₀ = XY[𝐈q₀a]
  XYqₙ = XY[𝐈qₙa]
  XYr₀ = XY[𝐈r₀a]
  XYrₙ = XY[𝐈rₙa]

  (𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹), (𝐈q₀a, 𝐈r₀a, 𝐈qₙa, 𝐈rₙa), (XYq₀, XYr₀, XYqₙ, XYrₙ)
end

###
# Functions to get the 2d stencil (variable) from the 1d version
###
"""
Get the SBP Dqq operator in 2d for variable coefficients
"""
function SBP_Drr_2d_variable(A, XY)
  # Extract the entries in the 2×2 tensor
  a₁₁(x) = A(x)[1,1]
  a₁₂(x) = A(x)[1,2]
  a₂₁(x) = A(x)[2,1]
  a₂₂(x) = A(x)[2,2]
  # Compute the matrix
  N = Int(√(length(XY)))
  DrrA = spzeros(Float64, 2N^2, 2N^2)
  xy = reshape(XY, (N,N))
  # E[i,i] = 1 
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  # Compute the full variable tensor SBP operator
  for i=1:N    
    DrrA  += [E(i) ⊗ SBP_VARIABLE_4(N, a₁₁.(xy[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₁₂.(xy[:,i]))[2]; 
              E(i) ⊗ SBP_VARIABLE_4(N, a₂₁.(xy[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₂₂.(xy[:,i]))[2]]    
  end
  DrrA
end

"""
Get the SBP Drr operator in 2d for variable coefficients
"""
function SBP_Dqq_2d_variable(A, XY)
  # Extract the entries in the 2×2 tensor
  a₁₁(x) = A(x)[1,1]
  a₁₂(x) = A(x)[1,2]
  a₂₁(x) = A(x)[2,1]
  a₂₂(x) = A(x)[2,2]
  # Compute the matrix
  N = Int(√(length(XY)))
  DqqA = spzeros(Float64, 2N^2, 2N^2)
  xy = reshape(XY, (N,N))
  # E[i,i] = 1 
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  # Compute the full variable tensor SBP operator
  for i=1:N    
    DqqA  += [SBP_VARIABLE_4(N, a₁₁.(xy[i,:]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₁₂.(xy[i,:]))[2] ⊗ E(i); 
              SBP_VARIABLE_4(N, a₂₁.(xy[i,:]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₂₂.(xy[i,:]))[2] ⊗ E(i)]    
  end
  DqqA
end

"""
Get the SBP Dqr, Drq operator in 2d for variable coefficients
"""
function SBP_Dqr_2d_variable(A, xy, sbp_2d)  
  a₁₁(x) = A(x)[1,1]
  a₁₂(x) = A(x)[1,2]
  a₂₁(x) = A(x)[2,1]
  a₂₂(x) = A(x)[2,2]  
  𝐃𝐪 = I(2) ⊗ sbp_2d[1][1]
  𝐃𝐫 = I(2) ⊗ sbp_2d[1][2]  
  𝐂 = [spdiagm(a₁₁.(xy)) spdiagm(a₁₂.(xy)); spdiagm(a₂₁.(xy)) spdiagm(a₂₂.(xy))] 
  𝐃𝐪*𝐂*𝐃𝐫, 𝐃𝐫*𝐂'*𝐃𝐪
end

"""
Get the SBP variable Tq operator
"""
function SBP_Tqr_2d_variable(A, B, C, xy, sbp_2d)
  # E[i,i] = 1 
  N = Int(√(length(xy)))
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  Dq, Dr, Sq, Sr = sbp_2d[1]
  𝐃𝐪 = I(2) ⊗ Dq
  𝐃𝐫 = I(2) ⊗ Dr
  𝐒𝐪 = I(2) ⊗ Sq
  𝐒𝐫 = I(2) ⊗ Sr

  # Tensor components as a function 
  a₁₁(x) = A(x)[1,1];  a₁₂(x) = A(x)[1,2];  a₂₁(x) = A(x)[2,1];  a₂₂(x) = A(x)[2,2]  
  b₁₁(x) = B(x)[1,1];  b₁₂(x) = B(x)[1,2];  b₂₁(x) = B(x)[2,1];  b₂₂(x) = B(x)[2,2]  
  c₁₁(x) = C(x)[1,1];  c₁₂(x) = C(x)[1,2];  c₂₁(x) = C(x)[2,1];  c₂₂(x) = C(x)[2,2]  

  # Get the coefficient matrices
  𝐀 = [spdiagm(a₁₁.(xy)) spdiagm(a₁₂.(xy)); spdiagm(a₂₁.(xy)) spdiagm(a₂₂.(xy))] 
  𝐁 = [spdiagm(b₁₁.(xy)) spdiagm(b₁₂.(xy)); spdiagm(b₂₁.(xy)) spdiagm(b₂₂.(xy))] 
  𝐂 = [spdiagm(c₁₁.(xy)) spdiagm(c₁₂.(xy)); spdiagm(c₂₁.(xy)) spdiagm(c₂₂.(xy))] 

  # Compute the stress tensor
  𝐀*𝐒𝐪 + 𝐂*𝐃𝐫, 𝐂'*𝐃𝐪 + 𝐁*𝐒𝐫
end

@testset "Checking the SBP approximation of the variable stress tensor against the constant case" begin
  # Get a sample discretization
  M = 40
  q = LinRange(0,1,M); r = LinRange(0,1,M);  
  XY = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);

  # Define constant material properties
  Ac = [c₁₁ 0; 0 c₃₃]
  Bc = [c₃₃ 0; 0 c₂₂]
  Cc = [0 c₁₂; c₃₃ 0]
  Cᵀc = [0 c₃₃; c₁₂ 0] 

  # Get the SBP stencil
  sbp_1d = SBP(M)
  sbp_2d = SBP_2d(XY, sbp_1d)

  # Get the constant coefficient SBP operators
  𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫 = sbp_2d[1]  
  𝐃𝐪𝐪, 𝐃𝐫𝐫 = sbp_2d[2]

  # Get the constant coefficient version of the elliptic operator
  𝐃𝐪𝐪ᴬ = Ac ⊗ 𝐃𝐪𝐪
  𝐃𝐫𝐫ᴮ = Bc ⊗ 𝐃𝐫𝐫
  𝐃𝐪C𝐃𝐫 = (I(2) ⊗ 𝐃𝐪)*(Cc ⊗ 𝐃𝐫)
  𝐃𝐫Cᵗ𝐃𝐪 = (I(2) ⊗ 𝐃𝐫)*(Cᵀc ⊗ 𝐃𝐪)
  𝐓𝐪 = (Ac ⊗ 𝐒𝐪 + Cc ⊗ 𝐃𝐫)
  𝐓𝐫 = (Cᵀc ⊗ 𝐃𝐪 + Bc ⊗ 𝐒𝐫)

  # Now 4 tests to check if the variable coefficient code reduces to the constant coefficient version
  @test 𝐃𝐪𝐪ᴬ ≈ SBP_Dqq_2d_variable(x->Ac, XY)
  @test 𝐃𝐫𝐫ᴮ ≈ SBP_Drr_2d_variable(x->Bc, XY)
  @test 𝐃𝐪C𝐃𝐫 ≈ SBP_Dqr_2d_variable(x->Cc, XY, sbp_2d)[1]
  @test 𝐃𝐫Cᵗ𝐃𝐪 ≈ SBP_Dqr_2d_variable(x->Cc, XY, sbp_2d)[2]
  @test 𝐓𝐪 ≈ SBP_Tqr_2d_variable(x->Ac, x->Bc, x->Cc, XY, sbp_2d)[1]
  @test 𝐓𝐫 ≈ SBP_Tqr_2d_variable(x->Ac, x->Bc, x->Cc, XY, sbp_2d)[2]
end;