include("SBP.jl")

"""
Function to get the 2d stencil from the 1d version
"""
function SBP_2d(SBP_1d)
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

  (𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐇𝐪₀⁻¹, 𝐇𝐫₀⁻¹, 𝐇𝐪ₙ⁻¹, 𝐇𝐫ₙ⁻¹)
end

###
# Functions to get the 2d stencil (variable) from the 1d version
###
"""
Get the SBP Drr operator in 2d for variable coefficients
  𝐃𝐫𝐫2d(A, QR)
  Here:
    (Input) (q,r)->A(q,r) is the 2x2 material property matrix
    (Input) QR is the coordinates in the reference grid (M^2 × 1)

    (Output)
     qr = reshape(QR,(M,M))     
     E(i) := 
         E[i,i]=1.0
     i = 1,...,M
     RESULT = Σᵢ [E(i) ⊗ SBP_VARIABLE_1d(a₁₁(qr[:,i])),  E(i) ⊗ SBP_VARIABLE_1d(a₁₂(qr[:,i])); 
                  E(i) ⊗SBP_VARIABLE_1d(a₂₁(qr[:,i])),   E(i) ⊗ SBP_VARIABLE_1d(a₂₂(qr[:,i]))]     
"""
function 𝐃𝐫𝐫2d(A, QR)
  # Extract the entries in the 2×2 tensor
  a₁₁(qr) = (det∘J)(𝒮,qr)*A(qr)[1,1]
  a₁₂(qr) = (det∘J)(𝒮,qr)*A(qr)[1,2]
  a₂₁(qr) = (det∘J)(𝒮,qr)*A(qr)[2,1]
  a₂₂(qr) = (det∘J)(𝒮,qr)*A(qr)[2,2]
  # Compute the matrix
  N = Int(√(length(QR)))
  DrrA = spzeros(Float64, 2N^2, 2N^2)
  qr = reshape(QR, (N,N))
  # E[i,i] = 1 
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  # Compute the full variable tensor SBP operator  
  for i=1:N    
    DrrA  += [E(i) ⊗ SBP_VARIABLE_4(N, a₁₁.(qr[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₁₂.(qr[:,i]))[2]; 
              E(i) ⊗ SBP_VARIABLE_4(N, a₂₁.(qr[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₂₂.(qr[:,i]))[2]]    
  end
  DrrA
end

"""
Get the SBP Dqq operator in 2d for variable coefficients
  𝐃𝐪𝐪2d(A, QR)
  Here:
    (Input) (q,r)->A(q,r) is the 2x2 material property matrix
    (Input) QR is the coordinates in the reference grid (M^2 × 1)

    (Output)
     qr = reshape(QR,(M,M)) 
     E(i) := 
          E[i,i]=1.0
     i = 1,...,M
     RESULT = Σᵢ [SBP_VARIABLE_1d(a₁₁(qr[i,:])) ⊗ E(i),  SBP_VARIABLE_1d(a₁₂(qr[i,:])) ⊗ E(i); 
                  SBP_VARIABLE_1d(a₂₁(qr[i,:])) ⊗ E(i),  SBP_VARIABLE_1d(a₂₂(qr[i,:])) ⊗ E(i)]    
"""
function 𝐃𝐪𝐪2d(A, QR)
  # Extract the entries in the 2×2 tensor
  a₁₁(qr) = (det∘J)(𝒮,qr)*A(qr)[1,1]
  a₁₂(qr) = (det∘J)(𝒮,qr)*A(qr)[1,2]
  a₂₁(qr) = (det∘J)(𝒮,qr)*A(qr)[2,1]
  a₂₂(qr) = (det∘J)(𝒮,qr)*A(qr)[2,2]
  # Compute the matrix
  N = Int(√(length(QR)))
  DqqA = spzeros(Float64, 2N^2, 2N^2)
  qr = reshape(QR, (N,N))
  # E[i,i] = 1 
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  # Compute the full variable tensor SBP operator  
  for i=1:N    
    DqqA  += [SBP_VARIABLE_4(N, a₁₁.(qr[i,:]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₁₂.(qr[i,:]))[2] ⊗ E(i); 
              SBP_VARIABLE_4(N, a₂₁.(qr[i,:]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₂₂.(qr[i,:]))[2] ⊗ E(i)]    
  end
  DqqA
end

"""
𝐃𝐪𝐫𝐃𝐫𝐪2d(A, QR)
  Here:
    (Input) (q,r)->A(q,r) is the 2x2 material property matrix
    (Input) qr is the coordinates in the reference grid (M^2 × 1)
    (Input) sbp_2d is the two-dimensional stencil

    (Output)
      𝐃𝐪 = I(2) ⊗ sbp_2d[1][1];     𝐃𝐫 = I(2) ⊗ sbp_2d[1][2] 
      𝐂 = [spdiagm(a₁₁.(qr)) spdiagm(a₁₂.(qr)); spdiagm(a₂₁.(qr)) spdiagm(a₂₂.(qr))]     
      𝐃𝐪*𝐂*𝐃𝐫, 𝐃𝐫*𝐂'*𝐃𝐪    
"""
function 𝐃𝐪𝐫𝐃𝐫𝐪2d(A, qr, sbp_2d)  
  a₁₁(qr) = (det∘J)(𝒮,qr)*A(qr)[1,1]
  a₁₂(qr) = (det∘J)(𝒮,qr)*A(qr)[1,2]
  a₂₁(qr) = (det∘J)(𝒮,qr)*A(qr)[2,1]
  a₂₂(qr) = (det∘J)(𝒮,qr)*A(qr)[2,2]  
  𝐃𝐪 = I(2) ⊗ sbp_2d[1][1]
  𝐃𝐫 = I(2) ⊗ sbp_2d[1][2] 
  𝐂 = [spdiagm(a₁₁.(qr)) spdiagm(a₁₂.(qr)); spdiagm(a₂₁.(qr)) spdiagm(a₂₂.(qr))] 
  𝐃𝐪*𝐂*𝐃𝐫, 𝐃𝐫*𝐂'*𝐃𝐪
end

"""
Get the SBP variable Tq, Tr operator
  𝐓𝐪𝐓𝐫2d(A, B, C, qr, sbp_2d)  
  Here:
    (Input) (q,r)->A(q,r), (q,r)->B(q,r), (q,r)->C(q,r) is the 2x2 material property matrix
    (Input) qr is the coordinates in the reference grid (M^2 × 1)
    (Input) sbp_2d is the two-dimensional stencil

    (Output)
      𝐀*𝐒𝐪 + 𝐂*𝐃𝐫, 𝐂'*𝐃𝐪 + 𝐁*𝐒𝐫
"""
function 𝐓𝐪𝐓𝐫2d(A, B, C, qr, sbp_2d)
  # E[i,i] = 1 
  N = Int(√(length(qr)))
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
  𝐀 = [spdiagm(a₁₁.(qr)) spdiagm(a₁₂.(qr)); spdiagm(a₂₁.(qr)) spdiagm(a₂₂.(qr))] 
  𝐁 = [spdiagm(b₁₁.(qr)) spdiagm(b₁₂.(qr)); spdiagm(b₂₁.(qr)) spdiagm(b₂₂.(qr))] 
  𝐂 = [spdiagm(c₁₁.(qr)) spdiagm(c₁₂.(qr)); spdiagm(c₂₁.(qr)) spdiagm(c₂₂.(qr))] 

  # Compute the stress tensor
  𝐀*𝐒𝐪 + 𝐂*𝐃𝐫, 𝐂'*𝐃𝐪 + 𝐁*𝐒𝐫
end