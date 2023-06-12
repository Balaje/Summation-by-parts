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
  DqqA = spzeros(Float64, 2N^2, 2N^2)
  xy = reshape(XY, (N,N))
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  for i=1:N    
    DqqA  += [E(i) ⊗ SBP_VARIABLE_4(N, a₁₁.(xy[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₁₂.(xy[:,i]))[2]; 
              E(i) ⊗ SBP_VARIABLE_4(N, a₂₁.(xy[:,i]))[2]  E(i) ⊗ SBP_VARIABLE_4(N, a₂₂.(xy[:,i]))[2]]    
  end
  DqqA
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
  @inline function E(i) 
    res = spzeros(N,N)
    res[i,i] = 1.0
    res
  end
  for i=1:N    
    DqqA  += [SBP_VARIABLE_4(N, a₁₁.(xy[:,i]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₁₂.(xy[:,i]))[2] ⊗ E(i); 
              SBP_VARIABLE_4(N, a₂₁.(xy[:,i]))[2] ⊗ E(i)  SBP_VARIABLE_4(N, a₂₂.(xy[:,i]))[2] ⊗ E(i)]    
  end
  DqqA
end


@testset "Checking the SBP approximation of the variable stress tensor against the constant case" begin
  M = 40
  q = LinRange(0,1,M); r = LinRange(0,1,M);  
  XY = vec([@SVector [q[j], r[i]] for i=1:lastindex(q), j=1:lastindex(r)]);
  Ac = [3.0 0.0; 0.7 1.0]  
  Dqq = Drr = SBP(M)[3][1];
  𝐃𝐪𝐪 = (Dqq ⊗ I(M))
  𝐃𝐫𝐫 = (I(M) ⊗ Drr)  
  @test (Ac ⊗ 𝐃𝐫𝐫) ≈  SBP_Drr_2d_variable(x->Ac, XY)
  @test (Ac ⊗ 𝐃𝐪𝐪) ≈  SBP_Dqq_2d_variable(x->Ac, XY)
end;