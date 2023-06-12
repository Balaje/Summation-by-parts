######################################################################
# Contains some tests to check the correctness of the implementation
######################################################################

using Test

@testset "Check the 1d SBP operators (first and second derivatives)" begin
  N = 10
  x = LinRange(0,1,N)
  fx = map(x->x^2, x)
  D1 = SBP(N)[2]
  D2 = SBP(N)[3][1]
  @test D1*fx ≈ map(x->2x, x)
  @test D2*fx ≈ map(x->2, x)
end;

@testset "Check the variable coefficient second derivative (MATLAB->Julia)" begin
  D2 = SBP(40)[3][1];
  D2_var = SBP_VARIABLE_4(40, ones(40))[2];
  @test D2 ≈ D2_var
end;

@testset "Some tests to verify the Gradient, Stress and Divergence." begin 
  v(x) = [sin(π*x[1])*sin(π*x[2]), sin(2π*x[1])*sin(2π*x[2])];
  ∇v(x) = vec([π*cos(π*x[1])*sin(π*x[2]) π*sin(π*x[1])*cos(π*x[2]); 
         2π*cos(2π*x[1])*sin(2π*x[2]) 2π*sin(2π*x[1])*cos(2π*x[2])]);
  σv(x) = vec(hcat(A(x)*([π*cos(π*x[1])*sin(π*x[2]), 2π*cos(2π*x[1])*sin(2π*x[2])]) + C(x)*([π*sin(π*x[1])*cos(π*x[2]), 2π*sin(2π*x[1])*cos(2π*x[2])]),
         Cᵀ(x)*([π*cos(π*x[1])*sin(π*x[2]), 2π*cos(2π*x[1])*sin(2π*x[2])]) + B(x)*([π*sin(π*x[1])*cos(π*x[2]), 2π*sin(2π*x[1])*cos(2π*x[2])])));
  div_σ_v(x) = A(x)*([-π^2*sin(π*x[1])*sin(π*x[2]), -4π^2*sin(2π*x[1])*sin(2π*x[2])]) + C(x)*([π^2*cos(π*x[1])*cos(π*x[2]), 4π^2*cos(2π*x[1])*cos(2π*x[2])]) + 
             Cᵀ(x)*([π^2*cos(π*x[1])*cos(π*x[2]), 4π^2*cos(2π*x[1])*cos(2π*x[2])]) + B(x)*([-π^2*sin(π*x[1])*sin(π*x[2]), -4π^2*sin(2π*x[1])*sin(2π*x[2])]);
  
  pt = @SVector rand(2)  
  σ∇(x) = σ(∇(v,x),x)
  # Test the divergence, stress and the divergence of the stress tensor
  @test ∇v(pt) ≈ ∇(v, pt);  
  @test σv(pt) ≈ σ(∇(v, pt), pt);
  @test div_σ_v(pt) ≈ div(σ∇, pt);
end;

@testset "Check the SBP approximation of the variable stress tensor against the constant case" begin
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