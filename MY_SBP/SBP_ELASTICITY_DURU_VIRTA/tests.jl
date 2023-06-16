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

  # Check for constant coefficient  
  D2 = SBP(40)[3][1];
  D2_var = SBP_VARIABLE_4(40, ones(40))[2];
  @test D2 ≈ D2_var

  # Check with MATLAB result for variable coefficient
  DD2fx_MATLAB = [ -0.000309393974552, 0.000269728080379, 0.001047548591239, 0.003591787192802, 0.008631298572127, 0.016858005023686,
  0.029130632680928, 0.046258365784993, 0.069050388577016, 0.098315885298133, 0.134864040189485, 0.179504037492204,
  0.233045061447426, 0.296296296296299, 0.370066926279945, 0.455166135639508, 0.552403108616133, 0.662587029450950,
  0.786527082385057, 0.925032451659644, 1.078912321515873, 1.248975876194768, 1.436032299937642, 1.640890776985447,
  1.864360491579454, 2.107250627960592, 2.370370370370495, 2.654528903049414, 2.960535410239547, 3.289199076181006,
  3.641329085116155, 4.017734621284681, 4.419224868929099, 4.846609012289136, 5.300696235607647, 5.782295723124477,
  6.291085452539143, 6.830687998707361, 7.405384446804646, 7.987005453068377]
  x = LinRange(0,1,40);
  fx = x.^2;
  Ax = x.^3;
  DD2 = SBP_VARIABLE_4(40, Ax)[2];
  DD2fx = DD2*fx
  @test DD2fx ≈ DD2fx_MATLAB
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