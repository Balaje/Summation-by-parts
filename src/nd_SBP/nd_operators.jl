###############################################################################
# Contains functions to implement the n-dimensional version of the SBP method #
###############################################################################
"""
Function to obtain the jump matrix corresponding to the normal vector
"""
function jump(mn₁, mn₂, 𝐧::AbstractVecOrMat{Int64}; X=[1])
  m₁, n₁ = mn₁  
  m₂, n₂ = mn₂
  n1, m1 =  N2S((m₁,n₁), 0, (n₁,m₁))[findall(𝐧 .!= [0,0])[1]-1]
  n2, m2 =  N2S((m₂,n₂), 0, (n₂,m₂))[findall(𝐧 .!= [0,0])[1]-1]
  BH = [-(X ⊗ kron(N2S(E1(m1,m1,(m1,m1)), E1(1,1,(m1,m1)), I(n1)).(𝐧)...))  (X ⊗ kron(N2S(E1(m1,1,(m1,m2)), E1(1,m2,(m1,m2)), I(n2)).(𝐧)...)); 
        -(X ⊗ kron(N2S(E1(1,m1,(m2,m1)), E1(m2,1,(m2,m1)), I(n1)).(𝐧)...))  (X ⊗ kron(N2S(E1(1,1,(m2,m2)), E1(m2,m2,(m2,m2)), I(n2)).(𝐧)...))]
  BT = [-(X ⊗ kron(N2S(E1(m1,m1,(m1,m1)), E1(1,1,(m1,m1)), I(n1)).(𝐧)...))  (X ⊗ kron(N2S(E1(m1,1,(m1,m2)), E1(1,m2,(m1,m2)), I(n2)).(𝐧)...)); 
        (X ⊗ kron(N2S(E1(1,m1,(m2,m1)), E1(m2,1,(m2,m1)), I(n1)).(𝐧)...))  -(X ⊗ kron(N2S(E1(1,1,(m2,m2)), E1(m2,m2,(m2,m2)), I(n2)).(𝐧)...))]
  BH, BT
end

"""
A Dictionary to establish the correspondence between the normal and the grid points
"""
N2S(x,y,z) = Dict([(0,z), (1,x), (-1,y)])
(d::Dict)(k) = d[k];

"""
Surface Jacobian matrix
"""
function _surface_jacobian(qr, Ω, 𝐧::AbstractVecOrMat{Int64}; X=[1])  
  m1, m2 = size(qr)
  n(x) = reshape(Float64.(𝐧), (length(𝐧),1))
  nqr = n.(qr)
  Jqr = (det∘J).(qr, Ω).*J⁻¹.(qr, Ω)
  J_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(Jqr, length(𝐧))))
  n_on_grid = spdiagm.(vec.(get_property_matrix_on_grid(nqr, length(𝐧))))  
  m2, m1 = N2S((m1,m2), 0, (m2,m1))[findall(𝐧 .!= [0,0])[1]-1]
  n2s = kron(N2S(E1(m2,m2,m2), E1(1,1,m2), sparse(I(m1))).(𝐧)...)
  Jn_on_grid = ((J_on_grid)*(n_on_grid));
  X⊗sqrt.(sum([(Ji*n2s) for Ji in Jn_on_grid].^2))
end

"""
Function to compute the jump with non-conforming interfaces
"""
function jump(m₁::Int64, m₂::Int64, 𝐧::AbstractVecOrMat{Int64}, qr₁, qr₂, Ω₁, Ω₂; X=[1])
  @assert length(𝐧)==2 "Only Inpterpolation on 2d grids implemented for now"
  if(m₁ < m₂)
    NC = m₁
    NF = m₂
    C2F, F2C = INTERPOLATION_4(NC) 
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))
    W₁ = (X ⊗ kron(N2S(E1(NC,NC,(NC,NC)), E1(1,1,(NC,NC)), sparse(I(NC))).(𝐧)...))
    Z₁ = (X ⊗ kron(N2S(E1(NC,1,(NC,NF)), E1(1,NF,(NC,NF)), J₁\(F2C*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,NC,(NF,NC)), E1(NF,1,(NF,NC)), J₂\(C2F*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(NF,NF)), E1(NF,NF,(NF,NF)), sparse(I(NF))).(𝐧)...))
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  else
    NF = m₁
    NC = m₂
    C2F, F2C = INTERPOLATION_4(NC) 
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))
    W₁ = (X ⊗ kron(N2S(E1(NF,NF,(NF,NF)), E1(1,1,(NF,NF)), sparse(I(NF))).(𝐧)...))
    Z₁ = (X ⊗ kron(N2S(E1(NF,1,(NF,NC)), E1(1,NC,(NF,NC)), J₁\(C2F*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,NF,(NC,NF)), E1(NC,1,(NC,NF)), J₂\(F2C*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(NC,NC)), E1(NC,NC,(NC,NC)), sparse(I(NC))).(𝐧)...))
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  end
  BH, BT
end

"""
Second version of jump() for non conforming interfaces
"""
function jump(mn₁, mn₂, 𝐪𝐫, 𝛀, 𝐧; X=[1])
  @assert length(𝐧)==2 "Only Inpterpolation on 2d grids implemented for now"
  m1, n1 = mn₁  
  m2, n2 = mn₂
  n₁, m₁ =  N2S((m1,n1), 0, (n1,m1))[findall(𝐧 .!= [0,0])[1]-1]
  n₂, m₂ =  N2S((m2,n2), 0, (n2,m2))[findall(𝐧 .!= [0,0])[1]-1]
  qr₁, qr₂ = 𝐪𝐫
  Ω₁, Ω₂ = 𝛀
  if(n₁ < n₂)
    NC = n₁
    NF = n₂
    C2F, F2C = INTERPOLATION_4(NC)     
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))    
    W₁ = (X ⊗ kron(N2S(E1(NC,NC,(NC,NC)), E1(1,1,(NC,NC)), sparse(I(m₁))).(𝐧)...))
    Z₁ = (X ⊗ kron(N2S(E1(m₁,1,(m₁,m₂)), E1(1,m₂,(m₁,m₂)), J₁\(F2C*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,m₁,(m₂,m₁)), E1(m₂,1,(m₂,m₁)), J₂\(C2F*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(NF,NF)), E1(NF,NF,(NF,NF)), sparse(I(m₂))).(𝐧)...))            
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  else    
    NF = n₁
    NC = n₂
    C2F, F2C = INTERPOLATION_4(NC) 
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))
    W₁ = (X ⊗ kron(N2S(E1(NF,NF,(NF,NF)), E1(1,1,(NF,NF)), sparse(I(m₁))).(𝐧)...))
    Z₁ = (X ⊗ kron(N2S(E1(m₁,m₂,(m₁,m₂)), E1(m₁,m₂,(m₁,m₂)), J₁\(C2F*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,m₁,(m₂,m₁)), E1(m₂,1,(m₂,m₁)), J₂\(F2C*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(NC,NC)), E1(NC,NC,(NC,NC)), sparse(I(m₂))).(𝐧)...))
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  end
  BH, BT
end