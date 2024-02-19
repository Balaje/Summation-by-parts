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
Second version of jump() for non conforming interfaces
"""
function jump(mn₁, mn₂, 𝐪𝐫, 𝛀, 𝐧; X=[1])
  @assert length(𝐧)==2 "Only Inpterpolation on 2d grids implemented for now"
  m₁, n₁ = mn₁  
  m₂, n₂ = mn₂
  qr₁, qr₂ = 𝐪𝐫
  Ω₁, Ω₂ = 𝛀
  if(m₁ < m₂)
    NC = m₁
    NF = m₂    
    @assert NF == 2*NC - 1
    C2F, F2C = INTERPOLATION_4(NC)     
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))      
    W₁ = (X ⊗ kron(N2S(E1(n₁,n₁,(n₁,n₁)), E1(1,1,(n₁,n₁)), sparse(I(NC))).(𝐧)...))
    Z₁ = (X ⊗ kron(N2S(E1(n₁,1,(n₁,n₂)), E1(1,n₂,(n₁,n₂)), J₁\(F2C*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,n₁,(n₂,n₁)), E1(n₂,1,(n₂,n₁)), J₂\(C2F*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(n₂,n₂)), E1(n₂,n₂,(n₂,n₂)), sparse(I(NF))).(𝐧)...))            
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  else    
    NF = m₁
    NC = m₂
    @assert NF == 2*NC - 1
    C2F, F2C = INTERPOLATION_4(NC) 
    J₁ = spdiagm(((_surface_jacobian(qr₁, Ω₁, 𝐧; X=[1]) |> diag).nzval).^(0.5))
    J₂ = spdiagm(((_surface_jacobian(qr₂, Ω₂, -𝐧; X=[1]) |> diag).nzval).^(0.5))      
    W₁ = (X ⊗ kron(N2S(E1(n₁,n₁,(n₁,n₁)), E1(1,1,(n₁,n₁)), sparse(I(NF))).(𝐧)...))    
    Z₁ = (X ⊗ kron(N2S(E1(n₁,1,(n₁,n₂)), E1(1,n₂,(n₁,n₂)), J₁\(C2F*J₂) ).(𝐧)...))  
    Z₂ = (X ⊗ kron(N2S(E1(1,n₁,(n₂,n₁)), E1(n₂,1,(n₂,n₁)), J₂\(F2C*J₁) ).(𝐧)...))
    W₂ = (X ⊗ kron(N2S(E1(1,1,(n₂,n₂)), E1(n₂,n₂,(n₂,n₂)), sparse(I(NC))).(𝐧)...))    
    BH = [-W₁   Z₁;   -Z₂   W₂]
    BT = [-W₁   Z₁;   Z₂   -W₂]
  end
  BH, BT
end