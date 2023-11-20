###############################################################################
# Contains functions to implement the n-dimensional version of the SBP method #
###############################################################################
"""
Function to obtain the jump matrix corresponding to the normal vector
"""
function jump(m::Int64, 𝐧::AbstractVecOrMat{Int64}; X=[1])
  BH = [-(X ⊗ kron(N2S(E1(m,m,m), E1(1,1,m), I(m)).(𝐧)...))  (X ⊗ kron(N2S(E1(m,1,m), E1(1,m,m), I(m)).(𝐧)...)); 
        -(X ⊗ kron(N2S(E1(1,m,m), E1(m,1,m), I(m)).(𝐧)...))  (X ⊗ kron(N2S(E1(1,1,m), E1(m,m,m), I(m)).(𝐧)...))]
  BT = [-(X ⊗ kron(N2S(E1(m,m,m), E1(1,1,m), I(m)).(𝐧)...))  (X ⊗ kron(N2S(E1(m,1,m), E1(1,m,m), I(m)).(𝐧)...)); 
        (X ⊗ kron(N2S(E1(1,m,m), E1(m,1,m), I(m)).(𝐧)...))  -(X ⊗ kron(N2S(E1(1,1,m), E1(m,m,m), I(m)).(𝐧)...))]
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