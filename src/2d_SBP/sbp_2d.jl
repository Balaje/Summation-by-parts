"""
SBP in two-dimensions obtained using Kronecker Product
"""
struct SBP_1_2_CONSTANT_0_1_0_1 <: SBP_TYPE
    D1::Tuple{AbstractMatrix{Float64}, AbstractMatrix{Float64}}
    D2::Tuple{AbstractMatrix{Float64}, AbstractMatrix{Float64}}
    S::Tuple{AbstractMatrix{Float64}, AbstractMatrix{Float64}}
    norm::Tuple{AbstractMatrix{Float64}, AbstractMatrix{Float64}}
    E::Tuple{AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}, AbstractMatrix{Float64}}
end

"""
Lazy Kronecker Product
"""
⊗(A,B) = ApplyArray(kron, A, B)


"""
Construct the 2d sbp operator using the 1d versions
"""
function SBP_1_2_CONSTANT_0_1_0_1(sbp_q::SBP_1_2_CONSTANT_0_1, sbp_r::SBP_1_2_CONSTANT_0_1)
    # Extract all the matrices from the 1d version
    Hq = sbp_q.norm
    Hr = sbp_r.norm
    Dq = sbp_q.D1
    Dr = sbp_r.D1
    Dqq = sbp_q.D2[1]
    Drr = sbp_r.D2[1]
    Sq = sbp_q.S
    Sr = sbp_r.S
    mq = sbp_q.M
    mr = sbp_r.M
    Iq = sbp_q.E[1]
    E₀q = sbp_q.E[2]
    Eₙq = sbp_q.E[3]
    Ir = sbp_r.E[1]
    E₀r = sbp_r.E[2]
    Eₙr =  sbp_r.E[3]
    # Create lazy versions of the 2d operator from 1d operators
    𝐃𝐪 = Dq ⊗ Ir
    𝐃𝐫 = Iq ⊗ Dr
    𝐒𝐪 = Sq ⊗ Ir
    𝐒𝐫 = Iq ⊗ Sr
    𝐃𝐪𝐪 = Dqq ⊗ Ir
    𝐃𝐫𝐫 = Drr ⊗ Iq
    𝐇𝐪 = Hq ⊗ Ir
    𝐇𝐫 = Iq ⊗ Hr
    𝐄₀q = E₀q ⊗ Ir
    𝐄ₙq = Eₙq ⊗ Ir
    𝐄₀r = Iq ⊗ E₀r 
    𝐄ₙr = Iq ⊗ Eₙr 
    𝐄 = Iq ⊗ Ir

    SBP_1_2_CONSTANT_0_1_0_1( (𝐃𝐪,𝐃𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐒𝐪,𝐒𝐫), (𝐇𝐪, 𝐇𝐫), (𝐄, 𝐄₀q, 𝐄₀r, 𝐄ₙq, 𝐄ₙr) )
end

struct SBP_2_VARIABLE_0_1_0_1 <: SBP_TYPE
    
end
"""
Variable 2d SBP operator
"""

function E1(i,M)
    res = spzeros(Float64, M, M)
    res[i,i] = 1.0
    res
end

function generate_2d_grid(mn::Tuple{Int64,Int64})
    m,n = mn
    q = LinRange(0,1,m); r = LinRange(0,1,n)
    qr = [@SVector [q[i],r[j]] for i=1:n, j=1:m];
    qr
end

function Dqq(a::Function, qr)    
    a_qr = a.(qr)
    m,n = size(qr)
    Dqq = [SBP_2_VARIABLE_0_1(m, a_qr[i,:]).D2 for i=1:m]
    Eq = [E1(i,m) for i=1:m]
    Dqq, Eq
end

function Drr(a::Function, qr)    
    a_qr = a.(qr)
    m,n = size(qr)
    Drr = [SBP_2_VARIABLE_0_1(m, a_qr[:,i]).D2 for i=1:n]
    Er = [E1(i,m) for i=1:m]
    Drr, Er
end

function Dqr(a::Function, qr, sbp::SBP_1_2_CONSTANT_0_1_0_1)
    a_qr = a.(qr)
    A = spdiagm(vec(a_qr))
    Dq, Dr = sbp.D1
    Dq'*A*Dr, Dr'*A*Dq
end
