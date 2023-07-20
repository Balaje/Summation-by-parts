"""
SBP in two-dimensions obtained using Kronecker Product
"""
struct SBP_1_2_CONSTANT_0_1_0_1 <: SBP_TYPE
    D1::Tuple{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}}
    D2::Tuple{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}}
    
    norm::SparseMatrixCSC{Float64, Int64}
    b_norm::Tuple{SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}}
end

function SBP_1_2_CONSTANT_0_1_0_1(sbp_q::SBP_1_2_CONSTANT_0_1, sbp_r::SBP_1_2_CONSTANT_0_1)
    # Extract all the matrices from the 1d version
    Dq = sbp_q.D1
    Dr = sbp_r.D1
    Dqq = sbp_q.D2[1]
    Drr = sbp_r.D2[1]
    Sq = sbp_q.S
    Sr = sbp_r.S
    mq = sbp_q.M
    mr = sbp_r.M
    E₀q = spzeros(mq, mq); E₀q[1,1] = 1.0
    Eₙq = spzeros(mq, mq); Eₙq[mq,mq] = 1.0
    E₀r = spzeros(mr, mr); E₀r[1,1] = 1.0
    Eₙr = spzeros(mr, mr); Eₙr[mr,mr] = 1.0
    Iq = I(mq)
    Ir = I(mr)
    # Create lazy versions of the 2d operator from 1d operators
    𝐃𝐪 = ApplyArray(kron, Dq, Ir)
    𝐃𝐫 = ApplyArray(kron, Iq, Dr)
    𝐒𝐪 = ApplyArray(kron, Sq, Ir)
    𝐒𝐫 = ApplyArray(kron, Iq, Sr)

    𝐃𝐪, 𝐃𝐫, 𝐒𝐪, 𝐒𝐫
end


