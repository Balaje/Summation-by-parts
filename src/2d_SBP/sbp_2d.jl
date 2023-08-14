"""
SBP in two-dimensions obtained using Kronecker Product
"""
struct SBP_1_2_CONSTANT_0_1_0_1 <: SBP_TYPE
    D1::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
    D2::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
    S::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
    norm::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
    E::Tuple{SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}, SparseMatrixCSC{Float64,Int64}}
end

"""
Lazy Kronecker Product
"""
⊗(A,B) = kron(A, B)


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
    𝐃𝐫𝐫 = Iq ⊗ Drr
    𝐄₀q = E₀q ⊗ Ir
    𝐄ₙq = Eₙq ⊗ Ir
    𝐄₀r = Iq ⊗ E₀r 
    𝐄ₙr = Iq ⊗ Eₙr
    𝐇𝐪₀ = ((Hq\Iq)*E₀q) ⊗ Ir
    𝐇𝐫₀ = Iq ⊗ ((Hr\Ir)*E₀r)
    𝐇𝐪ₙ = ((Hq\Iq)*Eₙq) ⊗ Ir
    𝐇𝐫ₙ = Iq ⊗ ((Hr\Ir)*Eₙr)
    𝐄 = Iq ⊗ Ir

    SBP_1_2_CONSTANT_0_1_0_1( (𝐃𝐪,𝐃𝐫), (𝐃𝐪𝐪, 𝐃𝐫𝐫), (𝐒𝐪,𝐒𝐫), (𝐇𝐪₀,𝐇𝐪ₙ,𝐇𝐫₀,𝐇𝐫ₙ), (𝐄, 𝐄₀q, 𝐄₀r, 𝐄ₙq, 𝐄ₙr) )
end

function E1(i,M)
    res = spzeros(Float64, M, M)
    res[i,i] = 1.0
    res
end

function generate_2d_grid(mn::Tuple{Int64,Int64})
    m,n = mn
    q = LinRange(0,1,m); r = LinRange(0,1,n)
    qr = [@SVector [q[j],r[i]] for i=1:n, j=1:m];
    qr
end

struct Dqq <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Dqq(a_qr::AbstractMatrix{Float64})    
    m,n = size(a_qr)
    D2q = [SBP_2_VARIABLE_0_1(m, a_qr[i,:]).D2 for i=1:n]
    Er = [E1(i,m) for i=1:n]
    sum(D2q .⊗ Er)
end

struct Drr <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Drr(a_qr::AbstractMatrix{Float64})
    m,n = size(a_qr)
    D2r = [SBP_2_VARIABLE_0_1(n, a_qr[:,i]).D2 for i=1:m]
    Eq = [E1(i,n) for i=1:m]
    sum(Eq .⊗ D2r)
end

struct Dqr <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Dqr(a_qr::AbstractMatrix{Float64})
    m,n = size(a_qr)
    A = spdiagm(vec(a_qr))
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)    
    D1q, D1r = sbp_2d.D1
    D1q*A*D1r
end

struct Drq <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Drq(a_qr::AbstractMatrix{Float64})
    m,n = size(a_qr)
    A = spdiagm(vec(a_qr))
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)    
    D1q, D1r = sbp_2d.D1
    D1r*A*D1q
end

struct Tq <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Tq(a_qr::AbstractMatrix{Float64}, c_qr::AbstractMatrix{Float64})
    m,n = size(a_qr)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)
    _, Dr = sbp_2d.D1
    Sq, _ = sbp_2d.S
    A = spdiagm(vec(a_qr))
    C = spdiagm(vec(c_qr))
    A*Sq + C*Dr
end

struct Tr <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
end
function Tr(c_qr::AbstractMatrix{Float64}, b_qr::AbstractMatrix{Float64})
    m, n = size(c_qr)
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)    
    Dq, _ = sbp_2d.D1
    _, Sr = sbp_2d.S
    C = spdiagm(vec(c_qr))
    B = spdiagm(vec(b_qr))
    C*Dq + B*Sr
end


"""
Linear Elasticity SBP operator
"""
struct Dᴱ <: SBP_TYPE
    A::Matrix{SparseMatrixCSC{Float64, Int64}}
end
function Dᴱ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
    Ptuple = Tuple.(Pqr)
    P_page = reinterpret(reshape, Float64, Ptuple)
    dim = length(size(P_page))
    P_vec = reshape(splitdimsview(P_page, dim-2), (4,4))
    Dᴱ₂ = [Dqq Dqq Dqr Dqr; Dqq Dqq Dqr Dqr; Drq Drq Drr Drr; Drq Drq Drr Drr]
    res = [Dᴱ₂[i,j](P_vec[i,j]) for i=1:4, j=1:4]
    Dᴱ(res)
end

function Pᴱ(D1::Dᴱ)
    D = D1.A
    [D[1,1] D[1,2]; D[2,1] D[2,2]] + [D[3,3] D[3,4]; D[4,3] D[4,4]] +
        [D[1,3] D[1,4]; D[2,3] D[2,4]] + [D[3,1] D[3,2]; D[4,1] D[4,2]]
end

"""
Linear Elasticity traction operator
"""
struct Tᴱ <: SBP_TYPE
    A::SparseMatrixCSC{Float64, Int64}
    B::SparseMatrixCSC{Float64, Int64}
end

function Tᴱ(Pqr::Matrix{SMatrix{4,4,Float64,16}})
    Ptuple = Tuple.(Pqr)
    P_page = reinterpret(reshape, Float64, Ptuple)
    dim = length(size(P_page))
    P_vec = spdiagm.(vec.(reshape(splitdimsview(P_page, dim-2), (4,4))))

    m,n = size(P_page)[2:3]
    sbp_q = SBP_1_2_CONSTANT_0_1(m)
    sbp_r = SBP_1_2_CONSTANT_0_1(n)
    sbp_2d = SBP_1_2_CONSTANT_0_1_0_1(sbp_q, sbp_r)

    Dq, Dr = sbp_2d.D1
    Sq, Sr = sbp_2d.S

    Tq = [P_vec[1,1] P_vec[1,2]; P_vec[2,1] P_vec[2,2]]*(I(2)⊗Sq) + [P_vec[1,3] P_vec[1,4]; P_vec[2,3] P_vec[2,4]]*(I(2)⊗Dr)
    Tr = [P_vec[3,1] P_vec[3,2]; P_vec[4,1] P_vec[4,2]]*(I(2)⊗Dq) + [P_vec[3,3] P_vec[3,4]; P_vec[4,3] P_vec[4,4]]*(I(2)⊗Sr)

    Tᴱ(Tq, Tr)
end
