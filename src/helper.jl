using LinearAlgebra

function cbas_(n::Int, nq::Int = 2)
    bas_ = Matrix{Float64}(I, n^nq, n^nq)
    return bas_
end

function qbas_(nstate::Int, idx::Int)
    bas_ = zeros(Complex{Float64}, nstate, 1)
    bas_[idx + 1] = 1.0
    return bas_
end

function anih(n::Int)
    a_ = zeros(Float64, n, n)
    for i in 1:n-1
        a_[i, i+1] = sqrt(i)
    end
    return a_
end

function create(n::Int)
    c_ = zeros(Float64, n, n)
    for i in 2:n
        c_[i, i-1] = sqrt(i - 1)
    end
    return c_
end

function initial_state(list1::Vector{Int}, nstate::Int = 2)
    tmp_ = qbas_(nstate, list1[1])
    for idx in list1[2:end]
        wrk_ = qbas_(nstate, idx)
        tmp_ = kron(tmp_, wrk_)
    end
    return tmp_
end
function pcoef(t::Float64, amp::Vector{Float64}, tseq::Vector{Float64}, freq::Float64, tfinal::Float64; conj::Bool=false, scale::Float64=1.0)
    sign_ = conj ? 1.0 : -1.0
    plist = Vector{Bool}(undef, length(tseq) + 1)

    for i in 1:length(tseq)
        if i == 1
            plist[i] = (0.0 < t <= tseq[i])
        else
            plist[i] = (tseq[i-1] < t <= tseq[i])
        end
    end

    plist[end] = (tseq[end] < t <= tfinal)
    
    coeff = 0.0 + 0.0im
    for i in 1:length(amp)
        if plist[i]
            coeff = amp[i] * exp(sign_ * scale * 1im * freq * t)
            break
        end
    end
    
    return coeff
end