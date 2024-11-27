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

"""
    on(op::Matrix{<:Number}, q::Int, n::Int)

Expand the single-qubit matrix operator `op` to act on qubit `q` of `n` qubits.

In other words, apply Kronecker-products such that identity `I` acts on the other qubits.

"""
function on(op::Matrix{<:Number}, q::Integer, n::Integer)
    A = ones(1,1)                   # A 1x1 IDENTITY MATRIX
    I = one(op)                     # AN IDENTITY MATRIX MATCHING THE DIMENSIONS OF `op`
    for i ∈ 1:n
        A = kron(A, i == q ? op : I)    # `op` ACTS ON QUBIT `q`, `I` ACTS ON ALL OTHERS
    end
    return A
end

"""
    vector_anhi(
    n::Int,
    m::Int=2;
    basis::Union{Matrix{<:Number},Nothing}=nothing,
)

Construct a vector of annihilation operators acting on each of `n` `m`-level systems.

Optionally, rotate these operators into the provided basis.

These matrices, in conjunction with their adjoints,
    form a complete algebra for the space of `n` `m`-level systems.

"""
function vector_anhi(
    n::Int,
    m::Int=2;
    basis::Union{Matrix{<:Number},Nothing}=nothing,
)
    a_ = anih(m)                        # SINGLE-QUBIT ANNIHILATION OPERATOR
    a = [on(a_, q, n) for q in 1:n]         # EACH OPERATOR, ACTING ON FULL HILBERT SPACE
    if !(basis === nothing)
        for q ∈ 1:n
            a[q] .= basis' * a[q] * basis   # CONJUGATE WITH BASIS
        end
    end
    return a
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

"""
    expectation(A::Matrix{ComplexF64}, ψ::Vector{ComplexF64})

Evaluate the expectation value ⟨ψ|A|ψ⟩.

`A` and `ψ` should have compatible dimensions.

"""
function expectation(
    A::AbstractMatrix{<:Number},
    ψ::Vector{T};

    # INFERRED VALUES (relatively fast, but pass them in to minimize allocations)
    N = length(ψ),                          # SIZE OF STATEVECTOR

    # PRE-ALLOCATIONS (for those that want every last drop of efficiency...)
    tmpV = Vector{T}(undef, N),    # FOR MATRIX-VECTOR MULTIPLICATION
) where T <: Number
    mul!(tmpV, A, ψ)
    return ψ' * tmpV
end

"""
    projector(n::Integer, m::Integer, m0::Integer)

Project a Hilbert space of `n` `m0`-level qubits onto that of `n` `m`-level qubits

Returns an (`n^m`, `n^m0`) shaped matrix `Π`.
To perform the projection on a vector, use ψ ← Πψ.
To perform the projection on a matrix, use A ← ΠAΠ'.

"""
function projector(n::Integer, m::Integer, m0::Integer)

    if m < m0; return projector(n, m0, m)'; end     # NOTE THE ADJOINT ' OPERATOR

    z = Vector{Int}(undef, n)       # PRE-ALLOCATION TO STORE BASE-m0 DECOMPOSITIONS
    N  = m^n; N0 = m0^n             # FULL HILBERT SPACE DIMENSIONS
    Id= Matrix{Bool}(I, N, N)       # IDENTITY MATRIX IN LARGER HILBERT SPACE
    Π = Matrix{Bool}(undef, N, N0)  # PROJECTOR MATRIX
    j = 1                           # ITERATES COLUMNS OF PROJECTOR

    for i ∈ 1:N                     # FILL PROJECTOR WITH SELECT COLUMNS FROM IDENTITY
        digits!(z, i-1, base=m)         # DECOMPOSE INDEX INTO LARGER BASE
        if any(z .>= m0); continue; end # SKIP INDEX IF ABSENT IN SMALLER SPACE
        Π[:,j] .= Id[:,i]               # COPY COLUMN OF IDENTITY MATRIX
        j += 1                          # ADVANCE COLUMN INDEX
    end

    return Π
end