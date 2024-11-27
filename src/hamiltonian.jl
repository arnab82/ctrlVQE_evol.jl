using LinearAlgebra
using SparseArrays
# using IterTools

# Struct for QuantumComputerHamiltonian
using LinearAlgebra

# Assume the existence of these helper functions
# static_ham, t_ham, cbas_, dresser, msdressed, mtdressed, initial_state

mutable struct QuantumComputerHamiltonian
    n::Int
    m::Int
    omegas::Vector{Float64}
    deltas::Vector{Float64}
    g::Matrix{Float64}
    basis::Matrix{Float64}
    a::Vector{Matrix{Float64}}
    Hdrive::Vector{Vector{SparseMatrixCSC{ComplexF64,Int64}}}
    dsham::SparseMatrixCSC{Float64,Int64}
end
mutable struct Device
    omegas::Vector{Float64}
    g::Matrix{Float64}
    deltas::Vector{Float64}

    function Device(n::Int, m::Int)
        pi2 = 2 * π
        
        # Generate omegas for each qubit
        omegas = pi2 .* rand(Float64, n) .+ pi2 * 4.8  # Example: around 4.8 GHz with some randomness
        
        # Generate coupling strengths g between each pair of qubits
        g = zeros(Float64, n, n)
        for i in 1:n
            for j in i+1:n
                g[i, j] = pi2 * (0.018 + 0.002 * rand())
                g[j, i] = g[i, j]  # Symmetric coupling
            end
        end
        
        # Generate detunings for each qubit
        deltas = pi2 .* rand(Float64, n) .+ pi2 * 0.3  # Example: around 0.3 GHz with some randomness
        
        new(omegas, g, deltas)
    end
end

function Base.length(device::Device)
    return length(device.omegas)
end

function build_quantum_computer_hamiltonian(device::Device, m::Int=2)
    n = length(device.omegas)
    
    # Static Hamiltonian
    h_static = static_hamiltonian(device, m)

    # Dressing the Hamiltonian
    basis = cbas_(m, n)
    ham_d_bas = dresser(h_static, basis)
    ds_ham = msdressed(ham_d_bas, h_static)
    
    # Time-dependent Hamiltonian
    h_drive = t_ham(m, n)
    h_drive_dressed = mtdressed(ham_d_bas, h_drive)
    
    a =vector_anhi(n, m)
    

    ds_ham_dense = Matrix(ds_ham)
    # println("states")
    # println(typeof(states))
    # println("in_state")
    # println(typeof(in_state))
    # println("istate")
    # println(typeof(istate))
    # println("annhilation operator")
    # println(typeof(a))
    # println("drive hamiltonian")
    # println(typeof(h_drive_dressed))
    # println("static hamiltonian")
    # println(typeof(ds_ham_dense))
    # println("basis")
    # println(typeof(basis))
    return QuantumComputerHamiltonian(
        n, m, device.omegas, device.deltas, device.g,
        basis, a, h_drive_dressed, sparse(ds_ham_dense),
    )
end

struct QubitCouple
    q1::Int
    q2::Int
    QubitCouple(q1, q2) = q1 > q2 ? new(q2, q1) : new(q1, q2)
end

# Struct for DriveHamiltonian
mutable struct DriveHamiltonian
    Ω::Vector{Float64} # drive amplitudes
    ν::Vector{Float64} # drive frequencies
end
mutable struct DriveParams
    amp::Vector{Float64}
    tseq::Vector{Vector{Float64}}
    freq::Vector{Float64}
    duration::Float64
end
# Struct for the device parameters
mutable struct Device_transmon
    omegas::Vector{Float64}
    g::Matrix{Float64}
    deltas::Vector{Float64}

    function Device_transmon()
        pi2 = 2 * π
        new(
            [pi2 * 4.808049015463495, pi2 * 4.833254817254613, pi2 * 4.940051121317842, pi2 * 4.795960998582043],
            [0.0  pi2*0.018312874435769682  pi2*0.019312874435769682  pi2*0.020312874435769682;
             pi2*0.018312874435769682  0.0  pi2*0.021312874435769682  pi2*0.019312874435769682;
             pi2*0.019312874435769682  pi2*0.021312874435769682  0.0  pi2*0.018312874435769682;
             pi2*0.020312874435769682  pi2*0.019312874435769682  pi2*0.018312874435769682  0.0],
            [pi2 * 0.3101773613134229, pi2 * 0.2916170385725456, pi2 * 0.3301773613134229, pi2 * 0.2616170385725456]
        )
    end
end



"""
    static_hamiltonian(device::Transmon, m::Int=2)

Constructs the transmon Hamiltonian
    ``\\sum_q ω_q a_q^† a_q
    - \\sum_q \\frac{δ_q}{2} a_q^† a_q^† a_q a_q
    + \\sum_{⟨pq⟩} g_{pq} (a_p^† a_q + a_q^\\dagger a_p)``.

"""
function static_hamiltonian(device, m::Int64=2)
    n = length(device)
    N = m ^ n

    a_ = vector_anhi(n, m)

    H = zeros(Float64, N,N)

    for q ∈ 1:n
        H += device.omegas[q]   * (a_[q]'   * a_[q])     # RESONANCE  TERMS
        H -= device.deltas[q]/2 * (a_[q]'^2 * a_[q]^2)   # ANHARMONIC TERMS
    end
    g=device.g
    G = Dict{QubitCouple,Float64}()
    for p in 2:n; for q in p+1:n
        G[QubitCouple(p,q)] = g[p,q]
    end; end

    # COUPLING TERMS
    for (pair, g) ∈ G
        term = g * a_[pair.q1]' * a_[pair.q2]
        H += term + term'
    end

    return Hermitian(H)
end


function t_ham(nstate::Int, nqubit::Int=2)
    astate = anih(nstate)
    cstate = create(nstate)
    eye_n = I(nstate)

    hdrive = []
    for i in 1:nqubit
        if i == 1
            tmp1 = cstate
            tmp2 = astate
        else
            tmp1 = eye_n
            tmp2 = eye_n
        end

        for j in 2:nqubit
            if j == i
                wrk1 = cstate
                wrk2 = astate
            else
                wrk1 = eye_n
                wrk2 = eye_n
            end

            tmp1 = kron(tmp1, wrk1)
            tmp2 = kron(tmp2, wrk2)
        end

        push!(hdrive, [tmp1, tmp2])
    end

    return hdrive
end

"""
Build the drive Hamiltonian for the quantum computer
    H_drive = ∑_i Ω_i (exp(iν_i t) a_i + exp(-iν_i t) a_i^†)
"""

# Build the drive Hamiltonian for the quantum computer
function build_drive_hamiltonian(qch::QuantumComputerHamiltonian, drive::DriveParams, t::Float64)
    n = qch.n
    hdrive = zeros(Complex{Float64}, size(qch.dsham))

    for i in 1:n
        # Calculate time-dependent coefficients
        hcoef = pcoef(t, drive.amp[i], drive.tseq[i], drive.freq[i], drive.duration)
        hcoefc = pcoef(t, drive.amp[i], drive.tseq[i], drive.freq[i], drive.duration, conj=true)

        # Add contributions from drive terms to the Hamiltonian
        hdrive += hcoef * qch.hdrive[i][1]
        hdrive += hcoefc * qch.hdrive[i][2]
    end

    # Diagonal component from dressed static Hamiltonian
    dsham_diag = -1im * diagm(diag(qch.dsham))
    dsham_diag = dsham_diag * t
    matexp_ = exp(dsham_diag)
    matexp_ = Diagonal(matexp_)

    # Compute the final time-dependent Hamiltonian
    hamr_ = matexp_' * hdrive * matexp_

    return hamr_
end

# Dressing functions
function dresser(H, basis::Matrix{Float64})
    evals, evecs = eigen(H)

    # Ensure evecs are treated as columns
    if size(evecs, 1) != size(H, 1)
        evecs = transpose(evecs)
    end

    res = []
    for i in eachcol(basis)
        # Find the eigenvector with the maximum overlap with vector i in basis
        idx = argmax(abs.(dot.(eachcol(evecs), Ref(i))))
        tmp = evecs[:, idx]  # Extract the entire column, ensuring tmp is a vector
        push!(res, tmp)
    end

    # Normalize the vectors and adjust phase
    for (i, part) in enumerate(res)
        # println("Checking res[", i, "] with size: ", size(res[i]))

        mask = abs.(res[i]) .< 1e-15
        # println("Mask size: ", size(mask))
        
        if any(mask)  # Proceed only if mask has true values
            # println("Mask has true values.")
            res[i][mask] .= 0.0
        else
            # println("Mask has no true values.")
        end
    end

    return hcat(res...)  # Convert list of vectors back to a matrix
end

function msdressed(dbasis::Matrix{Float64}, h::Matrix{Float64})
    h_ = dbasis * h * dbasis'
    h_sparse = sparse(h_)
    mask = abs.(h_sparse) .< 1.0e-15
    h_sparse[mask] .= 0.0
    return h_sparse
end


function msdressed(dbasis::Matrix{Float64}, h)#h::Matrix{Complex{Float64}})
    h_ = dbasis * h * dbasis'
    h_sparse = sparse(h_)
    mask = abs.(h_sparse) .< 1.0e-15
    h_sparse[mask] .= 0.0
    return h_sparse
end
function mtdressed(dbasis::Matrix{Float64}, h::Vector{Any})
    h_transformed = []
    for i in h
        push!(h_transformed, [msdressed(dbasis, i[1]), msdressed(dbasis, i[2])])
    end
    return h_transformed
end
function mtdressed(dbasis::Matrix{Complex{Float64}}, h::Vector{Vector{Matrix{Complex{Float64}}}})
    h_transformed = []
    for i in h
        push!(h_transformed, [msdressed(dbasis, i[1]), msdressed(dbasis, i[2])])
    end
    return h_transformed
end

