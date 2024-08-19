using LinearAlgebra
using SparseArrays
using IterTools
# using QuantumOptics
include("./helper.jl")
# Struct for QuantumComputerHamiltonian
using LinearAlgebra

# Assume the existence of these helper functions
# static_ham, t_ham, cbas_, dresser, msdressed, mtdressed, initial_state

mutable struct QuantumComputerHamiltonian
    n::Int # number of qubits
    m::Int # number of modes per qubit
    omegas::Vector{Float64} # qubit resonance frequencies
    deltas::Vector{Float64} # qubit anharmonicities
    g::Matrix{Float64} # qubit-qubit coupling strengths
    basis::Vector{Matrix{Float64}} # basis for each mode
    a::Vector{Matrix{Float64}} # annihilation operator for each mode
    Hdrive::Vector{Any} # non-dressed time-dependent Hamiltonian
    dsham::Matrix{Float64} # dressed static Hamiltonian
    states::Vector{Int} # indices of basis states
    initial_state::Vector{Float64} # initial quantum state vector
    istate::Vector{Int} # initial state configuration
end

function QuantumComputerHamiltonian(n::Int = 2, m::Int = 3, omegas::Vector{Float64} = Float64[], 
                                        deltas::Vector{Float64} = Float64[], g::Matrix{Float64} = zeros(Float64, 0, 0), 
                                        istate::Vector{Int} = Int[], Hstatic::Matrix{Float64} = zeros(Float64, 0, 0))
        
        if isempty(omegas) || isempty(deltas) || isempty(g)
            error("Provide system parameters: omegas, deltas, and g")
        end

        # Static Hamiltonian
        if isempty(Hstatic)
            Hstatic = build_static_hamiltonian(n, m, omegas, deltas, g)
        end

        # Dressing the Hamiltonian
        basis_ = cbas_(m, n)
        Hamdbas = dresser(Hstatic, basis_)
        dsham = msdressed(Hamdbas, Hstatic)
        
        # Time-dependent Hamiltonian
        Hdrive = t_ham(m, n)
        hdrive = mtdressed(Hamdbas, Hdrive)
        
        # State indices
        states = []
        # Create the cartesian product of [0, 1] repeated `n` times
        for i in IterTools.product(ntuple(_ -> [0, 1], n)...)
            sum_ = 0
            cout_ = 0
            for j in reverse(i)
                sum_ += j * m^cout_
                cout_ += 1
            end
            push!(states, sum_)
        end

        # Initial state
        if isempty(istate)
            if n == 2
                istate = [0, 1]
            elseif n == 4
                istate = [0, 0, 1, 1]
            elseif n == 6
                istate = [0, 0, 1, 0, 0, 1]
            else
                error("Provide initial state using initial_state function")
            end
        end
        in_state = initial_state(istate, m)
        dsham_dense = Matrix(dsham)
        println("number of mode is ",m)
        println("omegas:")
        display(omegas)
        println("deltas:")
        display(deltas)
        println("g:")
        display(g)
        println("basis:")
        display(basis_)
        println("dsham:")
        display(dsham)
        println("hdrive:")
        display(hdrive)
        println("states:")
        display(states)
        println("in_state:")
        display(in_state)
        println("istate:")
        display(istate)
        
        println("dsham_dense:")
        display(dsham_dense)
        return n, m, omegas, deltas, g, basis_, dsham_dense,hdrive,states, in_state, istate
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
# mutable struct Device
#     omegas::Vector{Float64}
#     g::Matrix{Float64}
#     deltas::Vector{Float64}

#     function Device()
#         pi2 = 2 * π
#         new(
#             [pi2 * 4.808049015463495, pi2 * 4.833254817254613, pi2 * 4.940051121317842, pi2 * 4.795960998582043],
#             [0.0  pi2*0.018312874435769682  pi2*0.019312874435769682  pi2*0.020312874435769682;
#              pi2*0.018312874435769682  0.0  pi2*0.021312874435769682  pi2*0.019312874435769682;
#              pi2*0.019312874435769682  pi2*0.021312874435769682  0.0  pi2*0.018312874435769682;
#              pi2*0.020312874435769682  pi2*0.019312874435769682  pi2*0.018312874435769682  0.0],
#             [pi2 * 0.3101773613134229, pi2 * 0.2916170385725456, pi2 * 0.3301773613134229, pi2 * 0.2616170385725456]
#         )
#     end
# end
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



"""
Build the static Hamiltonian for the quantum computer
    H=∑_i ω_i a_i^† a_i - ∑_i δ_i/2 a_i^† a_i^† a_i a_i + ∑_ij g_ij a_i^† a_j+ a_j^† a_i
"""


function build_static_hamiltonian(n,m,omegas,deltas,g,calculate_interactions = true)

    diag_n = Diagonal(1:m)
    eye_n = I(m)
    diag_eye = 0.5 * (diag_n * (diag_n - eye_n))
    astate = anih(m)  # Annihilation operator
    cstate = create(m)  # Creation operator

    ham_ = 0.0

    for i in 1:n
        # On-site Hamiltonian terms for qubit i
        # omega_i * n_i - (delta_i / 2) * n_i * (n_i - 1)
        h_ = omegas[i] * diag_n - deltas[i] * diag_eye
        # Initialize the temporary matrices
        if i == 1
            tmp_ = h_  # Start with on-site Hamiltonian term for the first qubit
            tmp_i = astate  # Initialize for interaction terms
        else
            tmp_ = eye_n  # Identity matrix for no effect on this qubit
            if i == n
                tmp_i = cstate  # Use creation operator for the last qubit
            else
                tmp_i = eye_n  # Identity matrix for intermediate qubits
            end
        end

        for j in i:n
            if j == i
                wrk = h_  # On-site Hamiltonian term for qubit i
                wrk_i = astate  # Annihilation operator for interaction terms
            elseif j == i + 1
                wrk = eye_n  # Identity for adjacent qubit in coupling term
                wrk_i = cstate  # Creation operator for adjacent qubit
            else
                wrk = eye_n  # Identity matrix for non-coupled qubits
                wrk_i = eye_n  # Identity matrix for non-coupled qubits in interaction term
            end

            # Build the full Hamiltonian term for the system
            if i==1 && j==i
                tmp_=wrk
            else
                tmp_ = kron(tmp_, wrk)  
            end
            if  calculate_interactions && j != i
                tmp_i = kron(tmp_i, wrk_i)
            end
        end

        # Add the constructed on-site terms to the Hamiltonian
        if i == 1
            ham_ = tmp_
        else
            ham_ += tmp_

        end

        # Add interaction terms to the Hamiltonian
        if calculate_interactions
            tmp_i = tmp_i + adjoint(tmp_i)  # Hermitian conjugate for symmetry
            tmp_i *= g[i, i]  # Coupling strength for qubit i with itself or neighbors
            ham_ += tmp_i

            if n == 2
                calculate_interactions = false  # Stop further interaction calculations for n=2
            end
        end
    end
    println("size of ham_ is",sizeof(ham_))
    return ham_
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
function dresser(H::Matrix{Float64}, basis::Matrix{Float64})
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


function msdressed(dbasis::Matrix{Complex{Float64}}, h::Matrix{Complex{Float64}})
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


n=2
m=3

device = Device(n,m)
Hstatic = build_static_hamiltonian(n, m, device.omegas, device.deltas, device.g)
display(Hstatic)
basis_ = cbas_(m, n)
# Dressing the Hamiltonian
basis_ = cbas_(m, n)
Hamdbas = dresser(Hstatic, basis_)
# Time-dependent Hamiltonian
Hdrive = t_ham(m, n)
hdrive = mtdressed(Hamdbas, Hdrive)
qch = QuantumComputerHamiltonian(n, m, device.omegas, device.deltas, device.g)