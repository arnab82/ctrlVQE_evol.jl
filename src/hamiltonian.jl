using LinearAlgebra
using SparseArrays
using QuantumOptics

# Define the struct for QuantumComputerHamiltonian
mutable struct QuantumComputerHamiltonian
    n::Int # number of qubits
    m::Int # number of modes per qubit
    omegas::Vector{Float64} # qubit resonance frequencies
    deltas::Vector{Float64} # qubit anharmonicities
    g::Matrix{Float64} # qubit-qubit coupling strengths
    basis::Vector{FockBasis} # basis for each mode
    a::Vector{Operator} # annihilation operator for each mode
end

# Define the struct for DriveHamiltonian
mutable struct DriveHamiltonian
    Ω::Vector{Float64} # drive amplitudes
    ν::Vector{Float64} # drive frequencies
end

# Define the device parameters
mutable struct Device
    omegas::Vector{Float64}
    g::Matrix{Float64}
    deltas::Vector{Float64}

    function Device()
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

# Create the QuantumComputerHamiltonian object

function create_hamiltonian(n::Int, m::Int, device::Device)
    omegas = device.omegas
    deltas = device.deltas
    g = device.g

    # Define the basis for each mode
    basis = [FockBasis(m) for _ in 1:n]

    # Define the annihilation operators for each mode
    a = [destroy(basis[i]) for i in 1:n]

    return QuantumComputerHamiltonian(n, m, omegas, deltas, g, basis, a)
end

# Define the annihilation operator
function anih(nstate::Int)
    a = spdiagm(1 => sqrt.(1:nstate-1))
    return a
end

# Define the creation operator
function create_op(nstate::Int)
    return adjoint(anih(nstate))
end
"""
Build the static Hamiltonian for the quantum computer
    H=∑_i ω_i a_i^† a_i - ∑_i δ_i/2 a_i^† a_i^† a_i a_i + ∑_ij g_ij a_i^† a_j+ a_j^† a_i
"""


function build_static_hamiltonian(qch::QuantumComputerHamiltonian)
    n, m, omegas, deltas, g, basis, a = qch.n, qch.m, qch.omegas, qch.deltas, qch.g, qch.basis, qch.a

    diag_n = Diagonal(0:m-1)
    eye_n = I(m)
    diag_eye = 0.5 * (diag_n * (diag_n - eye_n))
    astate = anih(m)
    cstate = create_op(m)

    
    ham_ = 0
    iwork = true

    for i in 1:n
        h_ = omegas[i] * diag_n - deltas[i] * diag_eye

        if i == 1
            tmp_ = h_
            tmp_i = astate
        else
            tmp_ = eye_n
            if i == n
                tmp_i = cstate
            else
                tmp_i = eye_n
            end
        end

        for j in 1:n
            if j == i
                wrk = h_
                wrk_i = astate
            elseif j == i+1
                wrk = eye_n
                wrk_i = cstate
            else
                wrk = eye_n
                wrk_i = eye_n
            end
            
            # Perform the tensor product
            tmp_ = kron(tmp_, wrk)
            if iwork
                tmp_i = kron(tmp_i, wrk_i)
            end
        end

        # Add the constructed terms to ham_
        if i == 1
            ham_ = tmp_
        else
            ham_ += tmp_
        end
        if iwork
            
            tmp_i = tmp_i + adjoint(tmp_i)
            tmp_i *= g[i,i]
            ham_ += tmp_i

            if n == 2
                iwork = false
            end
        end
    end

    return ham_
end
"""
Build the drive Hamiltonian for the quantum computer
    H_drive = ∑_i Ω_i (exp(iν_i t) a_i + exp(-iν_i t) a_i^†)
"""
function build_drive_hamiltonian(qch::QuantumComputerHamiltonian, drive::DriveHamiltonian, t::Float64)
    n, m, basis, a = qch.n, qch.m, qch.basis, qch.a
    Ω, ν = drive.Ω, drive.ν

    hdrive = zero(qch.a[1]) 

    for q in 1:n
        a_q = a[q]
        a_q_dag = dagger(a_q)

        term = Ω[q] * (exp(1.0im * ν[q] * t) * a_q + exp(-1.0im * ν[q] * t) * a_q_dag)
        hdrive += term
    end

    return hdrive
end



# Define the interaction Hamiltonian in the interaction picture
function build_interaction_hamiltonian(H0::AbstractMatrix, Vt, t::Float64)
    exp_iH0t = exp(1im * H0 * t)
    exp_minus_iH0t = exp(-1im * H0 * t)
    return exp_iH0t * Vt * exp_minus_iH0t
end


n = 2  # Number of qubits
m = 5  
device = Device()

qch = create_hamiltonian(n, m, device)

Ω = [0.1, 0.2]  # Drive amplitudes
ν = [0.9, 1.1]  # Drive frequencies
t = 1.0  # Time

# Build the static Hamiltonian
H0 = build_static_hamiltonian(qch)
println("Static Hamiltonian:")
println(H0)
Vt= build_drive_hamiltonian(qch, DriveHamiltonian(Ω, ν), t)
# Build the interaction Hamiltonian in the interaction picture
VtI = build_interaction_hamiltonian(H0, Vt, t)
println("Interaction Hamiltonian in the interaction picture:")
println(VtI)
