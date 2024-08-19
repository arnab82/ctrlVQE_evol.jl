include("./hamiltonian.jl")
"""
Module for time evolution of quantum states using differential equations.


"""
function time_evolution_diff_eqs(qch::QuantumComputerHamiltonian, dh::DriveHamiltonian, psi0::Vector{ComplexF64}, T::Float64, dt::Float64)
    function schrodinger_eq!(dpsi, psi, p, t)
        H = build_total_hamiltonian(qch, dh, t)
        dpsi[:] = -1im * H.data * psi
    end

    prob = ODEProblem(schrodinger_eq!, psi0, (0.0, T))
    sol = solve(prob, Tsit5(), dt=dt)
    return sol
end

"""
Module for time evolution of quantum states using Trotterization.
"""
function time_evolution_trotterization(qch::QuantumComputerHamiltonian, dh::DriveHamiltonian, psi0::Vector{ComplexF64}, T::Float64, r::Int)
    dt = T / r
    psi = psi0
    for i in 1:r
        t = i * dt
        Vt = build_drive_hamiltonian(qch, dh, t)
        psi = exp(-1im * dt * Vt.data) * psi
    end
    return psi
end

# Example usage
n = 2 # Number of modes
m = 3 # number of states

device = Device(n, m)

# Initial state (ground state)
psi0 = zeros(ComplexF64, m^n)
psi0[1] = 1.0

T = 1.0  # Total evolution time
dt = 0.01  # Time step for differential equation solver
r = 100  # Number of Trotter steps
dh=hdrive
# Time evolution using differential equations
sol_diff_eqs = time_evolution_diff_eqs(qch, dh, psi0, T, dt)
psi_final_diff_eqs = sol_diff_eqs(T)

# Time evolution using Trotterization
psi_final_trotterization = time_evolution_trotterization(qch, dh, psi0, T, r)

# Print the final states
println("Final state using differential equations:")
println(psi_final_diff_eqs)

println("Final state using Trotterization:")
println(psi_final_trotterization)

