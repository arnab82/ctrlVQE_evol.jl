using DifferentialEquations
using LinearAlgebra

using DifferentialEquations
using LinearAlgebra

# Define parameters
const Ω = 1.0        # Amplitude of the EM field
const ν = 1.0        # Frequency of the EM field
const Δ = 0.5        # Detuning (Ω - ν)

# Define the initial state (example: superposition of |0⟩ and |1⟩)
ψ0 = ComplexF64[1.0, 0.0]  # |ψ(0)⟩ = |0⟩

# Define the interaction Hamiltonian in the interaction picture
function VI(t)
    exp_iH0t = [1.0 0.0; 0.0 exp(im*Ω*t)]
    exp_minus_iH0t = [1.0 0.0; 0.0 exp(-im*Ω*t)]
    V_t = [0 Ω*exp(im*ν*t); Ω*exp(-im*ν*t) 0]
    return exp_iH0t * V_t * exp_minus_iH0t
end

# Define the Schrödinger equation in the interaction picture
function schrodinger!(dψI, ψI, p, t)
    dψI .= -im * VI(t) * ψI
end

# Define the time span for the evolution
tspan = (0.0, 100.0)  # From t=0 to t=100

# Set up the initial state for the differential equation solver
prob = ODEProblem(schrodinger!, ψ0, tspan)

# Solve the Schrödinger equation
sol = solve(prob)

# Extract the solution
times = sol.t
states = sol.u

# Print the evolved state at final time
println("State at t=$(tspan[2]):")
println(states[end])








function ctrl_VQE()
	return "hello ctrl-VQE"
end
