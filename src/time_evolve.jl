
using SparseArrays
using LinearAlgebra
mutable struct Pulsec
    nqubit::Int
    amp::Vector{Vector{Float64}}
    tseq::Vector{Vector{Float64}}
    freq::Vector{Float64}
    duration::Float64
end
function prune!(matrix, tol::T) where T
    for i in 1:nnz(matrix)  # nnz(matrix) returns the number of stored elements (non-zeros)
        if abs(matrix.nzval[i]) < tol
            matrix.nzval[i] = 0
        end
    end
    dropzeros!(matrix)
end
function getham(t::Float64, pobj::Pulsec,
                hdrive::Vector{Vector{SparseMatrixCSC{Complex{Float64}, Int64}}}, 
                # hdrive::Vector{Any},
                dsham::Matrix{ComplexF64}, dsham_len::Int,
                matexp_::SparseMatrixCSC{Complex{Float64}})
    
    # Initialize hamdr as a sparse matrix with the correct size
    hamdr = spzeros(Complex{Float64}, dsham_len, dsham_len)
    
    for i in 1:pobj.nqubit
        hcoef = pcoef(t, pobj.amp[i], pobj.tseq[i], pobj.freq[i], pobj.duration)
        hcoefc = conj(hcoef)
        
        if i == 1
            hamdr = hcoef * hdrive[i][1]
        else
            hamdr += hcoef * hdrive[i][1]
        end
        
        hamdr += hcoefc * hdrive[i][2]
    end
    #println(dsham_len)
    #display(dsham)
    # Exponentiate the diagonal elements of dsham
    for i in 1:dsham_len
        matexp_[i, i] = exp(dsham[i] * t)
    end
    #display(matexp_)
    #display(hamdr)
    # Perform the Hamiltonian transformation
    hamr_ = transpose(conj(matexp_)) * hamdr
    hamr_ = prune!(hamr_ * matexp_, 1e-10)

    return hamr_
end

function solve_func(t::Float64, y::Vector{ComplexF64},
                    pobj::Pulsec,
                    hdrive::Vector{Vector{SparseMatrixCSC{Complex{Float64}, Int64}}}, 
                    # hdrive::Vector{Any},
                    dsham::Matrix{ComplexF64}
                )
    
    dsham_len = size(dsham)[1]
    #println("length of rows in static hamiltonian", dsham_len)
    matexp_ = spzeros(Complex{Float64}, dsham_len, dsham_len)
    
    # Assuming getham is defined elsewhere
    H = getham(t, pobj, hdrive, dsham, dsham_len, matexp_)
    
    # Convert y to a Julia Vector for operations
    y_ = y
    
    # Perform the matrix-vector multiplication and scaling by -im (i.e., -1.0im in Julia)
    H_ = -1.0im * H * y_
    
    return H_
end
using DifferentialEquations

function evolve(ini_vec, pobj::Pulsec, qch::QuantumComputerHamiltonian; 
                solver::String="ode", nstep::Int=2000, twindow::Bool=true)
    # Extract the diagonal elements of qch.dsham, multiply by -1im, and convert to a vector
    dsham = -1im * diagm(diag(qch.dsham))

    
    # Create a time vector similar to numpy.linspace
    tlist = range(0, stop=pobj.duration, length=nstep)
    
    if solver == "ode"
        # Define the ODE problem
        function ode_func!(du, u, p, t)
            du .= solve_func(t, u, pobj, qch.Hdrive, dsham)
        end
        
        # Set up the ODE problem
        prob = ODEProblem(ode_func!, ini_vec, (0.0, pobj.duration))
        
        # Solve the ODE using DifferentialEquations.jl
        sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-12)
        
        return sol[end]  # Return the state vector at the final time
    
    elseif solver == "trotter"
        tmp_ = solve_trotter(tlist, ini_vec, pobj, qch.Hdrive, dsham)
        return tmp_
    
    else
        error("Solver doesn't exist or is not implemented")
    end
end
using Expokit

function solve_trotter(tlist, ini_vec,
                       pobj::Pulsec, hdrive, 
                       dsham)

    dsham_len = size(dsham)[1]
    matexp_ = spzeros(ComplexF64, dsham_len, dsham_len)
    
    trot_ = ini_vec  # This serves as the evolving state vector
    tlen = length(tlist)
    tau = tlist[end] / tlen
    im_tau = Complex{Float64}(0.0, -tau)

    for t in 1:tlen
        println("t=", t)
        H_ = getham(tlist[t], pobj, hdrive, dsham, dsham_len, matexp_)
        H1_ = im_tau * Matrix(H_)  # Convert H_ to dense matrix

        # Use matrix exponential to evolve the state vector
        trot_ = exp(H1_) * trot_
    end

    return trot_
end



