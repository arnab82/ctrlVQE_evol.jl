using ctrlVQE_evol
using Test

@testset "ctrlVQE_evol.jl" begin
    @test ctrlVQE_evol.ctrl_VQE() == "hello ctrl-VQE"
    @test ctrlVQE_evol.ctrl_VQE() != "Hello world!"
end
