using TightBindingToolBox
using Test

@testset verbose=true "TightBindingToolBox.jl" begin
    include("TightBindingHamiltonian.jl")
    include("berry.jl")
end
