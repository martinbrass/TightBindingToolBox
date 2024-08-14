using TightBindingToolBox
using Test

@testset verbose=true "TightBindingToolBox.jl" begin
    include("core.jl")
    include("berry.jl")
    include("model.jl")
    #include("TightBindingHamiltonian.jl")
end
