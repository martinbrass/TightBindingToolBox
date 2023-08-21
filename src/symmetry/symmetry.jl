module Symmetry
    include("groups.jl")
    export conjugate, coset, quotient, conjugacy_classes, generate_group

    include("symmetry_operations.jl")
    export SymOperation, translations, SpinRep
    
end