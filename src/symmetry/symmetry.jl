module Symmetry
    include("groups.jl")
    export conjugate, coset, quotient, conjugacy_classes, generate_group, irreps, orbit,
        stabilizer, isgroup, characters, stab, projection

    include("euclidean_symmetry.jl")
    export SymOperation, translations, SpinSym, representation

    include("wignerseitz.jl")
    export plot_wignerseitz
    
end