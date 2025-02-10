module Symmetry
    include("groups.jl")
    export conjugate, coset, quotient, conjugacy_classes, generate_group, irreps, orbit,
        stabilizer, isgroup, characters, stab, projection, class_labels

    include("euclidean_symmetry.jl")
    export SymOperation, translations, SpinSym, representation

    # implement in "../../ext/PlotsExt.jl"
    function plot_wignerseitz end
    export plot_wignerseitz
    
end