module TightBindingToolBox
    include("TightBindingHamiltonian.jl")
    export TightBindingHamiltonian,
        number_of_bands,
        bloch_hamiltonian!, 
        slab_hamiltonian!,
        derivatives!

    include("symmetry/symmetry.jl")
    using .Symmetry
    export conjugate, 
        coset, 
        quotient, 
        conjugacy_classes, 
        generate_group, 
        irreps,
        SymOperation, 
        translations, 
        SpinSym, orbit, 
        stabilizer, 
        isgroup, 
        characters,
        representation, 
        plot_wignerseitz, 
        projection, 
        class_labels

    include("parser.jl")
    export FPLO_import_TB,
        Wannier90_import_TB, 
        FPLO_import_space_group, 
        FPLO_get_symop

    include("bandstructure.jl")
    export path_in_bz, 
        bandstructure, 
        spectral_function,
        surface_bands,
        surface_spectral_density,
        surface_spectral_spin_density,
        projected_spectral_density

    include("berry.jl")
    export BerryCurvature,
        Plane,
        Sphere,
        flux,
        berry_flux,
        berry_curvature, 
        berry_curvature!,
        search_weyl_points
        
end