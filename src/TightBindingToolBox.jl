module TightBindingToolBox
    include("core.jl")
    using .Core
    export TB_Hamiltonian, add_hoppings!, bloch_hamiltonian, bloch_hamiltonian!,
        slab_hamiltonian!

    include("TBHamiltonian.jl")
    using .TBH
    export TBHamiltonian

    include("bandstructure.jl")
    using .Bandstructure
    export bandstructure, plot_Bandstructure, DOS, plot_DOS,
        density_matrix, plot_pDOS, surface_spectral_density,
        surface_bands, projected_spectral_density,
        surface_spectral_spin_density

    include("parser.jl")
    using .Parser
    export FPLO_import_TB, Wannier90_import_TB, FPLO_import_space_group, FPLO_get_symop

    include("berry.jl")
    using .Berry
    export discretize_BZ, assign_fibre, scan_BZ_for_Weyl_points, check_wp_candidates,
        integrate_berry_curvature_sphere, refine_wp, print_Berry_curvature,
        berry_flux_through_plane, integrate_connection_along_path, berry_force!,
        search_weyl_points, plot_curvature, integrate_berry_curvature_donut,
        spin_texture

    include("symmetry/symmetry.jl")
    using .Symmetry
    export conjugate, coset, quotient, conjugacy_classes, generate_group, irreps,
        SymOperation, translations, SpinSym, orbit, stabilizer, isgroup, characters,
        representation, plot_wignerseitz, projection, class_labels

    include("tb_model.jl")
    using .Model
    export TB_model, init_terms, add_terms!
end