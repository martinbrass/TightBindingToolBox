module TightBindingToolBox
    include("core.jl")
    using .Core
    export TB_Hamiltonian, add_hoppings!, bloch_hamiltonian, bloch_hamiltonian!,
           slab_hamiltonian!

    include("bandstructure.jl")
    using .Bandstructure
    export bandstructure, plot_Bandstructure, DOS, plot_DOS,
           density_matrix, plot_pDOS

    include("parser.jl")
    using .Parser
    export FPLO_import_TB, Wannier90_import_TB

    include("berry.jl")
    using .Berry
    export discretize_BZ, assign_fibre, scan_BZ_for_Weyl_points, check_wp_candidates,
           integrate_berry_curvature_sphere, refine_wp, print_Berry_curvature,
           berry_flux_through_plane, integrate_connection_along_path
end