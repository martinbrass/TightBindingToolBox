module Model
    using ..TightBindingToolBox, LinearAlgebra
    export TB_model, init_terms, add_terms!
    """
    TB_model{F,T} is a struct that holds important information for defining a tight binding model.
    it should only be created using TB_model(G::Set{T},ρ::Dict{T, Matrix{F}})

        G - point group
        ρ - representation of the orbitals
        ρH - operator representation, derived from ρ
        χ - characters of irreps of g
        op_basis - basis for hermitian matrices
        H - tight binding Hamiltonian of the model
    """
    struct TB_model{F,T}
        G::Set{T}
        ρ::Dict{T, Matrix{F}}
        ρH::Dict{T, Matrix{F}}
        χ::Matrix{F}
        op_basis::Vector{Matrix{ComplexF64}}
        H::TB_Hamiltonian{ComplexF64,Int}
    end

    """
    TB_model(G,ρ) initializes a tight binding model where
        G - point group
        ρ - representation of the orbitals

    """
    function TB_model(G::Set{T},ρ::Dict{T, Matrix{F}}) where {T,F}
        χ = irreps(G)
        n = size(first(ρ)[1])[1] # lattice dim
        d = size(first(ρ)[2])[1] # local orbitals dim
        op_basis = operator_basis(d)
        ρH = operator_rep(op_basis,ρ)
        H = TB_Hamiltonian{ComplexF64,Int}(d,n)

        return TB_model{F,T}(G,ρ,ρH,χ,op_basis,H)
    end

    """
    operator_basis(N) creates a basis of NxN hermitian matrices
    """
    operator_basis(N) = vcat(
        [Matrix(Diagonal([i==j ? 1.0+0im : 0.0im for j = 1:N])) for i=1:N],
        [begin
            E = zeros(ComplexF64,N,N);
            E[j,k] = 1/sqrt(2);
            E[k,j] = 1/sqrt(2);
            E
        end for j = 1:N for k = j+1:N
        ],
        [begin
            E = zeros(ComplexF64,N,N);
            E[j,k] = -1im/sqrt(2);
            E[k,j] = 1im/sqrt(2);
            E
        end for j = 1:N for k = j+1:N
        ]
    )

    """
    operator_rep(basis,ρ)
        basis spans the space of hermitian operators for some orbitals
        ρ is a representation acting on these orbitals
    the representation acting on operators H is defined as ρ*H*ρ'
    the function operator_rep maps this representation on matrices
    """
    operator_rep(basis,ρ) = Dict([
        g => [tr(A'*r*B*r') for A in basis, B in basis]
        for (g,r) in ρ 
    ])

    """
    orbit_rep(G,v)
        G is a group acting on some object v
        the orbit of v under G defines a representation of G which is returned by this function
    """
    function orbit_rep(G,v)
        orb = collect(orbit(G,v))
        d = length(orb)
        ρ = Dict([
            g => [ i == findfirst(x->x==g*orb[j],orb) ? 1.0 : 0.0 for i=1:d, j=1:d]
            for g in G
        ])
        return ρ
    end

    """
    get_basis_for_irrep(χ,ρ,G)
    projects a given representation ρ of group G onto the compononent isotypic to an irreducible
    representation witch chacracters χ and returns a basis for this sub space
    """
    function get_basis_for_irrep(χ,ρ,G)
        Cls = conjugacy_classes(G)
        dim = χ[TightBindingToolBox.Symmetry.get_index(TightBindingToolBox.Symmetry.neutral_element(G),Cls)]
        P = projection(χ,ρ,G)
        E,V = LAPACK.syev!('V','U',P)
        n = length(E)
        d = round(Int64,sum(E))
        return V[:,n-d+1:n] # TODO: is this important?

        if d == dim
            return V[:,n-d+1:n]
        elseif !(d % dim == 0)
            @error "get_basis_for_irrep: $dim does not divide $d"
            return nothing
        else
            v = @view V[:,n-d+1:n]
            r = [v'*g*v for g in values(ρ)]
            f = Diagonal(rand(d))
            H = sum(g*f*g' for g in r)
            E, W = LAPACK.syev!('V','U',H)
            return v*W
        end
    end

    """
    init_terms(M::TB_model,R)
    given a lattice vector R we can define hoppings along all directions in the orbit of R
    this function determines the number of necessary hopping parameters
        the output is intended only for usage as input for add_terms! (see below)
        on return:
            V - basis for the trivial representation
            Rs - orbit of R
        
    """
    function init_terms(M::TB_model,R)
        G = M.G
        Rs = collect(orbit(G,R))
        ρR = orbit_rep(G,R)
        Cls = conjugacy_classes(G)
        ω = Diagonal([length(C) for C in Cls]) / length(G)

        ρRH = Dict([
            g => kron(ρR[g],M.ρH[g])
            for g in M.G
        ])
        A1g = ones(Int,length(Cls))
        χRH = characters(ρRH,Cls)
        n_param = round(Int,real(((A1g ⋅ (ω*χRH) ))))
        @info "The TB_model needs $n_param parameters for the given hoppings."
        V=get_basis_for_irrep(A1g,ρRH,G)

        return V, Rs
    end

    """
    add_terms! adds all symmetry allowed hoppings to the Hamiltonian of the TB_model M
        V, Rs  - output from init_terms
        params - free parameters of the model, number is determined by init_terms
    """
    function add_terms!(M::TB_model,V,Rs,param)
        H = M.H
        d = length(M.op_basis)

        for (a,R) in pairs(Rs)
            for (b,τ) in pairs(M.op_basis)
                n = d*(a-1) + b
                for (i,t) in pairs(param)
                    h = t * V[n,i] * τ
                    add_hoppings!(H,R,h) # TODO: unroll loop
                end
            end
        end

    end
end