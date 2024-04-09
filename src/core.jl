module Core
    export TB_Hamiltonian, add_hoppings!, bloch_hamiltonian, bloch_hamiltonian!,
           slab_hamiltonian! 
    
    using LinearAlgebra, CSV, DataFrames,Dictionaries

    """
    TB_Hamiltonian{F,L} is the main structure that represents a tight-binding Hamiltonian.
    F denotes the field (nomplex or real numbers typically) and L denotes the type of the 
    components of reciprocal lattice vectors (typically Int or rational, don't use Float)
    local_dim = dimension of local Hilbert-space, i.e. number of (spin-)orbitals or bands
    lattice_dim = dimension of the Bravais-lattice
    hoppings = Dictionary which has lattice vectors as keys and hopping-matrices as values
    """
    struct TB_Hamiltonian{F,L}
        local_dim::Int64
        lattice_dim::Int64
        hoppings::Dictionary{Array{L,1},Array{F,2}}
    end

    TB_Hamiltonian{F,L}(n,d::Int64) where {F,L} = TB_Hamiltonian(n,d,Dictionary{Array{L,1},Array{F,2}}())

    """
    add_hoppings! adds the (key,value) pair (R,h) to the Hamiltonian h
    """
    function add_hoppings!(H::TB_Hamiltonian{F,L},R::Array{L,1},h) where {F,L}
        if haskey(H.hoppings,R)
            H.hoppings[R] .+= h
        elseif length(R) != H.lattice_dim
            throw(DimensionMismatch("at add_hoppings! : lattice vector has incompatible dimension"))
        elseif size(h) != (H.local_dim, H.local_dim)
            throw(DimensionMismatch("at add_hoppings! : hopping matrix has incompatible dimension"))
        else
            insert!(H.hoppings,R,copy(h))
        end
    end

    """
    bloch_hamiltonian fourier transforms H and evaluates at momentum k
    bloch_hamiltonian! is an inplace version where Hk needs to be initialized
    """
    function bloch_hamiltonian(H::TB_Hamiltonian{F,L},k) where {F,L}
        n = H.local_dim
        Hk = zeros(F,n,n)
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
        return Hk
    end

    function bloch_hamiltonian!(H::TB_Hamiltonian{F,L},k,Hk) where {F,L}
        Hk .*= 0
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
    end

    """
    slab_hamiltonian! calculates the Hamiltonian for a slap which is finite in direction bz
    where it has n_layer layers and infinite in the plane perpendicular to bz. It is evaluated 
    at momentum k and stored in the initialized matrix Hs
    """
    function slab_hamiltonian!(H::TB_Hamiltonian{F,L},
                               k::Array{T,1},
                               bz::Array{L,1},
                               n_layer::Integer,
                               Hs
                               ) where {F,L,T<:Real}
        Hs .*= 0
        d = H.local_dim
        for (R,t) in pairs(H.hoppings)
            n = bz ⋅ R
            ϕ = exp(2π*im * (k⋅R))
            if n > 0 
                for m = 0:n_layer-1-n
                    i = (m + n) * d
                    j = m*d
                    hs = @view Hs[i+1:i+d , j+1:j+d]
                    @. hs += t * ϕ 
                end
            else
                for m = 0:n_layer-1+n
                    i = m*d
                    j = (m - n) * d
                    hs = @view Hs[i+1:i+d , j+1:j+d]
                    @. hs += t * ϕ 
                end
            end
        end
    end


end