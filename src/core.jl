module Core
    export TB_Hamiltonian, add_hoppings!, bloch_hamiltonian, bloch_hamiltonian! 
    
    using LinearAlgebra, CSV, DataFrames

    struct TB_Hamiltonian{F,L}
        local_dim::Int64
        lattice_dim::Int64
        hoppings::Dict{Array{L,1},Array{F,2}}
        Hk::Array{F,2}
    end

    TB_Hamiltonian{F,L}(n,d::Int64) where {F,L} = TB_Hamiltonian(n,d,Dict{Array{L,1},Array{F,2}}(),zeros(F,n,n))

    function add_hoppings!(H::TB_Hamiltonian{F,L},R::Array{L,1},h::Array{F,2}) where {F,L}
        if haskey(H.hoppings,R)
            H.hoppings[R] .+= h
        elseif length(R) != H.lattice_dim
            throw(DimensionMismatch("at add_hoppings! : lattice vector has incompatible dimension"))
        elseif size(h) != (H.local_dim, H.local_dim)
            throw(DimensionMismatch("at add_hoppings! : hopping matrix has incompatible dimension"))
        else
            H.hoppings[R] = copy(h)
        end
    end

    function bloch_hamiltonian(H::TB_Hamiltonian{F,L},k) where {F,L}
        n = H.local_dim
        Hk = zeros(F,n,n)
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
        return Hk
    end

    function bloch_hamiltonian!(H::TB_Hamiltonian{F,L},k,Hk::Array{F,2}) where {F,L}
        Hk .*= zero(F) 
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
    end


end