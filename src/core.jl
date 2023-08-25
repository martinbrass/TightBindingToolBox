module Core
    export TB_Hamiltonian, add_hoppings!, bloch_hamiltonian, bloch_hamiltonian!,
           slab_hamiltonian! 
    
    using LinearAlgebra, CSV, DataFrames,Dictionaries

    struct TB_Hamiltonian{F,L}
        local_dim::Int64
        lattice_dim::Int64
        hoppings::Dictionary{Array{L,1},Array{F,2}}
    end

    TB_Hamiltonian{F,L}(n,d::Int64) where {F,L} = TB_Hamiltonian(n,d,Dictionary{Array{L,1},Array{F,2}}())

    function add_hoppings!(H::TB_Hamiltonian{F,L},R::Array{L,1},h::Array{T,2}) where {F,L,T<:Number}
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

    function bloch_hamiltonian(H::TB_Hamiltonian{F,L},k) where {F,L}
        n = H.local_dim
        Hk = zeros(F,n,n)
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
        return Hk
    end

    function bloch_hamiltonian!(H::TB_Hamiltonian{F,L},k,Hk::Array{ComplexF64,2}) where {F,L}
        Hk .*= zero(F) 
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
    end

    function slab_hamiltonian!(H::TB_Hamiltonian{F,L},k::Array{T,1},bz::Array{L,1},n_layer::Integer,Hs::Array{ComplexF64,2}) where {F,L,T<:Real}
        Hs .*= 0
        d = H.local_dim
        for (R,t) in pairs(H.hoppings)
            n = bz ⋅ R
            ϕ = exp(2π*im * (k⋅R))
            if n > 0 
                for m = 0:n_layer-1-n
                    i = (m + n) * d
                    j = m*d
                    Hs[i+1:i+d , j+1:j+d] .+= t .* ϕ 
                end
            else
                for m = 0:n_layer-1+n
                    i = m*d
                    j = (m - n) * d
                    Hs[i+1:i+d , j+1:j+d] .+= t .* ϕ 
                end
            end
        end
    end


end