using LinearAlgebra

struct TightBindingHamiltonian{F <: Number, I <: Number}
    terms::Dict{Array{I,1},Array{F,2}}
    Hk::Matrix{ComplexF64}
end

TightBindingHamiltonian(d) = TightBindingHamiltonian(Dict{Array{Int,1},Array{ComplexF64,2}}(),zeros(ComplexF64,d,d))

function Base.getindex(H::TightBindingHamiltonian{F,I},key) where {F,I}
    if haskey(H.terms,key)
        return getindex(H.terms,key)
    else
        d = number_of_bands(H)
        return zeros(F,d,d)
    end
end

#Base.getindex(H::TightBindingHamiltonian,keys...) = getindex(H.terms,[keys...])

Base.setindex!(H::TightBindingHamiltonian,h,R) = setindex!(H.terms,h,R)
    

#Base.setindex!(H::TightBindingHamiltonian,h,keys...) = setindex!(H,h,[keys...])

number_of_bands(H::TightBindingHamiltonian) = size(H.Hk,1)
lattice_dim(H::TightBindingHamiltonian) = length(first(keys(H.terms))) 

function (H::TightBindingHamiltonian)(k)
    Hk = zero(H.Hk)
    for (R,t) in pairs(H.terms)
        Hk .+= t .* exp(2π*im * (k⋅R)) 
    end
    return Hk
end

function bloch_hamiltonian!(H::TightBindingHamiltonian,k) 
    fill!(H.Hk,zero(ComplexF64))
    for (R,t) in pairs(H.terms)
        H.Hk .+= t .* exp(2π*im * (k⋅R)) 
    end
end

function derivatives!(H::TightBindingHamiltonian,k,∂Hk)
    fill!(H.Hk,zero(eltype(H.Hk)))
    fill!.(∂Hk,zero(eltype(H.Hk)))
    for (R,t) in pairs(H.terms)
        H.Hk .+= t .* exp(2π*im * (k⋅R)) 
        for (m,r) in pairs(R) 
            ∂Hk[m] .+= t .* (exp(2π*im * (k⋅R)) * im *r) .*2π
        end
    end
end

function derivatives!(H::TightBindingHamiltonian,k,∂Hk::Array{ComplexF64,3})
    fill!(H.Hk,zero(eltype(H.Hk)))
    fill!(∂Hk,zero(ComplexF64))
    for (R,t) in pairs(H.terms)
        H.Hk .+= t .* exp(2π*im * (k⋅R)) 
        for (m,r) in pairs(R) 
            (@view ∂Hk[:,:,m]) .+= t .* (exp(2π*im * (k⋅R)) * im *r) .*2π
        end
    end
end

"""
slab_hamiltonian! calculates the Hamiltonian for a slap which is finite in direction bz
where it has n_layer layers and infinite in the plane perpendicular to bz. It is evaluated 
at momentum k and stored in the initialized matrix Hs
"""
function slab_hamiltonian!(H::TightBindingHamiltonian,
                            k::Vector,
                            bz::Vector,
                            n_layer::Integer,
                            Hs) 
    fill!(Hs,zero(eltype(Hs)))
    d = number_of_bands(H)
    for (R,t) in pairs(H.terms)
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