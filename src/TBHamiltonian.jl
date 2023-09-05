module TBH
    export TBHamiltonian
    using LinearAlgebra, StaticArrays, ..TightBindingToolBox, .Core

    struct TBHamiltonian{D,N,F,DN}
        lattice::SMatrix{D,N,Int,DN}
        hopping::Array{F,3}
    end

    function TBHamiltonian{D,N}(H::TB_Hamiltonian{F}) where {D,N,F}
        @assert D == H.lattice_dim "conversion of TBHamiltonian failed due to incorrect lattice dimension"
        @assert N == length(H.hoppings) "conversion of TBHamiltonian failed due to incorrect number of hoppings"
        n = H.local_dim
        latt = Matrix{Int}(undef,D,N)
        hopp = Array{F,3}(undef,n,n,N)
        i = 1
        for (R, t) in pairs(H.hoppings)
            latt[:,i]   .= R
            hopp[:,:,i] .= t
            i += 1
        end
        return TBHamiltonian{D,N,F,D*N}(SMatrix{D,N}(latt),hopp)
    end

    function TightBindingToolBox.Core.bloch_hamiltonian!(H::TBHamiltonian{D,N,F},k,Hk::Array{ComplexF64,2}) where {D,N,F}
        Hk .*= zero(F) 
        for i = 1:N
            t = @view H.hopping[:,:,i]
            R = @view H.lattice[:,i]
            Hk .+= t .* exp(2π*im * (k⋅R)) 
        end
    end

end