using .TightBindingToolBox, LinearAlgebra

function bandstructure(H::TightBindingHamiltonian,k_path) 
    n = number_of_bands(H)
    l = length(k_path)
    bands = zeros(l,n)
    for (idx,k) in pairs(k_path)
        bloch_hamiltonian!(H,k)
        bands[idx,:] .=  LAPACK.syev!('N','U',H.Hk)
    end
    return bands
end

#TODO: surface

function path_in_bz(points,n)
    k1 = first(points)
    kpath = [k1]
    for k2 in Iterators.drop(points,1)
        δk = (k2-k1) ./n
        for i = 1:n
            k = k1 + i*δk
            push!(kpath,k)
        end
        k1 = k2
    end
    return kpath
end

gauss(x,σ) = exp( -(x/σ)^2 /2 ) /sqrt(2π) /σ

function spectral_function(H::TightBindingHamiltonian,W,nk,σ)
    rk = range(0,1-1/nk,nk)
    A = [zero(H.Hk) for i = 1:length(W)]
    N = nk*nk*nk #TODO: check correct norm
    for x in rk, y in rk, z in rk
        k = [x,y,z]
        bloch_hamiltonian!(H,k)
        E, V = LAPACK.syev!('V','U',H.Hk)
        for (i,w) in pairs(W)
            A[i] .+= V * Diagonal([gauss(w -ϵ,σ) for ϵ in E]) * V' ./N
        end
    end
    return A
end

"""
bz is a reciprocal lattice vector orthogonal to the surface
bx,by are reciprocal lattice vectors that span the surface
n_layer defines how many layers are contained in a superlayer (will become deprecated soon)
n_super_layer defines the number of superlayers (slap then has n_super_layer*n_layer layers)
the inplane momentum goes from -1/2 to 1/2 unless specified otherwise via kx_offset/ky_offset
"""
function surface_spectral_density(H::TightBindingHamiltonian,
                                ω::Number,
                                bx::Vector,
                                by::Vector,
                                bz::Vector,
                                n_super_layer::Integer,
                                n_kpts::Integer,
                                kx_offset = -1/2,
                                ky_offset = -1/2)
    n_layer = maximum([x for R in keys(H.terms) for x in R])
    d = number_of_bands(H)
    Hs = zeros(ComplexF64,2*n_layer*d,2*n_layer*d)
    A = zeros(n_kpts,n_kpts)
    Y = zeros(ComplexF64,n_layer*d,n_layer*d)
    for y = 1:n_kpts
        ky = (y-1)/(n_kpts-1) + ky_offset
        for x = 1:n_kpts
            kx = (x-1)/(n_kpts-1) + kx_offset
            k = kx * bx + ky * by
            slab_hamiltonian!(H,k,bz,2*n_layer,Hs)
            h0 = I*ω - Hs[1:d*n_layer,1:d*n_layer]
            T = @view Hs[1:d*n_layer,1+d*n_layer:2*d*n_layer];
            Td= T'
            Gr = inv(h0)                
            for i=1:n_super_layer-1
                mul!(Y,Gr,Td)
                mul!(Gr,T,Y)
                Gr .= h0 .- Gr
                Gr = inv(Gr)
            end
            A[x,y] -= imag(tr(@view Gr[1:d,1:d])) 
        end
    end
    return A
end

"""
same as surface_spectral_density but now for the spinoperator S (list of matrices)
"""
function surface_spectral_spin_density(H::TightBindingHamiltonian,
                                    S,
                                    ω::Number,
                                    bx::Vector,
                                    by::Vector,
                                    bz::Vector,
                                    n_super_layer::Integer,
                                    n_kpts::Integer,
                                    kx_offset = -1/2,
                                    ky_offset = -1/2)
    d = number_of_bands(H)
    n_layer = maximum([x for R in keys(H.terms) for x in R])
    Hs = zeros(ComplexF64,2*n_layer*d,2*n_layer*d)
    ns = length(S)
    A = zeros(ns,n_kpts,n_kpts)
    Y = zeros(ComplexF64,n_layer*d,n_layer*d)
    for y = 1:n_kpts
        ky = (y-1)/(n_kpts-1) + ky_offset
        for x = 1:n_kpts
            kx = (x-1)/(n_kpts-1) + kx_offset
            k = kx * bx + ky * by
            slab_hamiltonian!(H,k,bz,2*n_layer,Hs)
            h0 = I*ω - Hs[1:d*n_layer,1:d*n_layer]
            T = @view Hs[1:d*n_layer,1+d*n_layer:2*d*n_layer];
            Td= T'
            Gr = inv(h0)                
            for i=1:n_super_layer-1
                mul!(Y,Gr,Td)
                mul!(Gr,T,Y)
                Gr .= h0 .- Gr
                Gr = inv(Gr)
            end
            for (i,σ) in pairs(S)
                A[i,x,y] -= imag(tr(σ * @view Gr[1:d,1:d])) 
            end
        end
    end
    return A
end

"""
same as surface_spectral_density but for the bulk only, i.e without surface states
"""
function projected_spectral_density(H::TightBindingHamiltonian,
                                    ω::Number,
                                    bx::Vector,
                                    by::Vector,
                                    bz::Vector,
                                    n_kpts::Integer,
                                    n_kpts_z::Integer,
                                    kx_offset = -1/2,
                                    ky_offset = -1/2)
    A = zeros(n_kpts,n_kpts)
    r = range(-1/2,1/2,n_kpts)
    for y = 1:n_kpts
        ky = (y-1)/(n_kpts-1) + ky_offset
        for x = 1:n_kpts
            kx = (x-1)/(n_kpts-1) + kx_offset
            for z = 1:n_kpts_z
                kz = (z-1)/(n_kpts_z-1) - 1/2
                k = kx * bx + ky * by + kz * bz
                bloch_hamiltonian!(H,Hk)
                for i = 1:d
                    H.Hk[i,i] -= ω 
                end
                G = inv(H.Hk)
                A[x,y] += imag(tr(G)) 
            end
        end
    end
    return A
end

"""
energy dispersion along k_path (list of momenta) on a surface
    perpendicular to bz (reciprocal lattice vector)
n_layer defines how many layers are contained in a superlayer (will become deprecated soon)
n_super_layer defines the number of superlayers (slap then has n_super_layer*n_layer layers)
"""
function surface_bands(H::TightBindingHamiltonian,
                        k_path,
                        ω_pts,
                        bz::Vector,
                        n_super_layer::Integer)
    d = number_of_bands(H)
    n_layer = maximum([x for R in keys(H.terms) for x in R])
    n_kpts = length(k_path)
    n_ω = length(ω_pts)
    Hs = zeros(ComplexF64,2*n_layer*d,2*n_layer*d)
    A = zeros(n_ω,n_kpts)
    Y = zeros(ComplexF64,n_layer*d,n_layer*d)
    for (y,k) in pairs(k_path)
        for (x,z) in pairs(ω_pts)
            slab_hamiltonian!(H,k,bz,2*n_layer,Hs)
            h0 = I*z - Hs[1:d*n_layer,1:d*n_layer]
            T = @view Hs[1:d*n_layer,1+d*n_layer:2*d*n_layer];
            Td= T'
            Gr = inv(h0)                
            for i=1:n_super_layer-1
                mul!(Y,Gr,Td)
                mul!(Gr,T,Y)
                Gr .= h0 .- Gr
                Gr = inv(Gr)
            end
            A[x,y] -= imag(tr(@view Gr[1:d,1:d])) 
        end
    end
    return A
end