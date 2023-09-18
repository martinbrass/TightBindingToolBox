module Bandstructure
    export bandstructure, plot_Bandstructure, DOS, plot_DOS,
           density_matrix, plot_pDOS, surface_spectral_density,
           surface_bands, projected_spectral_density,
           surface_spectral_spin_density
    
    using ..TightBindingToolBox, LinearAlgebra, Plots, Base.Threads

    function bandstructure(H::TB_Hamiltonian{F,L},path,n_pts)  where {F,L}
        n = H.local_dim
        l = length(path)
        bands = zeros(n_pts*(l-1),n)
        lines = Array{Float64,1}(undef,l-1)
        x = Array{Float64,1}(undef,n_pts*(l-1))
        Hk = zeros(F,n,n)
        for i in 2:l
            lines[i-1]=(i-1)*n_pts+1
            k0 = path[i-1]
            Δk = path[i] - k0
            for j in 0:(n_pts-1)
                idx = (i-2)*n_pts + j + 1
                bloch_hamiltonian!(H,k0 + j*Δk/n_pts,Hk)
                bands[idx,:] .=  LAPACK.syev!('N','U',Hk)
                x[idx] = idx + 1
            end
        end
        return bands, lines, x
    end

    function plot_Bandstructure(H,path,n_pts,label;kwargs...)
        l = length(path)
        bands, lines, x = bandstructure(H,path,n_pts)
        pt = plot(x,bands, framestyle = :box, xlims=(1,n_pts*(l-1)+1),legend=false, xticks=(1:n_pts:(n_pts*(l-1)+1),label);kwargs...)
        vline!(lines,color=:black)
        return pt
    end

    function DOS(H::TB_Hamiltonian,wlist,ngrdpts,γ)
        npts = length(wlist)
        dos = zeros(npts)
        Hk = zeros(ComplexF64,H.local_dim,H.local_dim)
        npts = length(wlist)
        N = ngrdpts*ngrdpts*ngrdpts*π
        for x = 1:ngrdpts
            for y = 1:ngrdpts
                for z = 1:ngrdpts
                    k = [(x-1)/ngrdpts,(y-1)/ngrdpts,(z-1)/ngrdpts]
                    bloch_hamiltonian!(H,k,Hk)
                    ϵk = eigvals(Hk)
                    for i in 1:npts
                        dos[i] -= sum(map((x)->imag(1/(wlist[i]+im*γ-x)),ϵk))/N
                    end
                end
            end
        end
        
        return dos
    end

    function plot_DOS(H::TB_Hamiltonian,wmin,wmax,npts,ngrdpts,γ;kwargs...)
        wlist = map((x)->wmin+(x-1)/(npts-1)*(wmax-wmin),collect(1:npts))
        dos = DOS(H,wlist,ngrdpts,γ)
        return plot(wlist,dos, framestyle = :box,legend=false;kwargs...)
    end
    
    function plot_pDOS(H::TB_Hamiltonian,orbitals,wmin,wmax,npts,ngrdpts,γ)
        dos = zeros(npts)
        Hk = zeros(ComplexF64,H.local_dim,H.local_dim)
        wlist = map((x)->wmin+(x-1)/(npts-1)*(wmax-wmin),collect(1:npts))
        N = ngrdpts*ngrdpts*ngrdpts*π
        for x = 1:ngrdpts
            for y = 1:ngrdpts
                for z = 1:ngrdpts
                    k = [(x-1)/ngrdpts,(y-1)/ngrdpts,(z-1)/ngrdpts]
                    bloch_hamiltonian!(H,k,Hk)
                    ϵk, Tk = LAPACK.syev!('V','U',Hk)
                    for i in 1:npts
                        for j = 1:H.local_dim
                            res = 0
                            for m in orbitals
                                res += Tk[m,j]*conj(Tk[m,j])
                            end
                            dos[i] -= imag(res/(wlist[i]+im*γ-ϵk[j]))/N
                        end
                    end
                end
            end
        end
        plt = plot(wlist,dos, framestyle = :box,legend=false)
        return plt, wlist, dos   
    end
    
    function density_matrix(H::TB_Hamiltonian,ngrdpts)
        d = H.local_dim
        N = ngrdpts*ngrdpts*ngrdpts
        ρ = zeros(ComplexF64,d,d)
        Hk = zeros(ComplexF64,d,d)
        for x = 1:ngrdpts, y = 1:ngrdpts, z = 1:ngrdpts
            k = [(x-1)/ngrdpts,(y-1)/ngrdpts,(z-1)/ngrdpts]
            bloch_hamiltonian!(H,k,Hk)
            ϵk, Tk = LAPACK.syev!('V','U',Hk)
            for i = 1:d, j = 1:d
                for m = 1:d
                    if ϵk[m] > 0 break end
                    ρ[i,j] += Tk[i,m]*conj(Tk[j,m])/N
                end
            end
        end
        return ρ
    end
#The function below is old and embarrasingly slow, use the other one
    function surface_spectral_density(H::TB_Hamiltonian{F,L},
                                      ω::Number,
                                      bx::Array{L,1},
                                      by::Array{L,1},
                                      bz::Array{L,1},
                                      n_layer::Integer,
                                      n_kpts::Integer,
                                      kx_offset = -1/2,
                                      ky_offset = -1/2
                                      ) where {F,L}
        d = H.local_dim
        Hs = zeros(ComplexF64,n_layer*d,n_layer*d,nthreads())
        #ipiv = Array{Int,1}(undef,n_layer*d)
        A = zeros(n_kpts,n_kpts)
        r = range(-1/2,1/2,n_kpts)
        @threads for y = 1:n_kpts
            ky = (y-1)/(n_kpts-1) + ky_offset
            for x = 1:n_kpts
                kx = (x-1)/(n_kpts-1) + kx_offset
                k = kx * bx + ky * by
                slab_hamiltonian!(H,k,bz,n_layer,@view Hs[:,:,threadid()])
                for i = 1:d*n_layer 
                    Hs[i,i,threadid()] -= ω 
                end
                #LAPACK.hetri!('U',Hs,ipiv)
                #LAPACK.getrf!(Hs)
                #LAPACK.getri!(Hs,ipiv)
                G = inv(@view Hs[:,:,threadid()])
                #A[x,y] += imag(tr(@view Hs[1:d,1:d])) 
                A[x,y] += imag(tr(@view G[1:d,1:d])) 
            end
        end
        return A
    end

    function surface_spectral_density(H::TB_Hamiltonian{F,L},
                                    ω::Number,
                                    bx::Array{L,1},
                                    by::Array{L,1},
                                    bz::Array{L,1},
                                    n_super_layer::Integer,
                                    n_layer::Integer,
                                    n_kpts::Integer,
                                    kx_offset = -1/2,
                                    ky_offset = -1/2
                                    ) where {F,L}
        d = H.local_dim
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

    function surface_spectral_spin_density(H::TB_Hamiltonian{F,L},
                                        S,
                                        ω::Number,
                                        bx::Array{L,1},
                                        by::Array{L,1},
                                        bz::Array{L,1},
                                        n_super_layer::Integer,
                                        n_layer::Integer,
                                        n_kpts::Integer,
                                        kx_offset = -1/2,
                                        ky_offset = -1/2
                                        ) where {F,L}
        d = H.local_dim
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

    function projected_spectral_density(H::TB_Hamiltonian{F,L},
                                        ω::Number,
                                        bx::Array{T,1},
                                        by::Array{T,1},
                                        bz::Array{T,1},
                                        n_kpts::Integer,
                                        n_kpts_z::Integer,
                                        kx_offset = -1/2,
                                        ky_offset = -1/2
                                        ) where {F,L,T<:Real}
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)

        A = zeros(n_kpts,n_kpts)
        r = range(-1/2,1/2,n_kpts)
        for y = 1:n_kpts
            ky = (y-1)/(n_kpts-1) + ky_offset
            for x = 1:n_kpts
                kx = (x-1)/(n_kpts-1) + kx_offset
                for z = 1:n_kpts_z
                    kz = (z-1)/(n_kpts_z-1) - 1/2
                    k = kx * bx + ky * by + kz * bz
                    bloch_hamiltonian!(H,k,Hk)
                    for i = 1:d
                        Hk[i,i] -= ω 
                    end
                    G = inv(Hk)
                    A[x,y] += imag(tr(G)) 
                end
            end
        end
        return A
    end

    function surface_bands( H::TB_Hamiltonian{F,L},
                            k_path,
                            bz::Array{L,1},
                            n_layer::Integer,
                            ) where {F,L}
        d = H.local_dim
        n_kpts = length(k_path)
        bands = Array{Float64,2}(undef,d*n_layer,n_kpts)
        weights = Array{Float64,2}(undef,d*n_layer,n_kpts)
        Hs = zeros(ComplexF64,n_layer*d,n_layer*d)
        for i=1: n_kpts
            slab_hamiltonian!(H,k_path[i],bz,n_layer,Hs)
            E, T = LAPACK.syev!('V','U',Hs)
            bands[:,i] .= E
            for j = 1:d*n_layer
                weights[j,i] = norm(@view T[1:d,j])
            end
        end
        return bands, weights
    end

end