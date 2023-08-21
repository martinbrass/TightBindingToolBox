module Bandstructure
    export bandstructure, plot_Bandstructure, DOS, plot_DOS,
           density_matrix, plot_pDOS
    
    using ..TightBindingToolBox, LinearAlgebra, Plots

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
end