module Berry
    export discretize_BZ, assign_fibre, scan_BZ_for_Weyl_points, check_wp_candidates,
           integrate_berry_curvature_sphere, refine_wp,print_Berry_curvature, 
           berry_flux_through_plane, integrate_connection_along_path,berry_force!,
           search_weyl_points, plot_curvature, integrate_berry_curvature_donut,
           spin_texture

    using ..TightBindingToolBox
    using LinearAlgebra, Base.Threads, CSV, DataFrames, DifferentialEquations
    
    function discretize_BZ(N;origin=[0.,0,0])
        n = N+1
        BZ = Array{Vector{Float64},3}(undef,n,n,n)
        for x = 0:N, y = 0:N, z = 0:N
            BZ[x+1,y+1,z+1] = 1/N * [x,y,z] + origin
        end
        return BZ
    end

    function assign_fibre(BZ,H::TB_Hamiltonian,idx_band)
        d = H.local_dim
        nx,ny,nz = size(BZ)
        F = Array{Vector{ComplexF64},3}(undef,nx,ny,nz)
        #Hk = map(x->zeros(ComplexF64,d,d),1:nthreads())
        Hk = zeros(ComplexF64,d,d)
        for x = 1:nx, y = 1:ny, z = 1:nz
                bloch_hamiltonian!(H,BZ[x,y,z],Hk)
                E, T = LAPACK.syev!('V','U',Hk) # LAPACK.syevr!('V','I','U',H.Hk,-1.0,1.0,idx,idx,1E-6)
                F[x,y,z] = T[:,idx_band]

        end
        return F
    end

    """
    scan the Brillouin-zone for Weyl-points by calculating Chern numbers in boxes 
        and refining those boxes
    """
    function scan_BZ_for_Weyl_points(H::TB_Hamiltonian,BZ,idx_band;thresh=0.99)
        cells = Vector{Float64}[]
        nx,ny,nz = size(BZ)
        ψk = assign_fibre(BZ,H,idx_band)
        for x = 1:nx-1, y = 1:ny-1, z = 1:nz-1
                cern_number = berry_flux_cube(ψk[x:x+1,y:y+1,z:z+1])/2π
                if abs(cern_number) > thresh
                    push!(cells,BZ[x,y,z])
                end
        end
        return cells
    end

    function berry_flux_plaquette(ψ1,ψ2,ψ3,ψ4)
        prod = (ψ1⋅ψ2) * (ψ2⋅ψ3) * (ψ3⋅ψ4) * (ψ4⋅ψ1)
        return -angle(prod)
    end
    
    function berry_flux_cube(ψk)
        #faces perp to z
        Φ  = berry_flux_plaquette(ψk[1,1,1],ψk[2,1,1],ψk[2,2,1],ψk[1,2,1])
        Φ += berry_flux_plaquette(ψk[1,1,2],ψk[1,2,2],ψk[2,2,2],ψk[2,1,2])
        #faces perp to y
        Φ += berry_flux_plaquette(ψk[1,1,1],ψk[1,1,2],ψk[2,1,2],ψk[2,1,1])
        Φ += berry_flux_plaquette(ψk[1,2,1],ψk[2,2,1],ψk[2,2,2],ψk[1,2,2])
        #faces perp to x
        Φ += berry_flux_plaquette(ψk[1,1,1],ψk[1,2,1],ψk[1,2,2],ψk[1,1,2])
        Φ += berry_flux_plaquette(ψk[2,1,1],ψk[2,1,2],ψk[2,2,2],ψk[2,2,1])
        return Φ
    end  
        
    function get_Hk_∂Hk!(H::TB_Hamiltonian,k,Hk,∂Hk)
        Hk  .*= 0
        for m=1:H.lattice_dim 
            ∂Hk[m] .*= 0 
        end
        for (R,t) in pairs(H.hoppings)
            Hk .+= t .* exp(2π*im * (k⋅R)) 
            for m = 1:H.lattice_dim
                ∂Hk[m] .+= t .* (exp(2π*im * (k⋅R)) * im *R[m]) .*2π
            end
        end
    end

    function get_berry_curvature!(H::TB_Hamiltonian,k,band_idx,Hk,∂Hk,Ω)
        get_Hk_∂Hk!(H,k,Hk,∂Hk)
        Ek, Uk = LAPACK.syev!('V','U',Hk)
        nk = @view Uk[:,band_idx]
        Ω .*= 0
        for m = 1:band_idx-1
            mk = @view Uk[:,m]
            ΔE_sqr = real(Ek[band_idx]-Ek[m]) 
            ΔE_sqr *= ΔE_sqr
            for i = 1:H.lattice_dim-1
                for j = i+1:H.lattice_dim
                    ω = imag((nk ⋅ (∂Hk[i]*mk))*(mk ⋅ (∂Hk[j]*nk))/ΔE_sqr)
                    Ω[i,j] -= ω
                    Ω[j,i] += ω
                end
            end
        end
        for m = band_idx+1:H.local_dim
            mk = @view Uk[:,m]
            ΔE_sqr = real(Ek[band_idx]-Ek[m]) 
            ΔE_sqr *= ΔE_sqr
            for i = 1:H.lattice_dim-1
                for j = i+1:H.lattice_dim
                    ω = imag((nk ⋅ (∂Hk[i]*mk))*(mk ⋅ (∂Hk[j]*nk))/ΔE_sqr)
                    Ω[i,j] -= ω
                    Ω[j,i] += ω
                end
            end
        end
    end

    function check_wp_candidates(cells,n_divisions,H,band_idx,n_quad_pts=20,tol=0.1)
        d = 1/n_divisions
        r = sqrt(3)/2 * d
        wps = Vector{Float64}[]
        chern_nums = Float64[]
        for i = 1:size(cells)[1]
            wp = cells[i] .+ d/2
            chern = integrate_berry_curvature_sphere(H,band_idx,wp,r,n_quad_pts,n_quad_pts)/2π
            if abs(chern) > tol
                push!(wps,wp)
                push!(chern_nums,chern)
                println("i = ",i,"\t chern number = ",chern)
            end
        end
        return wps, chern_nums
    end

    """
    idx_band = Int defining the band for which berry_curvature is calculated
    k0 = origin of the sphere
    r = radius of the sphere 
    nφ, nθ = Int number of discretizations in the angles in spherical coordinates
    """
    function integrate_berry_curvature_sphere(H::TB_Hamiltonian,idx_band, k0,r,nφ,nθ)
        Φ = 0
        dφ = 2π/nφ
        dθ = π/nθ
    
        dim = H.local_dim
        d = H.lattice_dim
        Hk = zeros(ComplexF64,dim,dim)
        ∂Hk = map(x->copy(Hk),collect(1:d))
        Ω = zeros(Float64,d,d)
        
        for iφ = 0:nφ-1, iθ = 0:nθ-1 # TODO: remove -1???
            φ = iφ * dφ
            θ = iθ * dθ
            er = r*[cos(φ)*sin(θ), sin(φ) *sin(θ),cos(θ)]
            get_berry_curvature!(H,k0+er,idx_band,Hk,∂Hk,Ω)
    
            eφ = r*[-sin(φ)*sin(θ), cos(φ)*sin(θ), 0]
            eθ = r*[cos(φ)*cos(θ), sin(φ)*cos(θ), -sin(θ)]
            Φ += 2* eφ ⋅ (Ω * eθ) * dφ * dθ
        end
        return -Φ # the minus sign comes from accidentally using a left handed system dφ^dθ, but dθ^dϕ would be correct
    end

    function integrate_berry_curvature_donut(H::TB_Hamiltonian,idx_band, b1,b2,b3,r,nφ,nz)
        Φ = 0
        dφ = 2π/nφ
        dz = 1/nz
    
        dim = H.local_dim
        d = H.lattice_dim
        Hk = zeros(ComplexF64,dim,dim)
        ∂Hk = map(x->copy(Hk),collect(1:d))
        Ω = zeros(Float64,d,d)
        
        for iφ = 0:nφ-1, iz = 0:nz-1 #TODO remove -1 ???
            φ = iφ * dφ
            z = iz * dz
            k = r*(cos(φ)*b1 + sin(φ)*b2) + z*b3
            get_berry_curvature!(H,k,idx_band,Hk,∂Hk,Ω)
    
            eφ = r*(-sin(φ)*b1 + cos(φ)*b2)

            Φ += 2* eφ ⋅ (Ω * b3) * dφ * dz
        end
        return Φ
    end

    function refine_wp(H::TB_Hamiltonian,idx_band,wp,r,tol;thresh=0.9,nφ=20,nθ=20)
        wp_refined = copy(wp)
        r_refined = r
        while r_refined > tol
            wp_refined, r_refined = refine_wp_step(H,idx_band,wp_refined,r_refined;thresh,nφ,nθ)
        end
        return wp_refined, r_refined
    end

    function refine_wp_step(H::TB_Hamiltonian,band_idx,wp,r;thresh=0.9,nφ=20,nθ=20)
        println("refine at : ",r)
        koords = [-1/2,1/2]
        for x in koords, y in koords, z in koords
            k0 = wp + r*[x,y,z]
            kpts = init_kpts(k0 .-r/2,r,r,r)
            ϕ = init_ψk(H,band_idx,kpts)
            cern = berry_flux_cube(ϕ)/2π
            if abs(cern) > thresh
                return k0, r/2
            end
        end
        k0 = copy(wp)
        cern = integrate_berry_curvature_sphere(H,band_idx,k0,r/2,nφ,nθ)/2π
        if abs(cern) > thresh
            return k0, r/2
        end
        for x in koords, y in koords, z in koords
            k0 = wp + r*[x,y,z]
            cern = integrate_berry_curvature_sphere(H,band_idx,k0,sqrt(3)*r/2,nφ,nθ)/2π
            println(cern,'\t',[x,y,z])
            if abs(cern) > thresh
                return k0, sqrt(3)*r/2
            end
        end
        @warn "refine_wp: all sub-divisions returned vanishing chern number."
        return nothing, 0.0
    end
    
    function init_ψk(H,band_idx,k_pts)
        ψk = Array{Vector{ComplexF64},3}(undef,2,2,2)
        for x=1:2, y=1:2, z=1:2
            k = k_pts[x,y,z]
            Hk = bloch_hamiltonian(H,k)
            val,vec = LAPACK.syev!('V','U',Hk)
            ψk[x,y,z] = vec[:,band_idx]
        end
        return ψk
    end
    
    function init_kpts(k0,a,b,c)
        k_pts = Array{Vector{Float64},3}(undef,2,2,2)
        e1 = a*[1,0,0]
        e2 = b*[0,1,0]
        e3 = c*[0,0,1]
        k_pts[1,1,1] = k0
        k_pts[2,1,1] = k0 + e1
        k_pts[1,2,1] = k0 + e2
        k_pts[1,1,2] = k0 + e3
        k_pts[2,2,1] = k0 + e1 + e2
        k_pts[2,1,2] = k0 + e1 + e3
        k_pts[1,2,2] = k0 + e3 + e2
        k_pts[2,2,2] = k0 + e1 + e2 + e3
        return k_pts
    end

    function spin_texture(H::TB_Hamiltonian,idx_band,k_surface,S)
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)
        n = size(k_surface,1)
        dat = zeros(Float64,n,length(S))
        for i = 1:n
            k = k_surface[i,:]
            Hk = bloch_hamiltonian(H,k)
            E, T = LAPACK.syev!('V','U',Hk)
            ψ = @view T[:,idx_band]
            map!(σ -> real(ψ ⋅ (σ*ψ)),(@view dat[i,:]),S)
        end
        return dat
    end

    function print_Berry_curvature(H::TB_Hamiltonian,idx_band, k0,r,nφ,nθ,filename)
        dφ = 2π/nφ
        dθ = π/nθ
        n = H.local_dim
        d = H.lattice_dim
        Hk = zeros(ComplexF64,n,n)
        ∂Hk = map(x->copy(Hk),collect(1:d))
        Ω = zeros(Float64,d,d)
        dat = zeros(Float64,nφ*nθ,2*d)
        norm = 0.0
        
        for iφ = 0:nφ-1, iθ = 0:nθ-1
            φ = iφ * dφ
            θ = iθ * dθ
            er = r*[cos(φ)*sin(θ), sin(φ) *sin(θ),cos(θ)]
            dat[iφ*nθ + iθ + 1,1] = er[1]
            dat[iφ*nθ + iθ + 1,2] = er[2]
            dat[iφ*nθ + iθ + 1,3] = er[3]
            get_berry_curvature!(H,k0+er,idx_band,Hk,∂Hk,Ω)
            dat[iφ*nθ + iθ + 1,4] = Ω[2,3]
            dat[iφ*nθ + iθ + 1,5] = Ω[3,1]
            dat[iφ*nθ + iθ + 1,6] = Ω[1,2]
            norm = max(norm,sqrt(Ω[1,2]^2+Ω[2,3]^2+Ω[3,1]^2))
        end
        dat[:,4:6] .*= r/2/norm
        CSV.write(filename,DataFrame(dat,:auto),writeheader=false, delim=' ')
    end
    """
    this function uses the berry curvature to calculate the flux through a plane centered at 
        k0 and spanned by k1,k2
    """
    function berry_flux_through_plane(H,idx_band,k0,k1,k2;N=100)
        Ω = zeros(3,3)
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)
        ∂Hk = [zeros(ComplexF64,d,d),zeros(ComplexF64,d,d),zeros(ComplexF64,d,d)]
    
        h = 1/N
        Φ = 0.0
        for x in -1/2:h:1/2-h, y in -1/2:h:1/2-h
            k = k0 + x*k1 + y*k2
            get_Hk_∂Hk!(H,k,Hk,∂Hk)
            get_berry_curvature!(H,k,idx_band,Hk,∂Hk,Ω)
            Φ += 2 * Ω[1,2]
        end
        return Φ*h^2 / 2π
    end

    function integrate_connection_along_path(H,kpath,idx_band)
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)
        ψ1 = zeros(ComplexF64,d)
        ψ2 = zeros(ComplexF64,d)
        bloch_hamiltonian!(H,kpath[1],Hk)
        ψ1 .= (LAPACK.syev!('V','U',Hk)[2])[:,idx_band]
        prod = 1.0+0im
        for i = 2:length(kpath)
            bloch_hamiltonian!(H,kpath[i],Hk)
            ψ2 .= (LAPACK.syev!('V','U',Hk)[2])[:,idx_band]
            prod *= (ψ1⋅ψ2)
            t = ψ1
            ψ1 = ψ2
            ψ2 = t
        end
        return -angle(prod)
    end

    """
    berry_force! is used for Weyl point detection, 
        it gives the Vectorfield corresponding to the berry-curvature
    """
    function berry_force!(dk,k,p,x) # H,band_idx,k,Hk,∂Hk,Ω
        (Ham,idx,Hk,∂Hk,Ω,i) = p
        #get_Hk_∂Hk!(Ham,k,Hk,∂Hk) this should be irrelevant
        get_berry_curvature!(Ham,k,idx,Hk,∂Hk,Ω)
        dk[1] = Ω[2,3] * i
        dk[2] = Ω[3,1] * i
        dk[3] = Ω[1,2] * i
    end

    function evolve_to_weyl_point(H::TB_Hamiltonian,idx_band,k0,χ;trange = (0.,1.0))
        Ω = zeros(3,3)
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)
        ∂Hk = [zeros(ComplexF64,d,d),zeros(ComplexF64,d,d),zeros(ComplexF64,d,d)]
        p = (H,idx_band,Hk,∂Hk,Ω,χ)
        prob = ODEProblem(berry_force!,k0,trange,p)
        y = solve(prob,ImplicitEuler(autodiff=false);verbose=true)
        return y[:,end]
    end

    """
    Weyl point detectioon using the PRB XXX algorithm
    idx_band::Int is the index of the band for which we search Weyl points
    klist is a list of starting points in the Brillouin zone
    """
    function search_weyl_points(H::TB_Hamiltonian,idx_band,klist::Vector{T};atol=1E-4) where {T}
        d = H.local_dim
        wps = Vector{Vector{T}}(undef,nthreads())
        for i = 1:nthreads()
            wps[i] = Vector{T}()
        end
        Hk = zeros(ComplexF64,d,d)
        for k0 in klist
            for χ in (-1,1)
                wp = evolve_to_weyl_point(H,idx_band,k0,χ)
                bloch_hamiltonian!(H,wp,Hk)
                E = LAPACK.syev!('N','U',Hk)
                if  ((idx_band > 1) && (E[idx_band]-E[idx_band-1] < atol)) || ((idx_band < d) && (E[idx_band+1]-E[idx_band] < atol))
                    push!(wps[threadid()],wp)
                end
            end
        end
        wps_out = Vector{T}()
        for i = 1:nthreads()
            for w in wps[i]
                push!(wps_out,w)
            end
        end
        return wps_out
    end

    """
    visulaize the Berry curvature on a plane centered on k0 and spanned by k1,k2 (all vectors of momentum)
        N = number of points per direction for plotting
        returned is the Berry curvature Ω (a 3x3 Matrix for every point in the plane, i.e a 3x3xNxN array)
    """
    function plot_curvature(H,idx_band,k0,k1,k2;N=100)
        Ω = zeros(3,3,N,N)
        d = H.local_dim
        Hk = zeros(ComplexF64,d,d)
        ∂Hk = [zeros(ComplexF64,d,d),zeros(ComplexF64,d,d),zeros(ComplexF64,d,d)]
    
        r = range(-1/2,1/2,N)
        for (i,x) in pairs(r), (j,y) in pairs(r)
            k = k0 + x*k1 + y*k2
            get_Hk_∂Hk!(H,k,Hk,∂Hk)
            get_berry_curvature!(H,k,idx_band,Hk,∂Hk,@view Ω[:,:,i,j])
        end
        h = 1/N
        return Ω .*(h^2 / 2π)
    end
end