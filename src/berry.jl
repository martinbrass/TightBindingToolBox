using .TightBindingToolBox, LinearAlgebra, DifferentialEquations, Base.Threads

struct BerryCurvature{TBH<:TightBindingHamiltonian,I}
    H::TBH
    ∂Hk::Array{ComplexF64,3}
    curv::Array{Float64,2}
    band_idx::Tuple{I,I}
end

function BerryCurvature(H::TightBindingHamiltonian,band_idx::Tuple{I,I}) where {I<:Integer}
    n = number_of_bands(H)
    d = lattice_dim(H)
    ∂Hk = zeros(ComplexF64,n,n,d)
    curv = zeros(d,d)
    return BerryCurvature(H,∂Hk,curv,band_idx)
end

BerryCurvature(H::TightBindingHamiltonian,idx::Integer) = BerryCurvature(H,(idx,idx))

function (Ω::BerryCurvature)(k)
    derivatives!(Ω.H,k,Ω.∂Hk)
    Ek, Uk = LAPACK.syev!('V','U',Ω.H.Hk)
    fill!(Ω.curv,0.0)
    nbands = number_of_bands(Ω.H)
    dim = lattice_dim(Ω.H)
    for n = Ω.band_idx[1]:Ω.band_idx[2], m = 1:Ω.band_idx[1]-1
        nk = @view Uk[:,n]
        mk = @view Uk[:,m]
        ΔE_sqr = real(Ek[n]-Ek[m]) 
        ΔE_sqr *= ΔE_sqr
        for i = 1:dim-1
            dHi = @view Ω.∂Hk[:,:,i]
            for j = i+1:dim
                dHj = @view Ω.∂Hk[:,:,j]
                ω = imag((nk ⋅ (dHi*mk))*(mk ⋅ (dHj*nk))/ΔE_sqr)
                Ω.curv[i,j] -= ω
                Ω.curv[j,i] += ω
            end
        end
    end
    for n = Ω.band_idx[1]:Ω.band_idx[2], m = Ω.band_idx[2]+1:nbands
        nk = @view Uk[:,n]
        mk = @view Uk[:,m]
        ΔE_sqr = real(Ek[n]-Ek[m]) 
        ΔE_sqr *= ΔE_sqr
        for i = 1:dim-1
            dHi = @view Ω.∂Hk[:,:,i]
            for j = i+1:dim
                dHj = @view Ω.∂Hk[:,:,j]
                ω = imag((nk ⋅ (dHi*mk))*(mk ⋅ (dHj*nk))/ΔE_sqr)
                Ω.curv[i,j] -= ω
                Ω.curv[j,i] += ω
            end
        end
    end
    return Ω.curv
end

struct Plane
    orig::Vector{Float64}
    k1::Vector{Float64}
    k2::Vector{Float64}
end

(P::Plane)(r,s) = P.orig + r*P.k1 + s*P.k2

function flux(x,p::Tuple{BC,Plane}) where {BC <: BerryCurvature}
    Ω, P = p
    k = P(x...)
    curv = Ω(k)
    return dot(P.k1,curv*P.k2) / π
end

struct Sphere
    orig::Vector{Float64}
    r::Float64
end

(S::Sphere)(φ,θ) = S.orig + S.r*[cos(φ)*sin(θ), sin(φ) *sin(θ),cos(θ)]

function flux(x,p::Tuple{BC,Sphere}) where {BC <: BerryCurvature}
    φ,θ = x
    Ω, S = p
    k = S(φ,θ)
    curv = Ω(k)
    eφ = S.r*[-sin(φ)*sin(θ), cos(φ)*sin(θ), 0]
    eθ = S.r*[cos(φ)*cos(θ), sin(φ)*cos(θ), -sin(θ)]
    return dot(eθ,curv*eφ) / π    
end

function berry_curvature(H::TightBindingHamiltonian,k,band_idx)
    d = length(k)
    ∂Hk = [zero(H.Hk) for i = 1:d]
    Ω = zeros(Float64,d,d)
    berry_curvature!(H,k,band_idx,∂Hk,Ω)
    return Ω
end


function berry_curvature!(H::TightBindingHamiltonian,k,band_idx,∂Hk,Ω)
    n_bands = number_of_bands(H)
    d = length(k)
    derivatives!(H,k,∂Hk)
    Ek, Uk = LAPACK.syev!('V','U',H.Hk)
    nk = @view Uk[:,band_idx]
    fill!(Ω,0.0)
    for m = 1:band_idx-1
        mk = @view Uk[:,m]
        ΔE_sqr = real(Ek[band_idx]-Ek[m]) 
        ΔE_sqr *= ΔE_sqr
        for i = 1:d-1
            for j = i+1:d
                ω = imag((nk ⋅ (∂Hk[i]*mk))*(mk ⋅ (∂Hk[j]*nk))/ΔE_sqr)
                Ω[i,j] -= ω
                Ω[j,i] += ω
            end
        end
    end
    for m = band_idx+1:n_bands
        mk = @view Uk[:,m]
        ΔE_sqr = real(Ek[band_idx]-Ek[m]) 
        ΔE_sqr *= ΔE_sqr
        for i = 1:d-1
            for j = i+1:d
                ω = imag((nk ⋅ (∂Hk[i]*mk))*(mk ⋅ (∂Hk[j]*nk))/ΔE_sqr)
                Ω[i,j] -= ω
                Ω[j,i] += ω
            end
        end
    end
end

"""
band_idx = Int defining the band for which berry_curvature is calculated
k0 = origin of the sphere
r = radius of the sphere 
nφ, nθ = Int number of discretizations in the angles in spherical coordinates
"""
function berry_flux(H::TightBindingHamiltonian,band_idx::Integer, k0::Vector,r::Real,nφ::Integer,nθ::Integer)
    Φ = 0.0
    dφ = 2π/nφ
    dθ = π/nθ
    
    d = length(k0)
    ∂Hk = [zero(H.Hk) for i = 1:d]
    Ω = zeros(Float64,d,d)
    
    for iφ = 0:nφ-1, iθ = 0:nθ-1 
        φ = iφ * dφ
        θ = iθ * dθ
        er = r*[cos(φ)*sin(θ), sin(φ) *sin(θ),cos(θ)]
        
        berry_curvature!(H,k0+er,band_idx,∂Hk,Ω)

        eφ = r*[-sin(φ)*sin(θ), cos(φ)*sin(θ), 0]
        eθ = r*[cos(φ)*cos(θ), sin(φ)*cos(θ), -sin(θ)]
        Φ += 2* eφ ⋅ (Ω * eθ) * dφ * dθ
    end
    return -Φ/2π # the minus sign comes from accidentally using a left handed system dφ^dθ, but dθ^dϕ would be correct
end

"""
this function uses the berry curvature to calculate the flux through a plane centered at 
    k0 and spanned by k1,k2
"""
function berry_flux(H::TightBindingHamiltonian,band_idx::Integer,k0::Vector,k1::Vector,k2::Vector;N=100)
    d = length(k0)
    ∂Hk = [zero(H.Hk) for i = 1:d]
    Ω = zeros(Float64,d,d)

    h = 1/N
    Φ = 0.0
    for x in -1/2:h:1/2-h, y in -1/2:h:1/2-h
        k = k0 + x*k1 + y*k2
        berry_curvature!(H,k,band_idx,∂Hk,Ω)
        Φ += 2 * Ω[1,2]
    end
    return Φ*h^2 / 2π
end

"""
berry_force! is used for Weyl point detection, 
    it gives the Vectorfield corresponding to the berry-curvature
"""
function berry_force!(dk,k,p,x) # H,band_idx,k,Hk,∂Hk,Ω
    (H,idx,∂Hk,Ω,i) = p
    berry_curvature!(H,k,idx,∂Hk,Ω)
    dk[1] = Ω[2,3] * i
    dk[2] = Ω[3,1] * i
    dk[3] = Ω[1,2] * i
end

function evolve_to_weyl_point(H::TightBindingHamiltonian,band_idx,k0,χ;trange = (0.,1.0))
    d = length(k0)
    ∂Hk = [zero(H.Hk) for i = 1:d]
    Ω = zeros(Float64,d,d)

    p = (H,band_idx,∂Hk,Ω,χ)
    prob = ODEProblem(berry_force!,k0,trange,p)
    y = solve(prob,ImplicitEuler(autodiff=false);verbose=true)
    return y[:,end]
end

"""
Weyl point detectioon using the PRB XXX algorithm
idx_band::Int is the index of the band for which we search Weyl points
klist is a list of starting points in the Brillouin zone
"""
function search_weyl_points(H::TightBindingHamiltonian,band_idx,klist::Vector{T};atol=1E-4) where {T}
    n_bands = number_of_bands(H)
    wps = Vector{Vector{T}}(undef,nthreads())
    for i = 1:nthreads()
        wps[i] = Vector{T}()
    end
    for k0 in klist
        for χ in (-1,1)
            wp = evolve_to_weyl_point(H,band_idx,k0,χ)
            bloch_hamiltonian!(H,wp)
            E = LAPACK.syev!('N','U',H.Hk)
            if  ((band_idx > 1) && (E[band_idx]-E[band_idx-1] < atol)) || ((band_idx < n_bands) && (E[band_idx+1]-E[band_idx] < atol))
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