using TightBindingToolBox, LinearAlgebra, BenchmarkTools

function berry_curvature!(H::TB_Hamiltonian,k,band_idx,Hk,∂Hk,Ω)
    TightBindingToolBox.Berry.get_Hk_∂Hk!(H,k,Hk,∂Hk)
    Ek, Uk = LAPACK.syev!('V','U',Hk)
    Ω .*= 0

    for n = 1:band_idx, m = band_idx+1:H.local_dim
        nk = @view Uk[:,n]
        mk = @view Uk[:,m]
        ΔE_sqr = real(Ek[n]-Ek[m]) 
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

function berry_flux_through_sphere(H::TB_Hamiltonian,idx_band, k0,r,nφ,nθ)
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
        berry_curvature!(H,k0+er,idx_band,Hk,∂Hk,Ω)

        eφ = r*[-sin(φ)*sin(θ), cos(φ)*sin(θ), 0]
        eθ = r*[cos(φ)*cos(θ), sin(φ)*cos(θ), -sin(θ)]
        Φ += 2* eφ ⋅ (Ω * eθ) * dφ * dθ
    end
    return -Φ # the minus sign comes from accidentally using a left handed system dφ^dθ, but dθ^dϕ would be correct
end

dim = Hz.local_dim
d = Hz.lattice_dim
Hk = zeros(ComplexF64,dim,dim)
∂Hk = map(x->copy(Hk),collect(1:d))
Ω = zeros(Float64,d,d)

k = [0.1,0.32,0.456]
@benchmark berry_curvature!(Hz,k,114,Hk,∂Hk,Ω)