using LinearAlgebra, TightBindingToolBox, Plots, LaTeXStrings, Cubature

filename = "/home/brass/Projekte/LK99/TB_Ham_4bands/+hamdata"
H = FPLO_import_TB(filename);
idx_band=1;k0=[0.,0,0];r=1E-1;
@time integrate_berry_curvature_sphere(H,idx_band,k0,r,10*10,10*10)/2π

##
function berry_curv(φ,θ,r,H,idx_band,k0,Hk,∂Hk,Ω)
    er = r*[cos(φ)*sin(θ), sin(φ) *sin(θ),cos(θ)]
    TightBindingToolBox.Berry.get_berry_curvature!(H,k0+er,idx_band,Hk,∂Hk,Ω)

    eφ = r*[-sin(φ)*sin(θ), cos(φ)*sin(θ), 0]
    eθ = r*[cos(φ)*cos(θ), sin(φ)*cos(θ), -sin(θ)]
    return -2* eφ ⋅ (Ω * eθ) 
end

dim = H.local_dim
d = H.lattice_dim
Hk = zeros(ComplexF64,dim,dim)
∂Hk = map(x->copy(Hk),collect(1:d))
Ω = zeros(Float64,d,d)

@time pcubature(x->berry_curv(x[1],x[2],r,H,idx_band,k0,Hk,∂Hk,Ω),[0.,0],[2π,π],abstol=1E-6) ./2π