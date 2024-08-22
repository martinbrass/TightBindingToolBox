using LinearAlgebra, TightBindingToolBox, Plots, LaTeXStrings, WignerSymbols,Serialization, BenchmarkTools
filename = "/home/brass/Projekte/LK99/TB_Ham_4spinbands/+hamdata_larger_mesh"
H = FPLO_import_TB(filename);
n_max_hop = max(maximum(map(x->x[1],keys(H.hoppings))),
    maximum(map(x->x[2],keys(H.hoppings))),
    maximum(map(x->x[3],keys(H.hoppings)))
);

n_super_layer = 12*16   #*8
n_kpts  = 10
γ =  -0.02 +im*1E-4
bx = [0,1,0]
by = [0,0,1]
bz = [1,0,0]
#A = @time surface_spectral_density(H,γ,bx,by,bz,n_super_layer,n_max_hop,n_kpts,-1/2,0)';

n_layer = n_max_hop
d = H.local_dim
Hs = zeros(ComplexF64,2*n_layer*d,2*n_layer*d)
A = zeros(n_kpts,n_kpts)
Y = zeros(ComplexF64,n_layer*d,n_layer*d)
ky_offset = -1/2
kx_offset = -1/2
ω = γ

k = Vector{Float64}(undef,length(bx)) #
h0 = Matrix{ComplexF64}(undef,d*n_layer,d*n_layer) #
W = ω * Diagonal(ones(d*n_layer)) #
function foo()
    Gr = Matrix{ComplexF64}(undef,d*n_layer,d*n_layer) #
    for y = 1:n_kpts
        ky = (y-1)/(n_kpts-1) + ky_offset
        for x = 1:n_kpts
            kx = (x-1)/(n_kpts-1) + kx_offset
            @. k = kx * bx + ky * by #
            slab_hamiltonian!(H,k,bz,2*n_layer,Hs) 
            @. h0 = W - Hs[1:d*n_layer,1:d*n_layer] #
            T = @view Hs[1:d*n_layer,1+d*n_layer:2*d*n_layer];
            Td= T'
            Gr .= h0
            for i=1:n_super_layer-1
                LU = lu!(Gr)
                ldiv!(Y,LU,Td)
                mul!(Gr,T,Y)
                Gr .= h0 .- Gr
            end
            Gr = inv(Gr)
            A[x,y] -= imag(tr(@view Gr[1:d,1:d])) 
        end
    end
end
