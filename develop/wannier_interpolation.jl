using TightBindingToolBox, LinearAlgebra, Plots

H = FPLO_import_TB("/home/brass/Projekte/LK99/TB_Ham_4spinbands/+hamdata_larger_mesh");

##
function wannier_interpolation(Hk,dim,n1,n2,n3)
    H = TB_Hamiltonian{ComplexF64,Int}(dim,3)
    N1 = (2n1+1)
    N2 = (2n2+1)
    N3 = (2n3+1)
    N = N1*N2*N3
    for r1 = -n1:n1, r2 = -n2:n2, r3 = -n3:n3
        R = [r1,r2,r3]
        t = zeros(ComplexF64,dim,dim)
        for k1 = -n1:n1, k2 = -n2:n2, k3 = -n3:n3
            k = [k1/N1,k2/N2,k3/N3]
            A = exp(-2π*im*(k⋅R))/N
            @. t += Hk[:,:,k1+1+n1,k2+1+n2,k3+1+n3] * A
        end
        TightBindingToolBox.Core.add_hoppings!(H,R,t)
    end
    return H
end

function wannier_interpolation2(Hk,dim,n1,n2,n3)
    H = TB_Hamiltonian{ComplexF64,Int}(dim,3)
    N1 = 2n1
    N2 = 2n2
    N3 = 2n3
    N = N1*N2*N3
    for r1 = -n1+1:n1, r2 = -n2+1:n2, r3 = -n3+1:n3
        R = [r1,r2,r3]
        t = zeros(ComplexF64,dim,dim)
        for k1 = -n1+1:n1, k2 = -n2+1:n2, k3 = -n3+1:n3
            k = [k1/N1,k2/N2,k3/N3]
            A = exp(-2π*im*(k⋅R))/N
            @. t += Hk[:,:,k1+n1,k2+n2,k3+n3] * A
        end
        TightBindingToolBox.Core.add_hoppings!(H,R,t)
    end
    return H
end

function wannier_interpolation(H::TB_Hamiltonian,lower_band,upper_band,n1,n2,n3)
    d = H.local_dim
    dim = upper_band-lower_band + 1
    N1 = (2n1+1)
    N2 = (2n2+1)
    N3 = (2n3+1)
    Hk = zeros(ComplexF64,dim,dim,N1,N2,N3)
    hk = zeros(ComplexF64,d,d)
    for k1 = -n1:n1, k2 = -n2:n2, k3 = -n3:n3
        k = [k1/N1,k2/N2,k3/N3]
        bloch_hamiltonian!(H,k,hk)
        E = LAPACK.syev!('N','U',hk)[lower_band:upper_band]
        Hk[:,:,k1+1+n1,k2+1+n2,k3+1+n3] = Matrix(Diagonal(E))
    end

    return wannier_interpolation(Hk,dim,n1,n2,n3);
end

##
dim = H.local_dim
n1 = 7; n2 = 7; n3 = 7;
N1 = (2n1+1)
N2 = (2n2+1)
N3 = (2n3+1)
Hk = Array{ComplexF64,5}(undef,dim,dim,N1,N2,N3)

for k1 = -n1:n1, k2 = -n2:n2, k3 = -n3:n3
    k = [k1/N1,k2/N2,k3/N3]
    Hk[:,:,k1+1+n1,k2+1+n2,k3+1+n3] = bloch_hamiltonian(H,k)
end

Hw = wannier_interpolation(Hk,dim,n1,n2,n3);
##
kpath = [[0,0,0], [1/2,-1/2,0],[2/3,-1/3,0],[0,0,0],[0,0,1/2],[1/2,-1/2,1/2],[2/3,-1/3,1/2],[0,0,1/2]]
label = [     "Γ",         "M",         "K",     "Γ",      "A",        "L",           "H",        "A"]
nkpts = 200

plt = plot_Bandstructure(H,kpath,nkpts,label;ylabel="energy (eV)",c=:darkred)
bands, lines, x = bandstructure(Hw,kpath,nkpts)
plot!(plt,x,bands,c=:blue,line = (:dash, 1))

##
hk = bloch_hamiltonian(Hw,[0.0,0.0,0.0]);LAPACK.syev!('N','U',hk)
#norm(hk-hk')
##
n1 = 11; n2 = 11; n3 = 11;
Hw = @time wannier_interpolation(H,2,3,n1,n2,n3);
##
plt = plot_Bandstructure(H,kpath,nkpts,label;ylabel="energy (eV)",c=:darkred)
bands, lines, x = bandstructure(Hw,kpath,nkpts)
plot!(plt,x,bands,c=:blue,line = (:dash, 1))