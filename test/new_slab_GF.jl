using LinearAlgebra, TightBindingToolBox, Plots, LaTeXStrings, WignerSymbols, CSV, DataFrames
filename = "/home/brass/Projekte/LK99/TB_Ham_2bands/+hamdata"
H = FPLO_import_TB(filename);
##
n_max_hop = max(maximum(map(x->x[1],keys(H.hoppings))),
    maximum(map(x->x[2],keys(H.hoppings))),
    maximum(map(x->x[3],keys(H.hoppings)))
);

d = H.local_dim;

n_layer = 2*n_max_hop

Hs = zeros(ComplexF64,n_layer*d,n_layer*d)
k = [0.12,0.34,0.56]
bz = [1,0,0]
slab_hamiltonian!(H,k,bz,n_layer,Hs)
##
H00 = Hs[1:d*n_max_hop,1:d*n_max_hop]
H01 = Hs[1:d*n_max_hop,1+d*n_max_hop:2*d*n_max_hop];

##

ld = d*n_max_hop
n_super_layer = 10

Hs2 = zeros(ComplexF64,n_super_layer*ld,n_super_layer*ld)
slab_hamiltonian!(H,k,bz,n_super_layer*n_max_hop,Hs2)
M = zeros(ComplexF64,n_super_layer*ld,n_super_layer*ld)

for j=0:n_super_layer-2
    M[1+ld*j:ld*(j+1),1+ld*j:ld*(j+1)] = H00
    M[1+ld*j:ld*(j+1),1+ld*(j+1):ld*(j+2)] = H01
    M[1+ld*(j+1):ld*(j+2),1+ld*j:ld*(j+1)] = H01'
end
j=n_super_layer-1
M[1+ld*j:ld*(j+1),1+ld*j:ld*(j+1)] = H00

norm(Hs2-M)

##

ω = 1E-3*im
Gs = inv(I*ω-Hs2)
h0 = I*ω-H00
Gr = inv(h0)
T = H01
Td= H01'
Y = similar(Gr)
for i=1:n_super_layer-1
    mul!(Y,Gr,Td)
    mul!(Gr,T,Y)
    Gr .= h0 .- Gr
    Gr = inv(Gr)
end

norm(Gs[1:ld,1:ld]-Gr)


##
n = 10000
x = zeros(n,n)
y = rand(0.0:1,n,n)
z = rand(0.0:1,n,n);
##



@benchmark (x .= y .-z)