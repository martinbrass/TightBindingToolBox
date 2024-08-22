using TightBindingToolBox, LinearAlgebra, Plots

gen = [SymOperation([0 -1//1 0;1 -1 0;0 0 1],[0,0,0//1])]

ρ = Dict([gen[1] => Matrix(Diagonal([exp(2π*im/3),exp(-2π*im/3)])),
          gen[1]*gen[1] => Matrix(Diagonal([exp(2π*im/3*2),exp(-2π*im/3*2)])),
          gen[1]*gen[1]*gen[1] => Matrix(Diagonal([1.0+0im,1]))])
L = Lattice_Model(gen,[[0,0,0//1]],[ρ])

t = 1.0+0.5im
add_hopping!(L,-[t 0; 0 conj(t)],1,1,[0,0,1])
t2 = 0.5+0.0im
add_hopping!(L,-t2*[0.0 1; 0 0.0],1,1,[1,0,0])

t3 = 1/4+1.0im
add_hopping!(L,-[t3 0; 0 conj(t3)],1,1,[1,0,0])

##
kpath = [[0,0,0], [1/2,-1/2,0],[2/3,-1/3,0],[0,0,0],[0,0,1/2],[1/2,-1/2,1/2],[2/3,-1/3,1/2],[0,0,1/2]]
label = [     "Γ",         "M",         "K",     "Γ",      "A",        "L",           "H",        "A"]
nkpts = 2000

plt = plot_Bandstructure(L.H,kpath,nkpts,label;ylabel="energy (eV)",c=:darkred)
##
bloch_hamiltonian(L.H,[0,0,0.23675])
##
τ = [0 -1; -1 0]
for t in values(L.H.hoppings)
    t .= (t + τ*conj.(t)*τ)/2
end