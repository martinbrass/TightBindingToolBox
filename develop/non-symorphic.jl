using TightBindingToolBox, LinearAlgebra, Plots

function transform(ht::hopping_term,g::SymOperation,r1,r2,wycks)
    t = r1 * ht.t * r2'
    w1, R1 = fold_wyck_to_unit_cell(g.R*ht.w1-g.T,wycks)
    w2, R2 = fold_wyck_to_unit_cell(g.R*ht.w2-g.T,wycks)
    return hopping_term(t,w1,w2,round.(Int64,g.R*ht.R2 + R2 - R1))
end

G = generate_group([
    SymOperation([-1//1 0 0;0 -1 0;0 0 1],[0,0,0//1]),
    SymOperation([0 -1//1 0;1 0 0;0 0 1],[1//2,1//2,1//2]),
    SymOperation([-1//1 0 0;0 1 0;0 0 -1],[1//2,1//2,1//2]),
    SymOperation([-1//1 0 0;0 -1 0;0 0 -1],[0,0,0//1])
])

wycks = collect(Set([mod.(w,1) for w in orbit(G,[0//1,0,0])]))

χ = irreps(G)
Cls = conjugacy_classes(G)
ω = Diagonal([length(C) for C in Cls]) / length(G)
A1g = [1.0 for C in Cls]

idx_rep1 = 10
ρ = Dict([
    g => Matrix(Diagonal([χ[i,idx_rep1]])) # ,χ[i,idx_rep2]
    for (i,C) in pairs(collect(Cls)) for g in C
])

t0 = hopping_term([1.0+0.5im;;],wycks[1],wycks[2],[0,0,0])

ts = symmetrize(t0,ρ,ρ,wycks)

H = TB_Hamiltonian{ComplexF64,Int}(2,3)
dims = [0,1]
for t in ts
    add_hopping!(H,t,wycks,dims)
    add_hopping!(H,t',wycks,dims)
end

t0 = hopping_term([1.0+0.5im;;]/2,wycks[1],wycks[1],[1,0,0])
ts = symmetrize(t0,ρ,ρ,wycks)
dims = [0,1]
for t in ts
    add_hopping!(H,t,wycks,dims)
    add_hopping!(H,t',wycks,dims)
end

t0 = hopping_term([1.0+0.5im;;]/3,wycks[1],wycks[1],[0,0,1])
ts = symmetrize(t0,ρ,ρ,wycks)
dims = [0,1]
for t in ts
    add_hopping!(H,t,wycks,dims)
    add_hopping!(H,t',wycks,dims)
end

##
path = [[0,0,0],[0,1/2,0],[1/2,1/2,0],[1/2,1/2,1/2],[0,0,0],[0,1/2,1/2],[0,0,1/2]]
labels = ["Γ","X","M","A","Γ","R","Z"]
plot_Bandstructure(H,path,1000,labels;c=:darkred)
##
gs = collect(G);
gs[2].R*gs[2].T+gs[2].T

##
for X in keys(ρ)
   w1,R1 = fold_wyck_to_unit_cell(X*wycks[1],wycks)
   w2,R2 = fold_wyck_to_unit_cell(X*wycks[2],wycks)
   #println(R2-R1)
   println((X* [0,0,0] +R2-R1))
end