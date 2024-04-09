using TightBindingToolBox, LinearAlgebra, Plots

struct Lattice_Model{F}
    G::Set{SymOperation}
    wycks::Vector{Vector{Rational{Int64}}}
    reps::Vector{Dict{SymOperation, Matrix{F}}}
    dims::Vector{Int}
    H::TB_Hamiltonian{ComplexF64,Int}
end

function Lattice_Model(gen::Vector{SymOperation}, wycks::Vector{Vector{Rational{Int64}}}, reps::Vector{Dict{SymOperation, Matrix{F}}}) where {F}
    G = generate_group(gen)
    all_wycks = Vector{Vector{Rational{Int64}}}()
    all_reps = Vector{Dict{SymOperation, Matrix{F}}}() 
    dims = [0]
    for (i,w) in pairs(wycks)
        ws = collect(Set([mod.(v,1) for v in orbit(G,w)]))
        append!(all_wycks,ws)
        for j=1:length(ws)
            push!(all_reps,reps[i])
            push!(dims,dims[end]+size(first(values(reps[1])))[1])
        end
    end
    d = pop!(dims)
    n = length(wycks[1])
    H = TB_Hamiltonian{ComplexF64,Int}(d,n)
    return Lattice_Model(G,all_wycks,all_reps,dims,H)
end

struct hopping_term{T}
    t::Matrix{ComplexF64}
    w1::Vector{T}
    w2::Vector{T}
    R2::Vector{Int}
end

Base.adjoint(t::hopping_term) = hopping_term(Matrix(t.t'),t.w2,t.w1,-t.R2)

function fold_wyck_to_unit_cell(w,wycks)
    for v in wycks
        R = w - v
        if norm(mod.(R,1)) == 0
            return v, round.(Int,R)
        end
    end
    @error "fold_wyck_to_unit_cell failed for w=$w"
end

function transform(ht::hopping_term,g::SymOperation,r1,r2,wycks)
    t = r1 * ht.t * r2'
    w1, R1 = fold_wyck_to_unit_cell(g.R*ht.w1-g.T,wycks)
    w2, R2 = fold_wyck_to_unit_cell(g.R*ht.w2-g.T,wycks)
    return hopping_term(t,w1,w2,round.(Int64,g.R*ht.R2 + R2 - R1))
end

wyck_idx(w,wycks) = findfirst(x->x==w,wycks)

symmetrize(ht::hopping_term,ρ1,ρ2,wycks) = [transform(ht,g,r,ρ2[g],wycks) for (g,r) in pairs(ρ1)]

function add_hopping!(L::Lattice_Model,T::Matrix{ComplexF64},atom_pos1,atom_pos2,R)
    t0 = hopping_term(T,L.wycks[atom_pos1],L.wycks[atom_pos2],R)

    ρ1 = L.reps[atom_pos1]
    ρ2 = L.reps[atom_pos2]

    ts = symmetrize(t0,L.reps[atom_pos1],L.reps[atom_pos2],L.wycks)
    hops = Set([(h.w1,h.w2,h.R2) for h in ts])
    α = length(hops)/length(L.G)
    #for x in hops println(x) end
    for t in ts
        add_hopping!(L.H,t,L.wycks,L.dims,α)
        add_hopping!(L.H,t',L.wycks,L.dims,α)
    end

    #=
    for (g,r) in pairs(ρ1)
        t = transform(t0,g,r,ρ2[g],L.wycks)
        add_hopping!(L.H,t,L.wycks,L.dims)
        add_hopping!(L.H,t',L.wycks,L.dims)
    end =#

end

function add_hopping!(H::TB_Hamiltonian,ht::hopping_term,wycks,dims,α)
    d = H.local_dim
    (n1,n2) = size(ht.t)
    h = zeros(ComplexF64,d,d)
    i = wyck_idx(ht.w1,wycks)
    j = wyck_idx(ht.w2,wycks)
    h[dims[i]+1:dims[i]+n1,dims[j]+1:dims[j]+n2] .= ht.t * α
    add_hoppings!(H,ht.R2,h)
end

##
gen = [
    SymOperation([-1//1 0 0;0 -1 0;0 0 1],[0,0,0//1]),
    SymOperation([0 -1//1 0;1 0 0;0 0 1],[1//2,1//2,1//2]),
    SymOperation([-1//1 0 0;0 1 0;0 0 -1],[1//2,1//2,1//2]),
    SymOperation([-1//1 0 0;0 -1 0;0 0 -1],[0,0,0//1])
]

G = generate_group(gen)

χ = irreps(G)
Cls = conjugacy_classes(G)

idx_rep1 = 10
ρ = Dict([
    g => Matrix(Diagonal([χ[i,idx_rep1]])) # ,χ[i,idx_rep2]
    for (i,C) in pairs(collect(Cls)) for g in C
])

L = Lattice_Model(gen,[[0,0,0//1]],[ρ])

t = exp(2π*im/1)
add_hopping!(L,[t;;],1,2,[0,0,0])
α = exp(im*π)
β = exp(im*π/2)
add_hopping!(L,[-t/5;;],1,1,[1,0,0])
add_hopping!(L,[-t/5;;],1,1,[0,0,1])



#add_hopping!(L,[1.0+0.5im;;],L.wycks[1],L.wycks[2],[0,0,0])
#add_hopping!(L,[1.0+0.5im;;]/2,L.wycks[1],L.wycks[1],[1,0,0])
#add_hopping!(L,[1.0+0.5im;;]/3,L.wycks[1],L.wycks[1],[0,0,1])


path = [[0,0,0],[0,1/2,0],[1/2,1/2,0],[0,0,0],[0,0,1/2],[0,1/2,1/2],[1/2,1/2,1/2],[0,0,0],[0,0,1/2]]
labels = ["Γ","X","M","Γ","Z","R","A","Z"]
plot_Bandstructure(L.H,path,1000,labels;c=:darkred)