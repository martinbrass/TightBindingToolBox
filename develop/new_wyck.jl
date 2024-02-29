using LinearAlgebra, Plots, TightBindingToolBox

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

function transform(ht::hopping_term,g,r1,r2,wycks)
    t = r1 * ht.t * r2'
    w1, R1 = fold_wyck_to_unit_cell(g*ht.w1,wycks)
    w2, R2 = fold_wyck_to_unit_cell(g*ht.w2,wycks)
    return hopping_term(t,w1,w2,round.(Int64,g*ht.R2 + R2 - R1))
end

wyck_idx(w,wycks) = findfirst(x->x==w,wycks)

symmetrize(ht::hopping_term,ρ1,ρ2,wycks) = [transform(ht,g,r,ρ2[g],wycks) for (g,r) in pairs(ρ1)]

function add_hopping!(H::TB_Hamiltonian,ht::hopping_term,wycks,dims)
    d = H.local_dim
    (n1,n2) = size(ht.t)
    h = zeros(ComplexF64,d,d)
    i = wyck_idx(ht.w1,wycks)
    j = wyck_idx(ht.w2,wycks)
    h[dims[i]+1:dims[i]+n1,dims[j]+1:dims[j]+n2] .= ht.t
    add_hoppings!(H,ht.R2,h)
end
##

D6h = generate_group([
    [0 -1 0; 1 -1 0; 0 0 1],
    [-1 0 0; 0 -1 0; 0 0 1],
    [-1 0 0; 0 -1 0; 0 0 -1],
    [0 1 0; 1 0 0; 0 0 1]
])

wycks = collect(Set([mod.(w,1) for w in orbit(D6h,[2//3,1//3,0])]))

χ = irreps(D6h)
Cls = conjugacy_classes(D6h)
ω = Diagonal([length(C) for C in Cls]) / length(D6h)
A1g = [1.0 for C in Cls]

idx_rep1 = 12
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

##
path = [[0,0,0],[1/2,0,0],[1/3,1/3,0],[0,0,0]]
labels = ["Γ","M","K","Γ"]
plot_Bandstructure(H,path,1000,labels;c=:darkred)

