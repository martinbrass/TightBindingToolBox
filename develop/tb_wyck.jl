using LinearAlgebra, Plots, TightBindingToolBox, Rotations
D6h = generate_group([
    [0 -1 0; 1 -1 0; 0 0 1],
    [-1 0 0; 0 -1 0; 0 0 1],
    [-1 0 0; 0 -1 0; 0 0 -1],
    [0 1 0; 1 0 0; 0 0 1]
])


##
Set([mod.(x,1) for x in orbit(D6h,[2//3,1//3,0])])

[]

C3 = RotZ(2π/3)[1:2,1:2] 

inv([0 -sqrt(3)/2; 1 -1/2])*[-sqrt(3)*(1/3+1/2),1/2]

C3*[1,0,0]

B=[1 -1/2;0 sqrt(3)/2]

inv(B)*C3*B

inv(B)*[1/2,1/2/sqrt(3)]
##
w1 = [2//3,1//3,0]

ws = orbit(D6h,w1)

orbit(D6h,collect(Set([mod.(a-b,1) for a in ws, b in ws]))[1])

collect(Set([(a-b) for a in ws, b in ws]))

ws0 = collect(Set([mod.(w,1) for w in ws]))



[reduce_wyck_to_unit_cell(w,ws0) for w in ws]

##
function reduce_wyck_to_unit_cell(w,ws0)
    for (i,v) in pairs(ws0)
        R = w - v
        if norm(mod.(R,1)) == 0
            return i, R
        end
    end
    @error "reduce_wyck_to_unit_cell failed for w=$w"
end

function hopping_orbit(G,wycks,m,n,R)
    orb = Set([(m,n,R)])
    w1 = wycks[m]
    w2 = wycks[n]
    for g in G
        i, Ri = reduce_wyck_to_unit_cell(g*w1,wycks)
        j, Rj = reduce_wyck_to_unit_cell(g*w2,wycks)
        push!(orb,(i,j,g*R + Rj - Ri))
    end
    return orb
end

function hopping_rep(G,wycks,m,n,R,ρmn)
    orb = collect(hopping_orbit(G,wycks,m,n,R))
    d = length(orb)
    #ρmn = kron(ρm,ρn')
    ρ = Dict([
                g => kron(ρmn[g],[ 
                    begin
                        m,n,R = orb[j];
                        w1 = wycks[m];
                        w2 = wycks[n];
                        k, Rk = reduce_wyck_to_unit_cell(g*w1,wycks);
                        l, Rl = reduce_wyck_to_unit_cell(g*w2,wycks);
                        res = (k,l,g*R + Rl - Rk);
                        i == findfirst(x->x==res,orb) ? 1.0 : 0.0 
                    end for i=1:d, j=1:d])
                for g in G])
    return ρ
end
##
idx_rep1 = 12
idx_rep2 = 12
ρψ = Dict([
    g => Matrix(Diagonal([χ[i,idx_rep1],χ[i,idx_rep2]])) # ,χ[i,idx_rep2]
    for (i,C) in pairs(collect(Cls)) for g in C
])

wycks = collect(Set([mod.(w,1) for w in orbit(D6h,[2//3,1//3,0])]))

ρmn=Dict([g=>kron(r,r') for (g,r) in pairs(ρψ)])

ρ = hopping_rep(D6h,wycks,1,2,[0,0,0],ρmn)

orb = collect(hopping_orbit(D6h,wycks,1,2,[0,0,0]))

χ = irreps(D6h)
Cls = conjugacy_classes(D6h)
ω = Diagonal([length(C) for C in Cls]) / length(D6h)
χh = characters(ρ,Cls)

A1g = [1.0 for C in Cls]
round.(A1g ⋅ (ω*χh);digits=3)

##
V=TightBindingToolBox.Model.get_basis_for_irrep(A1g,ρ,D6h)



function add_terms!(H,V,hoppings,param,n_orb,n_wyck)
    N = length(hoppings)
    for (a,B) in pairs(hoppings)
        m,n,R = B
        h = zeros(ComplexF64,n_wyck*n_orb,n_wyck*n_orb)
        for i = 1:n_orb,j = 1:n_orb
            x = n_orb*(i-1) + j # ρ ⊗ ρ for orbitals
            k = N*(x-1) + a # (ρ ⊗ ρ) ⊗ wyck 
            for (idxparam,t) in pairs(param)
                h[(m-1)*n_orb+i,(n-1)*n_orb+j] = t * V[k,idxparam]
                add_hoppings!(H,R,h) # TODO: unroll loop
            end
        end
    end

end

H = TB_Hamiltonian{ComplexF64,Int}(2*2,3)

add_terms!(H,V,orb,[1.0,0.5,0.3,1.2],2,2)
##
path = [[0,0,0],[1/2,0,0],[1/3,1/3,0],[0,0,0]]
labels = ["Γ","M","K","Γ"]
plot_Bandstructure(H,path,1000,labels;c=:darkred)

##
ρ = hopping_rep(D6h,wycks,1,1,[0,0,0],ρmn)

orb = collect(hopping_orbit(D6h,wycks,1,1,[0,0,0]))

χ = irreps(D6h)
Cls = conjugacy_classes(D6h)
ω = Diagonal([length(C) for C in Cls]) / length(D6h)
χh = characters(ρ,Cls)

A1g = [1.0 for C in Cls]
round.(A1g ⋅ (ω*χh);digits=3)

##
V=TightBindingToolBox.Model.get_basis_for_irrep(A1g,ρ,D6h)

add_terms!(H,V,orb,[-0.1,-0.2,0.1,0.3]*1E-1,2,2)