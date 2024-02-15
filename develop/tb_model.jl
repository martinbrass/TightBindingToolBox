using TightBindingToolBox, LinearAlgebra

gen = [
    [-1//1 0 0; 0 -1 0; 0 0 1],
    [0 -1//1 0; 1 0 0; 0 0 1],
    [-1//1 0 0; 0 1 0; 0 0 -1]
]

D4 = generate_group(gen)

χ = irreps(D4)

Cls = conjugacy_classes(D4)

ϕ = π/5
U = [cos(ϕ) 0 sin(ϕ); 0 1 0; -sin(ϕ) 0 cos(ϕ)]
ρ = Dict([g=>U*g*U' for g in D4])

χρ = characters(ρ,Cls)

round.(χ'*Diagonal([length(C) for C in Cls])*χρ / length(D4))



P = projection(χ[:,3],ρ,D4)
##
function get_basis_for_irrep(χ,ρ,G)
    Cls = conjugacy_classes(G)
    dim = χ[TightBindingToolBox.Symmetry.get_index(TightBindingToolBox.Symmetry.neutral_element(G),Cls)]
    P = projection(χ,ρ,G)
    E,V = LAPACK.syev!('V','U',P)
    n = length(E)
    d = round(Int64,sum(E))
    if d == dim
        return V[:,n-d+1:n]
    elseif !(d % dim == 0)
        @error "get_basis_for_irrep: $dim does not divide $d"
        return nothing
    else
        return V#V[:,n-d+1:n] # BS
    end
end

function foo(ρ,V)
    d = size(V)[2]
    r = [V'*g*V for g in values(ρ)]
    f = Diagonal(1:d)
    H = sum(g*f*g' for g in r)
    E, W = LAPACK.syev!('V','U',H)

    return W
end

ρ = Dict([g=>kron(U*g*U',U*g*U') for g in D4])
P = projection(χ[:,3],ρ,D4)

V=get_basis_for_irrep(χ[:,3],ρ,D4)
W=foo(ρ,V)
Z = V*W
for g in values(ρ)
    i = 4
    display(real(round.(Z'*(g*Z),digits=6)))
    #display(norm(P*g-g*P))
end 