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
        v = @view V[:,n-d+1:n]
        r = [v'*g*v for g in values(ρ)]
        f = Diagonal(rand(d))
        H = sum(g*f*g' for g in r)
        E, W = LAPACK.syev!('V','U',H)
        return v*W
    end
end


function plane_wave_rep(G,N)
    N3 = N*N*N
    ρ = Dict([
            begin
                r = zeros(Int64,N3,N3)
                for nx = 1:N, ny = 1:N, nz = 1:N
                    k = convert.(Int64,(g*([nx,ny,nz] .-(N-1))));
                    j = nx + N*(ny-1) + N*N*(nz-1);
                    i = (k[1]+(N-1)) + N*(k[2]+(N-1)-1) + N*N*(k[3]+(N-1)-1);
                    r[i,j] = 1;
                end;
                g=>r
            end
         for g in G
    ])
    return ρ
end

rr = plane_wave_rep(D4,3)
χr = characters(rr,Cls)

round.(χ'*Diagonal([length(C) for C in Cls])*χr / length(D4))

##

function foo(ρ,V)
    d = size(V)[2]
    r = [V'*g*V for g in values(ρ)]
    f = Diagonal(rand(d))
    H = sum(g*f*g' for g in r)
    E, W = LAPACK.syev!('V','U',H)

    return W
end

#ρ = Dict([g=>kron(U*g*U',U*g*U') for g in D4])
ρ = Dict([g=>kron(g,g) for g in D4])
P = projection(χ[:,3],ρ,D4)

V=get_basis_for_irrep(χ[:,3],ρ,D4)
W=foo(ρ,V)
Z = V*W
for g in values(ρ)
    i = 4
    display(real(round.(V'*(g*V),digits=6)))
    #display(norm(P*g-g*P))
end 