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


function lattice_rep(G,N)
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

rr = lattice_rep(D4,3)
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

##



##
operator_rep(basis,ρ) = Dict([
    g => [tr(A'*r*B*r') for A in basis, B in basis]
    for (g,r) in ρ 
])

operator_basis(N) = vcat(
    [Matrix(Diagonal([i==j ? 1.0+0im : 0.0im for j = 1:N])) for i=1:N],
    [begin
        E = zeros(ComplexF64,N,N);
        E[j,k] = 1/sqrt(2);
        E[k,j] = 1/sqrt(2);
        E
    end for j = 1:N for k = j+1:N
    ],
    [begin
        E = zeros(ComplexF64,N,N);
        E[j,k] = -1im/sqrt(2);
        E[k,j] = 1im/sqrt(2);
        E
    end for j = 1:N for k = j+1:N
    ]
)

basis = operator_basis(2)

V=get_basis_for_irrep(χ[:,3],ρ,D4)

ρE = Dict([g=> V'*r*V for (g,r) in ρ])

ρO = operator_rep(basis,ρE)

χO = characters(ρO,Cls)

round.(χ'*Diagonal([length(C) for C in Cls])*χO / length(D4))

## Oh stuff

Oh =generate_group([
    Matrix(Diagonal([-1//1,-1,1])),
    Matrix(Diagonal([-1//1,1,-1])),
    Matrix(Diagonal([-1//1,-1,-1])),
    [0 0 1//1; 1 0 0; 0 1 0],
    [0 1//1 0; 1 0 0; 0 0 -1]
])

χ = irreps(Oh)
ρT = Dict([g=> g for g in Oh])
basis = operator_basis(3)
ρH = operator_rep(basis,ρT)
Cls = conjugacy_classes(Oh)
χH = characters(ρH,Cls)
ω = Diagonal([length(C) for C in Cls]) / length(Oh)
round.(χ'*ω*χH )
##
A1g = ones(10)

V=get_basis_for_irrep(A1g,ρH,Oh)

H_A1g = sum(V[i,1]*basis[i] for i = 1:length(basis))

function orbit_rep(G,v)
    orb = collect(orbit(G,v))
    d = length(orb)
    ρ = Dict([
        g => [ i == findfirst(x->x==g*orb[j],orb) ? 1.0 : 0.0 for i=1:d, j=1:d]
        for g in G
    ])
    return ρ
end

ρR = orbit_rep(Oh,[1,0,0])
χR = characters(ρR,Cls)
round.(χ'*ω*χR )

round.(χ[:,3])

V=get_basis_for_irrep(A1g,ρR,Oh)

##

H = TB_Hamiltonian{ComplexF64,Int}(3,3)
Rs = collect(orbit(Oh,[1,0,0]))

V=get_basis_for_irrep(A1g,ρH,Oh)

H_A1g = sum(V[i,1]*basis[i] for i = 1:length(basis))

V=get_basis_for_irrep(A1g,ρR,Oh)

t = 1
t2 = -1/2
for (a,R) in pairs(Rs)
    add_hoppings!(H,R,t*V[a,1]*H_A1g)
end

Eg = round.(χ[:,3])

vH=get_basis_for_irrep(Eg,ρH,Oh)
vR=get_basis_for_irrep(Eg,ρR,Oh)

ρRH = Dict([
    g => kron(vR'*ρR[g]*vR,vH'*ρH[g]*vH)
    for g in Oh
])

V=get_basis_for_irrep(A1g,ρRH,Oh)
d = size(vH)[2]
for i = 1:d, j = 1:d
    n = d*(i-1) + j
    for (a,R) in pairs(Rs)
        for (b,τ) in pairs(basis)
            h = t2 * vR[a,i] * vH[b,j] * V[n,1] * τ
            #display(h)
            add_hoppings!(H,R,h)
        end
    end
end


path = [[0,0,0],[1/2,0,0],[1/2,1/2,0],[1/2,1/2,1/2],[0,0,0]]
labels = ["Γ","X","M","R","Γ"]
plot_Bandstructure(H,path,100,labels;c=:darkred)