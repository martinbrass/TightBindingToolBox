using TightBindingToolBox, LinearAlgebra, Plots, LaTeXStrings

operator_rep(basis,ρ) = Dict([
    g => [tr(A'*r*B*r') for A in basis, B in basis]
    for (g,r) in ρ 
])

function orbit_rep(G,v)
    orb = collect(orbit(G,v))
    d = length(orb)
    ρ = Dict([
        g => [ i == findfirst(x->x==g*orb[j],orb) ? 1.0 : 0.0 for i=1:d, j=1:d]
        for g in G
    ])
    return ρ
end

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

function get_basis_for_irrep(χ,ρ,G)
    Cls = conjugacy_classes(G)
    dim = χ[TightBindingToolBox.Symmetry.get_index(TightBindingToolBox.Symmetry.neutral_element(G),Cls)]
    P = projection(χ,ρ,G)
    E,V = LAPACK.syev!('V','U',P)
    n = length(E)
    d = round(Int64,sum(E))
    return V[:,n-d+1:n] # TODO: is this important?

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

struct TB_model{F,T}
    G::Set{T}
    ρ::Dict{T, Matrix{F}}
    ρH::Dict{T, Matrix{F}}
    χ::Matrix{F}
    op_basis::Vector{Matrix{ComplexF64}}
    H::TB_Hamiltonian{ComplexF64,Int}
end

function TB_model(G::Set{T},ρ::Dict{T, Matrix{F}}) where {T,F}
    χ = irreps(G)
    n = size(first(ρ)[1])[1] # lattice dim
    d = size(first(ρ)[2])[1] # local orbitals dim
    op_basis = operator_basis(d)
    ρH = operator_rep(op_basis,ρ)
    H = TB_Hamiltonian{ComplexF64,Int}(d,n)

    return TB_model{F,T}(G,ρ,ρH,χ,op_basis,H)
end

function init_terms(M::TB_model,R)
    G = M.G
    Rs = collect(orbit(G,R))
    ρR = orbit_rep(G,R)
    Cls = conjugacy_classes(G)
    ω = Diagonal([length(C) for C in Cls]) / length(G)

    ρRH = Dict([
        g => kron(ρR[g],M.ρH[g])
        for g in M.G
    ])
    A1g = ones(Int,length(Cls))
    χRH = characters(ρRH,Cls)
    n_param = round(Int,real(((A1g ⋅ (ω*χRH) ))))
    @info "The TB_model needs $n_param parameters for the given hoppings."
    V=get_basis_for_irrep(A1g,ρRH,G)

    return V, Rs
end

function add_terms!(M::TB_model,V,Rs,param)
    H = M.H
    d = length(M.op_basis)

    for (a,R) in pairs(Rs)
        for (b,τ) in pairs(M.op_basis)
            n = d*(a-1) + b
            for (i,t) in pairs(param)
                h = t * V[n,i] * τ
                add_hoppings!(H,R,h) # TODO: unroll loop
            end
        end
    end

end

##

using TightBindingToolBox, LinearAlgebra, Plots, LaTeXStrings

Oh =generate_group([
    Matrix(Diagonal([-1//1,-1,1])),
    Matrix(Diagonal([-1//1,1,-1])),
    Matrix(Diagonal([-1//1,-1,-1])),
    [0 0 1//1; 1 0 0; 0 1 0],
    [0 1//1 0; 1 0 0; 0 0 -1]
])

ρT = Dict([g=> Float64.(g) for g in Oh])

M = TB_model(Oh,ρT)

V, Rs = init_terms(M,[1,0,0]);

add_terms!(M,V,Rs,[-1/2,1]);

path = [[0,0,0],[1/2,0,0],[1/2,1/2,0],[1/2,1/2,1/2],[0,0,0]]
labels = ["Γ","X","M","R","Γ"]
plot_Bandstructure(M.H,path,100,labels;c=:darkred)
##

V, Rs = init_terms(M,[1,1,0]);

#add_terms!(M,V,Rs,[0.2,0.1,-0.1]*5.1);
add_terms!(M,V,Rs,[0.2,0.1,-0.1]*7);

plot_Bandstructure(M.H,path,1000,labels;c=:darkred)
##
ρmx_y = ρT[[1 0 0;0 1 0;0 0 -1]];
H = M.H
idx1 = 2; kr = [1/2,0,0]; ks = [0,1/2,0]
Hk = bloch_hamiltonian(H,kr);
N=801
r1 = range(-1,1,N)
r2 = range(-1,1,N)
Z = @time [begin
    k = x*kr+y*ks;
    bloch_hamiltonian!(H,k,Hk);
    E, V = LAPACK.syev!('V','U',Hk);

    #Variante 1:
    #ϕ = real(V[:,idx1] ⋅ (ρmx_y * V[:,idx1])* exp(2π*im*(k⋅(inv(M)*τ))));
    #Variante 2:
    X = V' *(ρmx_y * V);
    ϕ = sum(max(0,real(X[i,i])) for i = 1:idx1);
    #Variante 3:
    #ϕ = V[:,idx1] ⋅ (ρmx_y * V[:,idx1]);
    if isapprox(E[idx1-1],E[idx1];atol=1E-8) || isapprox(E[idx1+1],E[idx1];atol=1E-8)
        ϕ=0.0
    end
    #=
    if !isapprox(abs(ϕ),1.0; atol=1E-8) 
        if isapprox(E[idx1-1],E[idx1];atol=1E-8) || isapprox(E[idx1+1],E[idx1];atol=1E-8)
            ϕ=0.0
        else
            display(E[idx1-1:idx1+1])
        end  
    end ;=#
    ϕ end for x in r1, y in r2];
    ##
    plt = heatmap(r1,r2,Z'; #max.(Z' .-1,0);
    c=:linear_kry_0_97_c73_n256, #:diverging_bkr_55_10_c35_n256, #:diverging_bkr_55_10_c35_n256, #:bwr,
    size=(450,500),
    aspect_ratio=1,
    xlabel=L"$k_{x}$",
    ylabel=L"$k_y$",
    ylims=(-1,1),
    xlims=(-1,1),
    #clims=(-1,1),
    legend=false,
    frame=:box
)