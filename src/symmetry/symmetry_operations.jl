using LinearAlgebra, Permutations

struct SymOperation
    R::Matrix{Rational{Int64}}
    T::Vector{Rational{Int64}}
    #::Matrix{Rational{Int64}} # trafo from cartesian to lattice vectors
end

Base.:*(x::SymOperation,y::SymOperation) = SymOperation(x.R*y.R,mod.(x.R*y.T+x.T,1)) #,x.L)
Base.:*(x::SymOperation,y::Vector) = mod.(x.R * y + x.T,1)
Base.:(==)(x::SymOperation,y::SymOperation) = isequal(x,y)
Base.hash(x::SymOperation) = Base.hash(x.R)
Base.inv(x::SymOperation) = SymOperation(inv(x.R),mod.(-inv(x.R)*x.T,1)) #,x.L)
LinearAlgebra.det(x::SymOperation) = det(x.R)

function Base.isequal(x::SymOperation,y::SymOperation)
    x.R != y.R && return false
    return iszero(mod.((x.T - y.T),1))
    #=
    t = x.L*mod.((x.T - y.T),1)
    for c in t
        denominator(c) != 1 && return false
    end
    return true =#
end

function translations(G)
    T = Set{SymOperation}();
    for g in G 
        if g.R == I
            push!(T,g)
        end
    end
    return T
end

struct SpinRep{F}
    sym_op::SymOperation
    ρ::Matrix{F}
end

Base.:*(x::SpinRep,y::SpinRep) = SpinRep(x.sym_op*y.sym_op,x.ρ*y.ρ)
Base.:*(x::SpinRep,v::Vector) = x.sym_op * v
Base.:(==)(x::SpinRep,y::SpinRep) = x.sym_op == y.sym_op && norm(x.ρ-y.ρ) < 1E-6
Base.hash(x::SpinRep) = Base.hash(x.sym_op)
Base.inv(x::SpinRep) = SpinRep(inv(x.sym_op),Matrix(x.ρ'))