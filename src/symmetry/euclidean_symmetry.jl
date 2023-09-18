using LinearAlgebra, Permutations, WignerD

struct SymOperation
    R::Matrix{Rational{Int64}}
    T::Vector{Rational{Int64}}
end

Base.:*(x::SymOperation,y::SymOperation) = SymOperation(x.R*y.R,mod.(x.R*y.T+x.T,1)) #,x.L)
#Base.:*(x::SymOperation,y::Vector) = mod.(x.R * y + x.T,1)
Base.:*(x::SymOperation,y::Vector) = x.R * y + x.T
Base.:(==)(x::SymOperation,y::SymOperation) = isequal(x,y)
Base.hash(x::SymOperation) = Base.hash(x.R)
Base.inv(x::SymOperation) = SymOperation(inv(x.R),mod.(-inv(x.R)*x.T,1)) #,x.L)
LinearAlgebra.det(x::SymOperation) = det(x.R)

function Base.isequal(x::SymOperation,y::SymOperation)
    x.R != y.R && return false
    return iszero(mod.((x.T - y.T),1))
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

function SO3_matrix(n::Int,k::Vector)
    K = [0 -k[3] k[2];
         k[3] 0 -k[1];
         -k[2] k[1] 0] ./norm(k)
    θ = 2π/n
    return I +sin(θ)*K + (1-cos(θ))*K*K
end

function rotation(n::Int,k::Vector,B::Matrix)
    R = SO3_matrix(n,k)
    return rationalize.(inv(B)*R*B;tol=1E-6)
end

function SymOperation(n::Int,k::Vector,T::Vector{Rational{Int64}}=[0//1,0,0],B::Matrix=[1 0 0;0 1 0; 0 0 1])
    R = rotation(n,k,B)
    return SymOperation(R,T)
end
SymOperation(n::Int,k::Vector,B::Matrix) = SymOperation(n,k,[0//1,0,0],B)

struct SpinSym
    R::SymOperation
    S::Matrix{ComplexF64}
end

Base.:*(x::SpinSym,y::SpinSym) = SpinSym(x.R*y.R,x.S*y.S)
Base.:*(x::SpinSym,v::Vector) = x.R * v
Base.:(==)(x::SpinSym,y::SpinSym) = x.R == y.R && norm(x.S-y.S) < 1E-6
Base.hash(x::SpinSym) = Base.hash(x.R)
Base.inv(x::SpinSym) = SpinSym(inv(x.R),Matrix(x.S'))

function SU2_matrix(n::Int,k::Vector)
    σ0 = [1 0; 0 1]
    σx = [0 1; 1 0]
    σ2 = [0 1; -1 0]
    σz = [1 0;0 -1]
    cosϕ = cos(π/n)
    sinϕ = sin(π/n)
    
    ψ = sinϕ/sqrt(k⋅k)
    a = cosϕ
    b = k[1] * ψ
    c = k[2] * ψ
    d = k[3] * ψ
    return a*σ0 + c*im*σx + b*σ2 + d*im*σz
end

SpinSym(n::Int,k::Vector,T::Vector{Rational{Int64}}=[0//1,0,0],B::Matrix=[1 0 0;0 1 0; 0 0 1]) = SpinSym(SymOperation(n,k,T,B),SU2_matrix(n,k))
SpinSym(n::Int,k::Vector,B::Matrix) = SpinSym(n,k,[0//1,0,0],B)


function euler_angles_ZYZ(R)
    if R[3,3]==1
        return atan(R[2,1],R[1,1]), 0.0, 0.0
    elseif R[3,3]==-1
        return atan(-R[2,1],-R[1,1]), π, 0.0
    else
        α = atan(R[2,3],R[1,3])
        β = atan(sqrt(1-R[3,3]^2),R[3,3])
        γ = atan(R[3,2],-R[3,1])
        return α,β,γ
    end
end

function representation(g::SymOperation,l::Int)
    σ = sign(det(g.R))
    return σ^l * wignerD(l,euler_angles_ZYZ(σ*g.R)...)
end

function representation(G::Set{SymOperation},l::Int)
    ρ = Dict{SymOperation,Matrix{ComplexF64}}()
    for g in G
        ρ[g] = representation(g,l)
    end
    return ρ
end

function representation(g::SpinSym,j::Rational,l::Int)
    σ = sign(det(g.R.R))
    α,β,γ = euler_angles_ZYZ(σ*g.R)
    S = WignerD(1//2,α,β,γ)
    if norm(S-g.S) < 1E-6
        return σ^l * wignerD(j,α,β,γ)
    else
        return -σ^l * wignerD(j,α,β,γ)
    end
end