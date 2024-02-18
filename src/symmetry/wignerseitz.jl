using Plots, LinearAlgebra

struct Plane{F} 
    normal::Vector{F}
end

struct Line{F}
    v::Vector{F}
    p::Vector{F}
end

function intersection(A::Plane,B::Plane)
    a = A.normal
    b = B.normal
    n = (a×b)
    if norm(n) < 1E-16
        return nothing
    else
        an = a×n
        α = ((b-a)⋅b)/(an⋅b)
        Q = a + α * an
        return Line(n,Q)
    end
end

function intersection(A::Line,B::Line)
    d = (B.p-A.p)
    n = A.v × B.v
    norm(n) < 1E-16 && return nothing
    if abs(d⋅n) < 1E-16
        M = [A.v -B.v]
        S = M'*M
        α = inv(S)*M'*d
        return A.p + α[1]*A.v
    else
        return nothing
    end
end

function intersection(A::Line,B::Plane)
    if abs(A.v ⋅ B.normal) < 1E-16
        return nothing
    else
        α = ((B.normal-A.p)⋅B.normal)/(A.v⋅B.normal)
        return A.p + α*A.v
    end
end

function intersection0(A::Plane,B::Plane)
    a = A.normal
    b = B.normal
    n = (a×b)
    if norm(n) < 1E-16
        return nothing
    else
        an = a×n
        α = ((b)⋅b)/(an⋅b)
        Q = α * an
        return Line(n,Q)
    end
end

intersection(B::Plane,A::Line) = intersection(A,B)

function is_in_wsz(q,P)
    R = [P*[x,y,z] for x in [-1,0,1], y in [-1,0,1], z in [-1,0,1] if x !=0 || y!=0||z!=0]
    n = norm(q)
    for r in R
        norm(r-q) < n && return false
    end
    return true
end

function is_on_line(x,l::Line)
    α = ((x-l.p) ⋅ l.v)/(l.v⋅l.v)
    return norm((x-l.p) - α*l.v) < 1E-6
end

function is_partner(x,y,L)
    #norm(x-y) < 1E-16 && return false
    for l in L
        is_on_line(x,l) && is_on_line(y,l) && return true
    end
    return false
end

function wignerseitz(P)
    F = [Plane(P*[x,y,z]) for x in [-1/2,0,1/2], y in [-1/2,0,1/2], z in [-1/2,0,1/2] if x !=0 || y!=0||z!=0]
    L = [intersection(f,g) for f in F, g in F if !isnothing(intersection(f,g))]
    Q = Set([round.(q,digits=6) for q in (intersection(f,g) for f in L, g in L) if !isnothing(q) && is_in_wsz(q,P)])
    partners = [(q,p) for q in Q, p in Q if is_partner(q,p,L)]
    return Q, partners
end

function plot_wignerseitz(P;kwargs...)
    Q,part = wignerseitz(P)
    X = [p[1] for p in Q]
    Y = [p[2] for p in Q]
    Z = [p[3] for p in Q]

    #plt=scatter(X,Y,Z,legend=false;kwargs...)
    (q,p) = part[1]
    plt = plot([p[1],q[1]],[p[2],q[2]],[p[3],q[3]],legend=false,c=:black;kwargs...)

    for (p,q) in part
        plot!(plt,[p[1],q[1]],[p[2],q[2]],[p[3],q[3]],legend=false,c=:black;kwargs...)
    end
    return plt
end