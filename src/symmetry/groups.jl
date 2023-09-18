using Permutations

conjugate(g,h) = h*g*inv(h)
coset(g,H::Set) = Set(map(h->g*h,collect(H)))
quotient(G,N) = Set(map(g->coset(g,N),collect(G)))


function is_in_conj_class(g,C,G)
    for h in G
        if (conjugate(g,h) in C)
            return true
        end
    end
    return false    
end

function conjugacy_classes(G)
    classes = Set{typeof(G)}()
    for g in G
        inQ = false
        for C in classes
            if is_in_conj_class(g,C,G)
                push!(C,g)
                inQ = true
                break
            end
        end
        if (!inQ) push!(classes,Set([g])) end
    end
    return classes
end

function generate_group(generators)
    G = Set(generators)
    T = eltype(G)
    old_set = deepcopy(G)
    new_set = Set{T}()
    while !isempty(old_set)
        for g in old_set, h in generators
            push!(new_set,g*h)
        end
        old_set = setdiff(new_set,G)
        union!(G,new_set)
    end
    return G
end

function orbit(G,k)
    orb = Set([k])
    for g in G
        push!(orb,g*k)
    end
    return orb
end

function left_regular_dict(G::Set{T}) where T
    lrd = Dict{T,Int64}()
    count = 1
    for g in G
        lrd[g] = count
        count += 1
    end
    return lrd
end

function left_regular_representation(G::Set{T}) where T
    n = length(G)
    rep = Dict{T,Matrix{Int64}}()
    lrd = left_regular_dict(G)
    for g in G
        rep[g] = zeros(Int64,n,n)
        for (h,j) in lrd
            i = lrd[g*h]
            rep[g][i,j] = 1
        end
    end
    return rep
end

function to_permutations(G) 
    rep = left_regular_representation(G)
    perms = map(x->Permutation(x[2]),collect(rep))
    return Set(perms)
end

characters(ρ,classes) = map(C->tr(ρ[first(C)]),collect(classes))

function cycle_length(g,e)
    h = g
    l = 1
    while h != e
        h = g*h
        l += 1
    end
    return l
end

function c_RST(R,S,T)
    t = first(T)
    c = 0
    for r in R, s in S
        if r*s == t
            c += 1
        end
    end
    return c
end

function c_RST(classes)
    N = length(classes)
    c = zeros(Int64,N,N,N)
    for (r,R) in pairs(collect(classes))
        for (s,S) in pairs(collect(classes))
            for (t,T) in pairs(collect(classes))
                c[r,s,t] = c_RST(R,S,T)
            end
        end
    end
    return c
end

function neutral_element(G)
    for g in G
        if g*g == g
            return g
        end
    end
end

function get_index(g,classes)
    for (i,C) in pairs(collect(classes))
        if (g in C)
            return i
        end
    end
end

function irreps(G)
    e = neutral_element(G)
    classes = conjugacy_classes(G)
    idx_e = get_index(e,classes)
    N = length(classes)
    ns = map(x->length(x),collect(classes))
    crst = c_RST(classes)
    M = zeros(Int64,N,N)
    for k = 1:N
        M .+= 10^(k-1) * crst[k,:,:]
    end
    U = eigvecs(M)
    for k = 1:N
        #U[:,k] ./= U[idx_e,k]
        U[:,k] ./= ns
        len = sqrt((U[:,k] ⋅ (ns .* U[:,k]))/length(G))
        U[:,k] ./= len / abs(U[idx_e,k])*U[idx_e,k]
    end
    return U
end


    
function class_labels(G)
    e = neutral_element(G)
    classes = conjugacy_classes(G)
    N = length(classes)
    cyclens = map(x->cycle_length(first(x),e),collect(classes))
    n_elem = map(x->length(x),collect(classes))
    dets = map(x->round(Int64,det(first(x))),collect(classes))
    labels = Matrix{String}(undef,1,N)
    for k = 1:N
        if (cyclens[k] == 1)
           labels[1,k] = "E"
        else
            s = "$(n_elem[k])"
            if (dets[k] == 1)
                s *= "C$(cyclens[k])"
            else
                s *= "S$(cyclens[k])"
            end
            labels[1,k] = s
        end
    end
    return labels
end

function element_labels(G)
    e = neutral_element(G)
    N = length(G)
    cyclens = map(x->cycle_length(x,e),collect(G))
    dets = map(x->round(Int64,det(x)),collect(G))
    labels = Matrix{String}(undef,1,N)
    for k = 1:N
        if (cyclens[k] == 1)
           labels[1,k] = "E"
        else
            if (dets[k] == 1)
                labels[1,k] = "C$(cyclens[k])"
            else
                labels[1,k] = "S$(cyclens[k])"
            end
            
        end
    end
    return labels
end

function stabilizer(G,k)
    S = typeof(G)()
    for g in G
        if (g*k == k)
            push!(S,g)
        end
    end
    return S
end

function multiplication_table(G)
    labels = element_labels(G)
    dict = left_regular_dict(G)
    N = length(G)
    M = zeros(Int64,N,N)
    for g in G, h in G
        i = dict[g]
        j = dict[h]
        M[i,j] = dict[g*h]
    end
    lM = vcat(labels,M)
    return hcat(vcat([" "],labels[:]),lM)
end

function projection(χ,ρ,G)
    n = size(first(ρ)[2])[1]
    P = zeros(ComplexF64,n,n)
    clsss = conjugacy_classes(G)
    d = χ[get_index(neutral_element(G),clsss)]
    for g in G
        k = get_index(g,clsss)
        P .+= conj(χ[k])*ρ[g]
    end
    return P * d/length(G)
end

function isgroup(G)
    e = neutral_element(G)
    (e === nothing) && return false
    for g in G
        inv(g) ∉ G && return false #, g
        for h in G
            g*h ∉ G && return false #, g, h
        end
    end
    return true
end