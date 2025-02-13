using .TightBindingToolBox
using LinearAlgebra, CSV, DataFrames

"""
Wannier90_import_TB reads the tight-binding output from Wannier90 (typically hr.dat) 
    and transform it into a TB_Hamiltonian structure
"""
function Wannier90_import_TB(file_name)
    local_dim = CSV.read(file_name,DataFrame,delim=" ",ignorerepeated=true,header=false,skipto=2,limit=1)[1,1]
    n2 = local_dim*local_dim
    n_hoppings= CSV.read(file_name,DataFrame,delim=" ",ignorerepeated=true,header=false,skipto=3,limit=1)[1,1]
    lattice_dim = 3
    n_weight_lines = ceil(Int64,n_hoppings/15) # TODO is 15 always true?
    n_weights_last_line = n_hoppings % 15
    df=CSV.read(file_name,DataFrame,delim=" ",ignorerepeated=true,header=false,skipto=4,limit=n_weight_lines)
    rweights = collect(df[1,:]) # TODO: numbers need to be adjusted for future use
    for i in 2:n_weight_lines-1
        append!(rweights,collect(df[i,:]))
    end
    append!(rweights,collect(df[n_weight_lines,1:n_weights_last_line]))
    
    H = TightBindingHamiltonian{ComplexF64,Int}(local_dim)
    df = CSV.read(file_name,DataFrame,delim=" ",ignorerepeated=true,header=false,skipto=4+n_weight_lines)
    
    for idxR in 1:n_hoppings
        R = collect(df[1+(idxR-1)*n2,1:3])
        h = zeros(Complex{Float64},local_dim,local_dim)
        for i in 1:local_dim
            for j in 1:local_dim
                h[i,j] = df[(idxR-1)*n2+(j-1)*local_dim+i,6] + im*df[(idxR-1)*n2+(j-1)*local_dim+i,7]
                
                idxi = df[(idxR-1)*n2+(j-1)*local_dim+i,4]
                idxj = df[(idxR-1)*n2+(j-1)*local_dim+i,5]
                if idxi != i || idxj != j
                    throw(ErrorException("at import_TB_from_file: inconsistency in the input file"))
                end
            end
        end
        H[R] += h/rweights[idxR]
    end
    return H
end

"""
FPLO_import_TB reads the tight-binding output from FPLO (typically +hamdata) 
    and transform it into a TB_Hamiltonian structure
"""
function FPLO_import_TB(filename)
    file = open(filename)
    R = FPLO_get_lattice_vectors(file)
    basis_trafo = inv(R)
    n_WFs = FPLO_get_number_WFs(file)
    H = TightBindingHamiltonian(n_WFs)
    centers = FPLO_get_WF_centers(file,n_WFs)
    FPLO_get_hoppings!(file,n_WFs,basis_trafo,centers,H)
    return H
end

"""
import symmetry operation from the FPLO file +hamdata
opnames must match the FPLO naming conventions
"""
FPLO_import_symops(filename,opnames::Vector{String}) = map(s->FPLO_import_symop(filename,s),opnames)

"""
import symmetry operation from the FPLO file +hamdata
"""
function FPLO_get_symop(filename,opname::String,a)
    file = open(filename)
    R = FPLO_get_lattice_vectors(file)
    n_WFs = FPLO_get_number_WFs(file)
    ρ = zeros(ComplexF64,n_WFs,n_WFs)
    skip_to_line_containing!(opname,file)
    α = [0 0 0; 0 0 0; 0 0 0//1]
    readline(file)
    α[1,:] = rationalize.(parse.(Float64,split(readline(file))))
    α[2,:] = rationalize.(parse.(Float64,split(readline(file))))
    α[3,:] = rationalize.(parse.(Float64,split(readline(file))))
    readline(file)
    τ = rationalize.(parse.(Float64,split(readline(file)))/a, tol=1E-6)
    line = skip_to_line_containing!("iwan eqMO:",file)
    while !eof(file) && !occursin("operation:",line) && !occursin("spin:",line)
        idxs = parse.(Int64,split(readline(file)))
        i = idxs[1]
        readline(file)
        coef = parse.(Float64,split(readline(file)))
        for k = 2:length(idxs)
            j = idxs[k]
            ρ[j,i] = coef[2(k-1)-1] + im * coef[2(k-1)]
        end
        line = readline(file)
    end
    return α, τ, ρ
end

function FPLO_import_symop(filename,opname::String)
    file = open(filename)
    R = FPLO_get_lattice_vectors(file)
    n_WFs = FPLO_get_number_WFs(file)
    ρ = zeros(ComplexF64,n_WFs,n_WFs)
    skip_to_line_containing!(opname,file)
    line = skip_to_line_containing!("iwan eqMO:",file)
    while !eof(file) && !occursin("operation:",line) && !occursin("spin:",line)
        idxs = parse.(Int64,split(readline(file)))
        i = idxs[1]
        readline(file)
        coef = parse.(Float64,split(readline(file)))
        for k = 2:length(idxs)
            j = idxs[k]
            ρ[j,i] = coef[2(k-1)-1] + im * coef[2(k-1)]
        end
        line = readline(file)
    end
    return ρ
end

function FPLO_import_space_group(filename)
    file = open(filename)
    U = FPLO_get_lattice_vectors(file)
    G = Set{SymOperation}()
    labels = Dict{SymOperation,String}()
    while !eof(file)
        line = readline(file)
        if occursin("operation:",line)
            label = split(readline(file))[2] # name
            line = readline(file) # alpha
            g = Matrix{Float64}(undef,3,3)
            g[1,:] = parse.(Float64,split(readline(file)))
            g[2,:] = parse.(Float64,split(readline(file)))
            g[3,:] = parse.(Float64,split(readline(file)))
            h = inv(U)*g*U
            if norm(h-round.(Int64,h)) > 1E-6
                @error "FPLO_import_space_group: rotation contains non-integer entries"
            else
                line = readline(file) # tau
                t = parse.(Float64,split(readline(file)))
                s = SymOperation(h,rationalize.(t))
                labels[s] = label
                push!(G,s)
            end
        end
    end
    return G, labels
end

function skip_to_line_containing!(text,file)
    line = readline(file)
    while !eof(file) && !occursin(text,line)
        line = readline(file)        
    end
    return line
end

"""
import the lattice vectors form FPLO's +hamdata
"""
function FPLO_get_lattice_vectors(file)
    R = zeros(3,3)
    while !eof(file)
        line = readline(file)
        if occursin("lattice_vectors:",line)
            for i in 1:3
                s = split(readline(file))
                R[:,i] = parse.(Float64,s)
            end
            return R
        end
    end
    throw(error("lattice_vectors: not found in file\n"))
end

function FPLO_get_number_WFs(file)
    while !eof(file)
        line = readline(file)
        if occursin("nwan:",line)
            return parse(Int64,split(readline(file))[1])
        end
    end
    throw(error("nwan: not found in file\n"))
end

"""
import the wavefunction-centers form FPLO's +hamdata
"""
function FPLO_get_WF_centers(file,n_WFs)
    R = zeros(3,n_WFs)
    while !eof(file)
        line = readline(file)
        if occursin("wancenters:",line)
            for i in 1:n_WFs
                s = split(readline(file))
                R[:,i] = parse.(Float64,s)
            end
            return R
        end
    end
    throw(error("wancenters: not found in file\n"))
end

function FPLO_get_hoppings!(file,n_WFs,basis_trafo,centers,H)
    h0 = zeros(ComplexF64,n_WFs,n_WFs)
    while !eof(file)
        line = readline(file)
        if occursin("Tij, Hij",line) 
            idx = parse.(Int64,split(readline(file)))
            i = idx[1]
            j = idx[2]
            line = readline(file)
            while !occursin("end Tij, Hij",line) 
                # parse a, convert it to lattice multiples, add h0, modify Hij
                s = split(line)
                a = parse.(Float64,s[1:3])
                R_cartesian = basis_trafo * (a - centers[:,j] + centers[:,i])
                R_lattice = round.(Int64,R_cartesian)
                if norm(R_cartesian-R_lattice) > 0.01
                    throw(error("Error: in FPLO_get_hoppings! lattice conversion failed\n"))
                end
                H[R_lattice] += h0
                H[R_lattice][i,j] = parse(Float64,s[4]) + im*parse(Float64,s[5])
                line = readline(file)
            end
        end
    end
end
