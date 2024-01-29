using TightBindingToolBox, LinearAlgebra

filename = "/home/brass/Projekte/Ce3Bi4Pd3/FPLO/+hamdata"
file = open(filename)

a0 = 1.889726125
a = 10.051 * a0
# TODO: do this for a != b != c
M = rationalize.(TightBindingToolBox.Parser.FPLO_get_lattice_vectors(file) / a,tol=1E-6)
Nwf = TightBindingToolBox.Parser.FPLO_get_number_WFs(file)
centers = rationalize.(TightBindingToolBox.Parser.FPLO_get_WF_centers(file,Nwf)/ a,tol=1E-6);

ρm = TightBindingToolBox.Parser.FPLO_import_symop(filename,"m(x-y)")


function FPLO_get_symop(filename,opname::String)
    file = open(filename)
    R = TightBindingToolBox.Parser.FPLO_get_lattice_vectors(file)
    n_WFs = TightBindingToolBox.Parser.FPLO_get_number_WFs(file)
    ρ = zeros(ComplexF64,n_WFs,n_WFs)
    TightBindingToolBox.Parser.skip_to_line_containing!(opname,file)
    α = [0 0 0; 0 0 0; 0 0 0//1]
    readline(file)
    α[1,:] = rationalize.(parse.(Float64,split(readline(file))))
    α[2,:] = rationalize.(parse.(Float64,split(readline(file))))
    α[3,:] = rationalize.(parse.(Float64,split(readline(file))))
    readline(file)
    τ = rationalize.(parse.(Float64,split(readline(file)))/a, tol=1E-6)
    line = TightBindingToolBox.Parser.skip_to_line_containing!("iwan eqMO:",file)
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

α, τ, ρ = FPLO_get_symop(filename,"m(x-y)");

function get_diff_lattice_vec(α,τ,w,ws)
    _, n = size(ws)
    v = α*w+τ 
    for i = 1:n
        R = inv(M)*(v - ws[:,i])
        #display(R)
        if mod(R[1],1)==0 && mod(R[2],1)==0 && mod(R[3],1)==0
            #println(i)
            return R
        #elseif mod(R[1],1)==1//2 && mod(R[2],1)==1//2 && mod(R[3],1)==1//2

        end
    end
    return nothing
end

get_diff_lattice_vec(α,τ,centers[:,1],centers)

diff_vecs = zeros(Rational{Int64},size(centers))

_, n = size(centers)
for i = 1:n
    println(i)
    diff_vecs[:,i] .= get_diff_lattice_vec(α,τ,centers[:,i],centers)
end

##
H = FPLO_import_TB(filename);
##
N = [0,0,1//2]
P = [1//4,1//4,1//4]

k = P -N
Hk = bloch_hamiltonian(H,k)

φ = Diagonal([exp(-2π*im*(k⋅R)) for R in eachcol(diff_vecs)])

norm(ρ*φ*Hk-Hk*ρ*φ) # this is the correct order: ρ*φ
ρmx_y = ρ*φ
##
foo = [exp(-im*(k⋅R)) for R in eachcol(diff_vecs)]
