using LinearAlgebra, TightBindingToolBox, Plots, LaTeXStrings, BenchmarkTools

filename = "/home/brass/Projekte/LK99/TB_Ham_4spinbands/+hamdata"
#filename = "/home/brass/Projekte/Ce3Bi4Pd3/FPLO/+hamdata"
H = FPLO_import_TB(filename);

##
d = H.local_dim;
Hs = zeros(ComplexF64,d,d);
k = [0.,0,0];
@benchmark bloch_hamiltonian!(H,k,Hs)

##
H2 = TBHamiltonian{3,length(H.hoppings)}(H);
##
using StaticArrays
k = SVector(0.,0,0)
@benchmark bloch_hamiltonian!(H2,k,Hs)
##
H3 = TBHam{3,length(H.hoppings)}(H);
@benchmark bloch_hamiltonian!(H3,k,Hs)
##
@code_warntype bloch_hamiltonian!(H2,k,Hs)