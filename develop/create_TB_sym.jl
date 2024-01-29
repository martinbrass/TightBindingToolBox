using TightBindingToolBox, WignerD, LinearAlgebra

g1 = SymOperation(2,[0,0,1])
g2 = SymOperation(2,[0,2,0])
g3 = SymOperation(3,[1,1,1])
g4 = SymOperation(2,[1,1,0])
g5 = SymOperation([-1//1 0 0; 0 -1 0; 0 0 -1],[0//1,0,0])

G = generate_group([g1,g2,g3,g4,g5]); length(G)
##
k = [1,0,0]
O = orbit(G,k); length(O)
##
S = stabilizer(G,k)

isgroup(S)
##

##
ρ = representation(G,2);

ccs = conjugacy_classes(G);

χ = characters(ρ,ccs);

χI = irreps(G);

ω = Diagonal(map(x->length(x)/length(G),collect(ccs)));

Int.(round.(χI' * ω * χ, digits=6))

round.(Int,χI')

e = TightBindingToolBox.Symmetry.neutral_element(G)

idx_e = TightBindingToolBox.Symmetry.get_index(e,ccs)