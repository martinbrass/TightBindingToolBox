include("../src/groups.jl")
include("../src/euclidean_symmetry.jl")

R = SO3_matrix(3,[0,0,1]);
a = [1,0,0.]
b = R*a
c = [0,0,1.]
B = [a b c]

gen = [SpinSym(3,[0,0,1],B)]

G = generate_group(gen); length(G)

irreps(G)

class_labels(G)

