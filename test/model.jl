using TightBindingToolBox, LinearAlgebra, Test

@testset "Model" begin
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
    @test size(V) == (9*6,2)
    @test length(Rs) == 6
    add_terms!(M,V,Rs,[-1/2,1]);

    path = [[0,0,0],[0.3,0,0][1/2,0,0],[1/2,0.1,0][1/2,1/2,0],[1/2,1/2,0.4],[1/2,1/2,1/2],[-0.12,0.34,-0.7]]
    for k in path
        

end