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
    @testset "init_terms" begin
        V, Rs = init_terms(M,[1,0,0]);
        @test size(V) == (9*6,2)
        @test length(Rs) == 6
        add_terms!(M,V,Rs,[-1/2,1]);
    end

    path = [[0,0,0],[0.3,0,0],[1/2,0,0],[1/2,0.1,0],[1/2,1/2,0],[1/2,1/2,0.4],[1/2,1/2,1/2],[-0.12,0.34,-0.7]]
    @testset "hermiticity/commutativity" begin
        for k in path
            Hk = bloch_hamiltonian(M.H,k)
            @test norm(Hk-Hk') < 1E-15
            for (g,ρ) in pairs(ρT)
                Hk2 = bloch_hamiltonian(M.H,g*k)
                @test (norm(Hk-ρ'*Hk2*ρ) < 1E-15)
            end
        end
    end
    @testset "real" begin
        for k in path
            Hk = bloch_hamiltonian(M.H,k)
            @test norm(imag(Hk))< 1E-15
        end
    end
    @testset "operator_basis" begin
        σ = TightBindingToolBox.Model.operator_basis(2)
        @test σ[1] == [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im]
        @test σ[2] == [0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0 + 0.0im]
        @test σ[3] == [0.0 + 0.0im 0.7071067811865475 + 0.0im; 0.7071067811865475 + 0.0im 0.0 + 0.0im]
        @test σ[4] == [0.0 + 0.0im 0.0 - 0.7071067811865475im; 0.0 + 0.7071067811865475im 0.0 + 0.0im]     
    end

end

