using TightBindingToolBox, LinearAlgebra, Test

@testset "Core" begin
    
    H = TB_Hamiltonian{ComplexF64,Int}(2,3)
    id = [1. 0; 0 1];
    add_hoppings!(H,[0,0,0],id);
    d = H.local_dim
    n_layer = 5;
    Hs = zeros(ComplexF64,n_layer*d,n_layer*d)
    bz = [0,0,1]
    k  = [0,0,0]
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test norm(Hs-I)==0

    add_hoppings!(H,[0,0,1],2*id);
    add_hoppings!(H,[0,0,-1],2*id);
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test norm(Hs) == sqrt(n_layer*d + 8*(n_layer*d-2))
    @test ishermitian(Hs)

    add_hoppings!(H,[0,0,2],3*id);
    add_hoppings!(H,[0,0,-2],3*id);
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test ishermitian(Hs)
    @test norm(Hs) == sqrt(n_layer*d + 8*(n_layer*d-2) + 2*9*(n_layer*d-4))
    
    Hs_old = copy(Hs)
    add_hoppings!(H,[1,0,0],id);
    add_hoppings!(H,[-1,0,0],id);
    slab_hamiltonian!(H,k,bz,n_layer,Hs)    
    @test ishermitian(Hs)
    @test Hs == Hs_old + 2I
    
    k=[0.5,0,0]
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test Hs == Hs_old - 2I
    @test ishermitian(Hs)
    
    add_hoppings!(H,[0,1, 0],im*id);
    add_hoppings!(H,[0,-1,0],-im*id);
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test ishermitian(Hs)
    @test Hs == Hs_old - 2I
    
    k=[0,1/4,0]
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test ishermitian(Hs)
    @test Hs == Hs_old 
    
    k=[0,1/8,0]
    add_hoppings!(H,[0,1, 1],im*id);
    add_hoppings!(H,[0,-1,-1],-im*id);
    slab_hamiltonian!(H,k,bz,n_layer,Hs)
    @test ishermitian(Hs)
    @test abs(norm(imag(Hs))^2 - (d*n_layer-2)) <= 2E-15
end