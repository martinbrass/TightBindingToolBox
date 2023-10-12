using TightBindingToolBox, LinearAlgebra, Test

@testset "Berry" begin
    
    H = TB_Hamiltonian{ComplexF64,Int}(2,3)

    t = 1.0;
    k0 =π/2;
    γ = cos(k0)
    tx = t; ty = t; tz = t;

    σ = [[0 1; 1 0],im*[0. -1; 1 0],[1 0;0 -1]]

    add_hoppings!(H,[0,0,0],(2+γ)tz*σ[3])
    add_hoppings!(H,[1,0,0],-tz/2*σ[3]-im/2*tx*σ[1])
    add_hoppings!(H,[-1,0,0],-tz/2*σ[3]+im/2*tx*σ[1])
    add_hoppings!(H,[0,1,0],-tz/2*σ[3]-im/2*ty*σ[2])
    add_hoppings!(H,[0,-1,0],-tz/2*σ[3]+im/2*ty*σ[2])
    add_hoppings!(H,[0,0,1],-tz/2*σ[3])
    add_hoppings!(H,[0,0,-1],-tz/2*σ[3])
    
    @testset "integration" begin
        @test abs(integrate_berry_curvature_sphere(H,1,[0,0,1/4],0.2,50,50)/2π-1) < 1E-3
        @test abs(berry_flux_through_plane(H,1,[0,0,1/2],[1,0,0],[0,1,0];N=100)) < 1E-16
        @test abs(berry_flux_through_plane(H,1,[0,0,0],[1,0,0],[0,1,0];N=100)+1)  < 2E-15

    end

end