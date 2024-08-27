using TightBindingToolBox, LinearAlgebra, Test, Integrals

@testset "berry.jl" begin
    
    H = TightBindingHamiltonian(2)

    t = 1.0;
    k0 =π/2;
    γ = cos(k0)
    tx = t; ty = t; tz = t;

    σ = [[0 1; 1 0],im*[0. -1; 1 0],[1 0;0 -1]]

    H[[0,0,0]] =(2+γ)tz*σ[3]
    H[[1,0,0]] =-tz/2*σ[3]-im/2*tx*σ[1]
    H[[-1,0,0]] =-tz/2*σ[3]+im/2*tx*σ[1]
    H[[0,1,0]] =-tz/2*σ[3]-im/2*ty*σ[2]
    H[[0,-1,0]] =-tz/2*σ[3]+im/2*ty*σ[2]
    H[[0,0,1]] =-tz/2*σ[3]
    H[[0,0,-1]] =-tz/2*σ[3]
    
    @test abs(berry_flux(H,1,[0,0,1/4],0.2,50,50)-1) < 1E-3
    @test abs(berry_flux(H,1,[0,0,1/2],[1,0,0],[0,1,0];N=100)) < 1E-16
    @test abs(berry_flux(H,1,[0,0,0],[1,0,0],[0,1,0];N=100)+1)  < 2E-15

    # new version with Integrals.jl:

    Ω = BerryCurvature(H,1)
    S = Sphere([0,0,1/4],0.2)
    dom =([0.0, 0.0],[2π,π])
    prob = IntegralProblem(flux,dom,(Ω,S))    
    sol = solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)
    @test abs(sol.u-1) < 1E-6
    
    P = Plane([0,0,0],[1,0,0],[0,1,0])
    dom =([0.0, 0.0],[1,1.0])
    prob = IntegralProblem(flux,dom,(Ω,P))    
    sol = solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)
    @test abs(sol.u+1) < 1E-6

    P = Plane([0,0,1/2],[1,0,0],[0,1,0])
    prob = IntegralProblem(flux,dom,(Ω,P))    
    sol = solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)
    @test abs(sol.u) < 1E-6
end;