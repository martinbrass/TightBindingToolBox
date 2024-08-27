using TightBindingToolBox, LinearAlgebra, Test

@testset "TightBindingHamiltonian" begin
    H = TightBindingHamiltonian(2)

    E0 = 0.01
    t0 = -6E-3
    t1 = -2E-3 
    u0 = -1.2E-2 
    g0 = -1.25E-2 
    u1 = 1E-3 
    g1 = 1E-2 
    g0z= 4E-3 
    ω1 = 1 
    ω2 = exp(-2π*im/3) 
    ω3 = exp(2π*im/3)

    H[[0,0,0]] = [E0 0;0 E0]

    C3 = [  0 -1 0;
            1 -1 0;
            0  0 1   
        ]
    R1 = [1,0,0]
    R2 = C3 * R1
    R3 = C3 * R2
    ez = [0,0,1]

    t = [   t0-u0*im    ω1*t1;
            ω1*t1      t0+u0*im]
    H[R1] += t
    H[-R1] +=t'

    t = [   t0-u0*im    ω2*t1;
            ω3*t1      t0+u0*im]
    H[R2] += t
    H[-R2] +=t'

    t = [   t0-u0*im    ω3*t1;
            ω2*t1      t0+u0*im]
    H[R3] +=t
    H[-R3] +=t'

    t = [   g0-g0z*im    0;
            0        g0+g0z*im]
    H[ez] +=t
    H[-ez] +=t'

    t = [   -u1/2*im    ω1*g1/2;
            ω1*g1/2    u1/2*im]
    H[R1+ez] += t
    H[-R1-ez] +=t'
    H[R1-ez] +=t
    H[-R1+ez] +=t'

    t = [   -u1/2*im    ω2*g1/2;
            ω3*g1/2    u1/2*im]
    H[R2+ez] +=t
    H[-R2-ez] +=t'
    H[R2-ez] +=t
    H[-R2+ez] +=t'

    t = [   -u1/2*im    ω3*g1/2;
            ω2*g1/2    u1/2*im]
    H[R3+ez] +=t
    H[-R3-ez] +=t'
    H[R3-ez] +=t
    H[-R3+ez] +=t'

    k = [0.72,-0.64,0.05];
    
    function hk_test(k)
        q = 2π*[k[1],k[2],0]
        kz = 2π*k[3]
        C = cos(q⋅R1)+cos(q⋅R2)+cos(q⋅R3)
        S = sin(q⋅R1)+sin(q⋅R2)+sin(q⋅R3)
        Cm = ω1*cos(q⋅R1)+ω2*cos(q⋅R2)+ω3*cos(q⋅R3)
        Cp = ω1*cos(q⋅R1)+ω3*cos(q⋅R2)+ω2*cos(q⋅R3) # ==conj(Cm)
        Cz = cos(kz)
        Sz = sin(kz)
        hp = E0 +2t0*C +2u0*S +2g0*Cz +2u1*S*Cz +2g0z*Sz
        hm = E0 +2t0*C -2u0*S +2g0*Cz -2u1*S*Cz -2g0z*Sz
        hpm = 2t1*Cm +2g1*Cm*Cz
        hmp = 2t1*Cp +2g1*Cp*Cz
        return [hp hpm; hmp hm]
    end

    N = 10
    for kx in range(0,1,N),ky in range(0,1,N),kz in range(0,1,N)
        k = [kx,ky,kz]
        bloch_hamiltonian!(H,k)
        nn = norm(H.Hk-hk_test(k))
        @test nn < 5E-16
    end
    for kx in range(0,1,N),ky in range(0,1,N),kz in range(0,1,N)
        k = [kx,ky,kz]
        Hk = H(k)
        nn = norm(Hk-hk_test(k))
        @test nn < 5E-16
    end
    @test number_of_bands(H) == 2
end