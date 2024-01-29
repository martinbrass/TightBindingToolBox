using TightBindingToolBox, LinearAlgebra, Plots

H = TB_Hamiltonian{ComplexF64,Int}(2,3)

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

add_hoppings!(H,[0,0,0],[E0 0;0 E0])

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
add_hoppings!(H,R1,t)
add_hoppings!(H,-R1,t')

t = [   t0-u0*im    ω2*t1;
        ω3*t1      t0+u0*im]
add_hoppings!(H,R2,t)
add_hoppings!(H,-R2,t')

t = [   t0-u0*im    ω3*t1;
        ω2*t1      t0+u0*im]
add_hoppings!(H,R3,t)
add_hoppings!(H,-R3,t')

t = [   g0-g0z*im    0;
        0        g0+g0z*im]
add_hoppings!(H,ez,t)
add_hoppings!(H,-ez,t')

t = [   -u1/2*im    ω1*g1/2;
         ω1*g1/2    u1/2*im]
add_hoppings!(H,R1+ez,t)
add_hoppings!(H,-R1-ez,t')
add_hoppings!(H,R1-ez,t)
add_hoppings!(H,-R1+ez,t')

t = [   -u1/2*im    ω2*g1/2;
         ω3*g1/2    u1/2*im]
add_hoppings!(H,R2+ez,t)
add_hoppings!(H,-R2-ez,t')
add_hoppings!(H,R2-ez,t)
add_hoppings!(H,-R2+ez,t')

t = [   -u1/2*im    ω3*g1/2;
         ω2*g1/2    u1/2*im]
add_hoppings!(H,R3+ez,t)
add_hoppings!(H,-R3-ez,t')
add_hoppings!(H,R3-ez,t)
add_hoppings!(H,-R3+ez,t')

k = [0.72,-0.64,0.05];
Hk = bloch_hamiltonian(H,k);

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
#=
norm(Hk-hk_test(k))
N = 100
for kx in range(0,1,N),ky in range(0,1,N),kz in range(0,1,N)
    k = [kx,ky,kz]
    bloch_hamiltonian!(H,k,Hk)
    nn = norm(Hk-hk_test(k))
    if nn > 5E-16
        println(nn)
    end
end
=#

kpath = [[0,0,0], [1/2,-1/2,0],[2/3,-1/3,0],[0,0,0],[0,0,1/2],[1/2,-1/2,1/2],[2/3,-1/3,1/2],[0,0,1/2]]
label = [     "Γ",         "M",         "K",     "Γ",      "A",        "L",           "H",        "A"]
nkpts = 200
plot_Bandstructure(H,kpath,nkpts,label)

##
r = 1E-2; integrate_berry_curvature_sphere(H,1,[0,0,1/2],r,1*200,1*200)/2π

##
H = TB_Hamiltonian{ComplexF64,Int}(4,3)

id = [1. 0; 0 1]
λ = 6E-3
lz = [1. 0; 0 -1]
σz = [1. 0; 0 -1]

t = kron([E0 0;0 E0],id) + λ/2 * kron(lz,σz)
add_hoppings!(H,[0,0,0],t)

t = kron([   t0-u0*im    ω1*t1;
        ω1*t1      t0+u0*im],id)
add_hoppings!(H,R1,t)
add_hoppings!(H,-R1,t')

t = kron([   t0-u0*im    ω2*t1;
        ω3*t1      t0+u0*im],id)
add_hoppings!(H,R2,t)
add_hoppings!(H,-R2,t')

t = kron([   t0-u0*im    ω3*t1;
        ω2*t1      t0+u0*im],id)
add_hoppings!(H,R3,t)
add_hoppings!(H,-R3,t')

t = kron([   g0-g0z*im    0;
        0        g0+g0z*im],id)
add_hoppings!(H,ez,t)
add_hoppings!(H,-ez,t')

t = kron([   -u1/2*im    ω1*g1/2;
         ω1*g1/2    u1/2*im],id)
add_hoppings!(H,R1+ez,t)
add_hoppings!(H,-R1-ez,t')
add_hoppings!(H,R1-ez,t)
add_hoppings!(H,-R1+ez,t')

t = kron([   -u1/2*im    ω2*g1/2;
         ω3*g1/2    u1/2*im],id)
add_hoppings!(H,R2+ez,t)
add_hoppings!(H,-R2-ez,t')
add_hoppings!(H,R2-ez,t)
add_hoppings!(H,-R2+ez,t')

t = kron([   -u1/2*im    ω3*g1/2;
         ω2*g1/2    u1/2*im],id)
add_hoppings!(H,R3+ez,t)
add_hoppings!(H,-R3-ez,t')
add_hoppings!(H,R3-ez,t)
add_hoppings!(H,-R3+ez,t')

plot_Bandstructure(H,kpath,nkpts,label)
##
r = 5E-1; integrate_berry_curvature_sphere(H,1,[0,0,1/2],r,4*200,4*200)/2π

Hk =bloch_hamiltonian(H,[0.1,-0.1,1/2])
LAPACK.syev!('N','U',Hk)

##
l = 2
od = map(m->sqrt((l-m)*(l+m+1)),l-1:-1:-l)
Lp = Bidiagonal(zeros(2l+1),od,:U)
Lm = Lp'
Lz = Diagonal(map(m->m,l:-1:-l))
l = 1/2
od = map(m->sqrt((l-m)*(l+m+1)),l-1:-1:-l)
Sp = Bidiagonal(zeros(2),od,:U)
Sm = Sp'
Sz = Diagonal(map(m->m,l:-1:-l))

SOC = λ/2 * ((kron(Lp,Sm)+kron(Lm,Sp))/2 + kron(Lz,Sz))