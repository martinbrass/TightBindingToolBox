# This tutorial is related to Phys. Rev. B 109, 085103 (2024)
# It explains how to:
#   import a tight binding Hamiltonian from FPLO
#   plot Fermi arcs on a surface
#   find Weyl points and calculate their chern number

using TightBindingToolBox

# import the tight binding Hamiltonian from +hamdata file of FPLO:
filename = "/home/brass/Projekte/LK99/TB_Ham_2bands/+hamdata"
H = FPLO_import_TB(filename);

# define k-path in brillouin zone:
klabel = ["Γ",         "M",        "K",      "Γ",     "A",        "L",           "H",        "A"]
kpts = [[0,0,0], [1/2,-1/2,0],[2/3,-1/3,0],[0,0,0],[0,0,1/2],[1/2,-1/2,1/2],[2/3,-1/3,1/2],[0,0,1/2]]
nkpts = 200
kpath = path_in_bz(kpts,nkpts)

# calculate bandstructure:
bands = bandstructure(H,kpath)

# plot bands with CairoMakie:
using CairoMakie, LaTeXStrings

vertical_lines = [1+i*nkpts for i =0:length(kpts)-1]
fig = Figure()
ax = Axis(fig[1,1];
        limits=((1,nkpts*(length(kpts)-1)+1),nothing),
        xticks=(vertical_lines,klabel),
        ylabel="energy (eV)")
for b in eachcol(bands)
    lines!(ax,b;color=:darkred)
end

vlines!(ax,vertical_lines;color=:black) 
display(fig)

## calculate Fermi arcs on a surface in reciprocal space:
# vector orthogonal to the surface:
b_normal = [1,0,0] 
# vectors spanning the surface:
b1 = [0,1,0]
b2 = [0,0,1]


n_super_layer = 12*8   # number of layers of the slab:
n_kpts  = 200          # number of k-points for each direction b1,b2
μ =  0.0 +im*1E-3      # Fermi energy plus small imaginary part

A = surface_spectral_density(H,μ,b1,b2,b_normal,n_super_layer,n_kpts,-1/2,0);

# plot with CairoMakie:
r1 = range(-1/2,1/2,n_kpts);
r2 = range(0,1,n_kpts);
fig = Figure(size=(500,500))
ax = Axis(fig[1,1];aspect=1,
    xlabel=L"$k_y$ $(2\pi/a)$",
    ylabel=L"$k_z$ $(2\pi/c)$"
)
heatmap!(ax,r1,r2,A,colormap=:linear_kryw_0_100_c71_n256)
display(fig)

## symmetry considerations tell us that there are Weyl points at Γ and A
# we calculate their Chern numbers:
using Integrals

S = TightBindingToolBox.Sphere([0,0,0],1E-1)
Ω = BerryCurvature(H,1)

dom =([0.0, 0.0],[2π,π])
prob = IntegralProblem(flux,dom,(Ω,S))    
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
println("Chern number of Weyl point at Γ is $(sol.u)")

S = TightBindingToolBox.Sphere([0,0,1/2],1E-1)
Ω = BerryCurvature(H,1)

dom =([0.0, 0.0],[2π,π])
prob = IntegralProblem(flux,dom,(Ω,S))    
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
println("Chern number of Weyl point at A is $(sol.u)")

# let's do a case with spin orbit coupling:
filename = "/home/brass/Projekte/LK99/TB_Ham_4spinbands/+hamdata"
H = FPLO_import_TB(filename);
bands = bandstructure(H,kpath)

# plot bands with CairoMakie:
using CairoMakie, LaTeXStrings

vertical_lines = [1+i*nkpts for i =0:length(kpts)-1]
fig = Figure()
ax = Axis(fig[1,1];
        limits=((1,nkpts*(length(kpts)-1)+1),nothing),
        xticks=(vertical_lines,klabel),
        ylabel="energy (eV)")
for b in eachcol(bands)
    lines!(ax,b;color=:darkred)
end

vlines!(ax,vertical_lines;color=:black) 
display(fig)

## from the bandstructure we can see that there are
# band crossings along Γ → A in all bands
# we want to see if they are Weyl points and find their positions

start_points = [[0,0,1/4]]
band_index = 2 # search for crossings in band number 2

# search_weyl_points evolves each start point to a sink and a source of Berry curvature
# hence here we find two Weyl points:
weyls = search_weyl_points(H,band_index,start_points)

# now we check their chern numbers:
Ω = BerryCurvature(H,band_index)
S = TightBindingToolBox.Sphere(weyls[1],1E-2)
dom =([0.0, 0.0],[2π,π])
prob = IntegralProblem(flux,dom,(Ω,S))    
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
println("Weyl point at $(round.(weyls[1];digits=2)) has chern number $(round(sol.u))")
S = TightBindingToolBox.Sphere(weyls[2],1E-2)
prob = IntegralProblem(flux,dom,(Ω,S))    
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
println("Weyl point at $(round.(weyls[2];digits=2)) has chern number $(round(sol.u))")
# indeed the 1st is a source and the 2nd a sink of berry curvature

## surface spin density:
using LinearAlgebra, WignerSymbols
# we define the spin operator for our bands:
l=2;s=1//2;j=5//2; # quantum numbers of the bands:
od = [  clebschgordan(l,-1,s,-s,j,-3//2)*clebschgordan(l,-1,s,s,j,-1//2),
        clebschgordan(l,0,s,-s,j,-1//2)*clebschgordan(l,0,s,s,j,1//2),
        clebschgordan(l,1,s,-s,j,1//2)*clebschgordan(l,1,s,s,j,3//2)
     ]
Sp = Bidiagonal(zeros(4),od,:L)
Sm = Sp'
Sx = 1/2 * (Sp+Sm)
Sy = -im/2 * (Sp-Sm)

Sz =Diagonal([
    clebschgordan(l,-2,s,s,j,-3//2)^2-clebschgordan(l,-1,s,-s,j,-3//2)^2,
    clebschgordan(l,-1,s,s,j,-1//2)^2-clebschgordan(l, 0,s,-s,j,-1//2)^2,
    clebschgordan(l, 0,s,s,j, 1//2)^2-clebschgordan(l, 1,s,-s,j, 1//2)^2,
    clebschgordan(l, 1,s,s,j, 3//2)^2-clebschgordan(l, 2,s,-s,j, 3//2)^2
] ./2)

S = [Sx,Sy,Sz];

# define the surface as above:
b1 = [0,1,0]
b2 = [0,0,1]
b_normal = [1,0,0]

n_super_layer = 12*8    # number of layers of the slab:
n_kpts  = 200           # number of k-points for each direction b1,b2
μ = -0.055 + im*1E-4    # energy at which to plot with small imaginary part


A = @time surface_spectral_spin_density(H,S,μ,b1,b2,b_normal,n_super_layer,n_kpts,0,-1/2);

# plotting:
using ImageUtils
# to use complex coloring scheme we map the y and z component
# of the spin density to complex numbers:
Z = A[2,:,:] + im*A[3,:,:]
col = complexColoring(Z;amax=1E2)

# plot with CairoMakie:
r1 = range(-1/2,1/2,n_kpts);
r2 = range(0,1,n_kpts);
fig = Figure(size=(500,500))
ax = Axis(fig[1,1];aspect=1,
    xlabel=L"$k_y$ $(2\pi/a)$",
    ylabel=L"$k_z$ $(2\pi/c)$"
)
heatmap!(ax,r1,r2,col)
display(fig)
