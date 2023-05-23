#################################################################
# Refer Duru, Virta 2014
# Two layer linear elastic wave propagation:
# Contains the functions that describes the material parameters
#################################################################


include("../include.jl")

using FillArrays
using LazyArrays

## Material parameters

const E = 1e9;
const ν = 0.33;
const E′ = 0.5e9;
const ν′ = 0.33;

const μ = E/(2*(1+ν));
const λ = E*ν/((1+ν)*(1-2ν));
const μ′ = E′/(2*(1+ν′));
const λ′ = E′*ν′/((1+ν′)*(1-2ν′));

const ρ = 922.5
const ρ′ = 922.5

const c₁₁ = c₂₂ = 2μ+λ
const c₃₃ = μ
const c₁₂ = λ

const c₁₁′ = c₂₂′ = 2μ′+λ′
const c₃₃′ = μ′
const c₁₂′ = λ′

# Constructs a Lazy diagonal matrix depending on the length of the grid
grid_function(x,y,v) = Diagonal(Fill(v, length(x)*length(y)));

# Constructs a Lazy version of the material properties using the diagonal matrices of the scalar matrices.
A(x, y) = Vcat(Hcat(grid_function(x,y,c₁₁), grid_function(x,y,0.0)), Hcat(grid_function(x,y,0.0), grid_function(x,y,c₃₃)));
B(x, y) = Vcat(Hcat(grid_function(x,y,c₃₃), grid_function(x,y,0.0)), Hcat(grid_function(x,y,0.0), grid_function(x,y,c₂₂)));
C(x, y) = Vcat(Hcat(grid_function(x,y,0.0), grid_function(x,y,c₁₂)), Hcat(grid_function(x,y,c₃₃), grid_function(x,y,0.0)));
Cᵀ(x, y) = Vcat(Hcat(grid_function(x,y,0.0), grid_function(x,y,c₃₃)), Hcat(grid_function(x,y,c₁₂), grid_function(x,y,0.0)));

A′(x, y) = Vcat(Hcat(grid_function(x,y,c₁₁′), grid_function(x,y,0.0)), Hcat(grid_function(x,y,0.0), grid_function(x,y,c₃₃′)));
B′(x, y) = Vcat(Hcat(grid_function(x,y,c₃₃′), grid_function(x,y,0.0)), Hcat(grid_function(x,y,0.0), grid_function(x,y,c₂₂′)));
C′(x, y) = Vcat(Hcat(grid_function(x,y,0.0), grid_function(x,y,c₁₂′)), Hcat(grid_function(x,y,c₃₃′), grid_function(x,y,0.0)));
C′ᵀ(x, y) = Vcat(Hcat(grid_function(x,y,0.0), grid_function(x,y,c₃₃′)), Hcat(grid_function(x,y,c₁₂′), grid_function(x,y,0.0)));

## Coordinate transforms
x(q,r) = q;
y(q,r) = r;
q(x,y) = x;
r(x,y) = y;

qx(x,y) = ForwardDiff.derivative(t->q(t,y), x);
qy(x,y) = ForwardDiff.derivative(t->q(x,t), y);
rx(x,y) = ForwardDiff.derivative(t->r(t,y), x);
ry(x,y) = ForwardDiff.derivative(t->r(x,t), y);

