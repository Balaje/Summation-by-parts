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
"""
Object to store the Grid Function at the discrete Points
"""
struct GridFunction{T<:Number} 
  X::AbstractVecOrMat{Tuple{T,T}}
  V::AbstractVecOrMat{T}
end
function GridFunction(f::Function, x::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  fx = BroadcastArray(f, x)
  GridFunction(x, fx)
end

# These are the material property functions. 
# In this case we assume orthotropic, anisotropic media
A₁₁(x) = c₁₁; A₁₂(x) = 0.0; A₂₁(x) = 0.0; A₂₂(x) = c₃₃;
B₁₁(x) = c₃₃; B₁₂(x) = 0.0; B₂₁(x) = 0.0; B₂₂(x) = c₂₂;
C₁₁(x) = 0.0; C₁₂(x) = c₁₂; C₂₁(x) = c₃₃; C₂₂(x) = 0.0;

A′₁₁(x) = c₁₁′; A′₁₂(x) = 0.0; A′₂₁(x) = 0.0; A′₂₂(x) = c₃₃′;
B′₁₁(x) = c₃₃′; B′₁₂(x) = 0.0; B′₂₁(x) = 0.0; B′₂₂(x) = c₂₂′;
C′₁₁(x) = 0.0; C′₁₂(x) = c₁₂′; C′₂₁(x) = c₃₃′; C′₂₂(x) = 0.0;

""" 
Material property matrix of the first layer
"""
function A(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(A₁₁, X).V) spdiagm(GridFunction(A₁₂, X).V);  
    spdiagm(GridFunction(A₂₁, X).V) spdiagm(GridFunction(A₂₂, X).V)]
  dropzeros(X)
end
function B(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(B₁₁, X).V) spdiagm(GridFunction(B₁₂, X).V);  
    spdiagm(GridFunction(B₂₁, X).V) spdiagm(GridFunction(B₂₂, X).V)]
  dropzeros(X)
end
function C(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(C₁₁, X).V) spdiagm(GridFunction(C₁₂, X).V);  
    spdiagm(GridFunction(C₂₁, X).V) spdiagm(GridFunction(C₂₂, X).V)]
  dropzeros(X)
end

""" 
Material property matrix of the second layer
"""
function A′(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(A′₁₁, X).V) spdiagm(GridFunction(A′₁₂, X).V);  
    spdiagm(GridFunction(A′₂₁, X).V) spdiagm(GridFunction(A′₂₂, X).V)]
  dropzeros(X)
end
function B′(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(B′₁₁, X).V) spdiagm(GridFunction(C′₁₂, X).V);  
    spdiagm(GridFunction(C′₂₁, X).V) spdiagm(GridFunction(C′₂₂, X).V)]
  dropzeros(X)
end
function C′(X::AbstractVecOrMat{Tuple{T,T}}) where T<:Number
  X = [spdiagm(GridFunction(C′₁₁, X).V) spdiagm(GridFunction(C′₁₂, X).V);  
    spdiagm(GridFunction(C′₂₁, X).V) spdiagm(GridFunction(C′₂₂, X).V)]
  dropzeros(X)
end