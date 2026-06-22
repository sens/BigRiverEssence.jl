# Library

using BigRiverSchneider
using Test
using LinearAlgebra, Statistics, Random

@testset "BigRiverSchneider.jl" begin
    @testset "PCA (pca)" begin include("pca_opt_test.jl") end
end
