# Library

using BigRiverSchneider
using Test
using LinearAlgebra, Statistics, Random, DelimitedFiles

@testset "BigRiverSchneider.jl" begin
    @testset "Principal Component Analysis (pca)" begin include("pca_test.jl") end
    @testset "Penalized Matrix Decomposition (pmd)" begin include("pmd_test.jl") end
    @testset "Sparse Principal Component Analysis (spc)" begin include("spc_test.jl") end
end


