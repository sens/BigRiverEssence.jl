# Library

import MultivariateStats
const MVS = MultivariateStats
using BigRiverSchneider
using Test
using LinearAlgebra, Statistics, Random, DelimitedFiles


@testset "BigRiverSchneider.jl" begin
    @testset "Principal Component Analysis (pca)" begin include("pca_test.jl") end
    @testset "Penalized Matrix Decomposition (pmd)" begin include("pmd_test.jl") end
    @testset "Sparse Principal Component Analysis (spc)" begin include("spc_test.jl") end
    @testset "Kernel Partial Least Squares (plskern)" begin include("plskern_test.jl") end
    @testset "Joint and Individual Variation Explained (jive)" begin include("jive_test.jl") end
    @testset "Sparse Partial Least Squares Discriminant Analysis (splsda)" begin include("splsda_test.jl") end
    @testset "Canonical Correlation Analysis (cca)" begin include("cca_test.jl") end
    @testset "Sparse Canonical Correlation Analysis (scca)" begin include("scca_test.jl") end
end


