
using BigRiverSchneider   # this makes the exported names from BigRiverSchneider.jl available in this test file, so we can call pca, pca_transform, etc. directly without prefixing with BigRiverSchneider.

using Random, LinearAlgebra # we need Random for seeding the random number generator, and LinearAlgebra for matrix operations in the test code.
Random.seed!(1234) 





################################### TESTING ###################################
n, p, r = 5000, 200, 10    # 5000 observations, 200 features, 10 hidden signals
latent  = randn(n, r)              #  10 latent signals (5000 × 10) (components) that we will try to recover with PCA
mixing  = randn(r, p)              # random mixing matrix (10 × 200) that mixes the latent signals into the observed features
X       = latent * mixing .+ 0.05 .* randn(n, p)    # observed data (5000 × 200) is the mixed latent signals plus some noise
println("X is $(size(X,1)) × $(size(X,2))  ($(length(X)) entries)\n")
k    = 15
msvd = BigRiverSchneider.pca(X; k = k, method = :svd)
mcov = BigRiverSchneider.pca(X; k = k, method = :cov)
println("propOFvar (svd), first $k components:")
println(round.(msvd.propOFvar, digits = 4))
println("\ncumulative variance explained:")
println(round.(cumsum(msvd.propOFvar), digits = 4))
println("\nmax |variance| difference, svd vs cov : ",maximum(abs.(msvd.variances .- mcov.variances)))

scores = BigRiverSchneider.pca_transform(msvd, X)
println("\nscores size : ", size(scores))          # (5000, 15)
 
Xhat = BigRiverSchneider.pca_invtransform(msvd, scores)
rmse = sqrt(sum(abs2, X .- Xhat) / length(X))
println("reconstruction RMSE (k = $k) : ", round(rmse, digits = 5))
 

m10    = BigRiverSchneider.pca(X; k = 10, method = :svd)
rmse10 = sqrt(sum(abs2, X .- BigRiverSchneider.pca_invtransform(m10, BigRiverSchneider.pca_transform(m10, X))) / length(X))
println("reconstruction RMSE (k = 10) : ", round(rmse10, digits = 5))

println("\n-- timing (compilation already warmed up) --")
BigRiverSchneider.pca(X; k = k, method = :svd); BigRiverSchneider.pca(X; k = k, method = :cov)   # warmup
print("svd : "); msvd = @time BigRiverSchneider.pca(X; k = k, method = :svd);  
print("cov : "); mcov = @time BigRiverSchneider.pca(X; k = k, method = :cov);




 
 
















