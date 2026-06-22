# test/TestScca.jl — ground truth + agreement vs PMA + benchmark
using RCall, BigRiverSchneider
using LinearAlgebra, Statistics, Random
using BenchmarkTools

# ============================================================
# PART 1 — GROUND TRUTH: recover KNOWN sparse canonical structure
# ============================================================
# Build X1, X2 sharing a latent factor, where only the first few features
# of each load on it. Sparse CCA should select those features.
println("="^60); println("PART 1 — GROUND TRUTH"); println("="^60)

Random.seed!(42)
n = 100
p1, p2 = 500, 1000
nz1, nz2 = 25, 40                       # # truly-nonzero features in each view
lat = randn(n)                          # shared latent factor (length n)

# rows = obs (PMA layout). First nz features load on `lat`; rest are noise.
Xr = randn(n, p1)
Zr = randn(n, p2)
Xr[:, 1:nz1] .+= lat * (2.0 .* ones(nz1))'
Zr[:, 1:nz2] .+= lat * (2.0 .* ones(nz2))'
true_x = Set(1:nz1)
true_z = Set(1:nz2)

# scca uses columns = obs → transpose
mine = scca(Matrix(transpose(Xr)), Matrix(transpose(Zr));
            penaltyx=0.2, penaltyz=0.2, K=1, niter=15)

sel_x = Set(findall(!iszero, mine.u[:, 1]))
sel_z = Set(findall(!iszero, mine.v[:, 1]))
println("True X features: 1..$nz1 ;  True Z features: 1..$nz2")
println("Selected X: $(length(sel_x)) nonzeros, overlap with truth: $(length(intersect(sel_x, true_x)))/$nz1")
println("Selected Z: $(length(sel_z)) nonzeros, overlap with truth: $(length(intersect(sel_z, true_z)))/$nz2")
println("Fraction of selected-X that are TRUE: ", round(length(intersect(sel_x,true_x))/max(1,length(sel_x)), digits=3))
println("Fraction of selected-Z that are TRUE: ", round(length(intersect(sel_z,true_z))/max(1,length(sel_z)), digits=3))
println("Canonical correlation found: ", round(mine.cors[1], digits=4))

# ============================================================
# PART 2 — AGREEMENT vs PMA::CCA
# ============================================================
println("\n", "="^60); println("PART 2 — vs PMA::CCA"); println("="^60)

R"""
library(PMA)
set.seed(3189)
uu  <- matrix(c(rep(1,25),rep(0,75)),ncol=1)
v1  <- matrix(c(rep(1,50),rep(0,450)),ncol=1)
v2  <- matrix(c(rep(0,50),rep(1,50),rep(0,900)),ncol=1)
xpma <- uu%*%t(v1) + matrix(rnorm(100*500),ncol=500)
zpma <- uu%*%t(v2) + matrix(rnorm(100*1000),ncol=1000)
out <- CCA(xpma, zpma, typex="standard", typez="standard", K=3,
           penaltyx=0.3, penaltyz=0.3, niter=15, trace=FALSE)
ru <- out$u; rv <- out$v; rd <- out$d; rcors <- out$cors
"""
@rget xpma zpma ru rv rd rcors

# scca: column-major input
mine2 = scca(Matrix(transpose(xpma)), Matrix(transpose(zpma));
             penaltyx=0.3, penaltyz=0.3, K=3, niter=15)

for k in 1:3
    su_match = Set(findall(!iszero, mine2.u[:,k])) == Set(findall(!iszero, ru[:,k]))
    sv_match = Set(findall(!iszero, mine2.v[:,k])) == Set(findall(!iszero, rv[:,k]))
    println("comp $k | u |cor|: ", round(abs(cor(mine2.u[:,k], ru[:,k])), digits=5),
            "  v |cor|: ", round(abs(cor(mine2.v[:,k], rv[:,k])), digits=5),
            "  u-sel: ", su_match, "  v-sel: ", sv_match,
            "  cors Δ: ", round(abs(mine2.cors[k] - rcors[k]), sigdigits=3),
            "  d Δ: ", round(abs(mine2.d[k] - rd[k]), sigdigits=3))
end

# ============================================================
# PART 3 — BENCHMARK: scca vs PMA
# ============================================================
println("\n", "="^60); println("PART 3 — BENCHMARK"); println("="^60)

Xc = Matrix(transpose(xpma)); Zc = Matrix(transpose(zpma))

println("\n[K=1]")
print("mine (Julia): ")
@btime scca($Xc, $Zc; penaltyx=0.3, penaltyz=0.3, K=1, niter=15);
R"""
library(microbenchmark)
mb1 <- microbenchmark(
  CCA(xpma, zpma, typex="standard", typez="standard", K=1,
      penaltyx=0.3, penaltyz=0.3, niter=15, trace=FALSE),
  times=10)
cat("PMA  (R)    : median", round(median(mb1$time)/1e6, 3), "ms\n")
"""

println("\n[K=3]")
print("mine (Julia): ")
@btime scca($Xc, $Zc; penaltyx=0.3, penaltyz=0.3, K=3, niter=15);
R"""
mb3 <- microbenchmark(
  CCA(xpma, zpma, typex="standard", typez="standard", K=3,
      penaltyx=0.3, penaltyz=0.3, niter=15, trace=FALSE),
  times=10)
cat("PMA  (R)    : median", round(median(mb3$time)/1e6, 3), "ms\n")
"""