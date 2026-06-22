# test/TestJiveRjiveFull.jl — comprehensive jive_rjive validation on SIMULATED data.
# Tests: (A) given-ranks similarity+speed vs r.jive, (B) ground-truth recovery,
#        (C) auto-ranks similarity+speed vs r.jive, (D) do given & auto agree.
using BigRiverSchneider, RCall, BenchmarkTools
using LinearAlgebra, Statistics, Random
Random.seed!(2024)

# ----------------------------------------------------------------
# Simulated data with KNOWN structure (supplement §6.2 generative model).
# Moderate size so the permutation path runs in reasonable time.
# Two datasets, shared joint scores S, individual scores S1/S2.
# ----------------------------------------------------------------
n = 80                       # samples (shared columns)
rT, r1T, r2T = 2, 3, 3       # TRUE ranks
p1, p2 = 60, 50              # variables per dataset
S  = randn(rT, n)
U1 = randn(p1, rT); U2 = randn(p2, rT)
S1 = randn(r1T, n); W1 = randn(p1, r1T)
S2 = randn(r2T, n); W2 = randn(p2, r2T)
X1 = U1*S + W1*S1 .+ 0.3 .* randn(p1, n)     # mild noise (realistic; perm test needs some)
X2 = U2*S + W2*S2 .+ 0.3 .* randn(p2, n)
nm = ["Dataset1","Dataset2"]
println("Simulated: X1 $(size(X1)), X2 $(size(X2)); true ranks joint=$rT, indiv=[$r1T,$r2T]\n")

# push to R once
@rput X1 X2

# helper: variance explained
ve(J,A,D) = (norm(J)^2/norm(D)^2, norm(A)^2/norm(D)^2, norm(D.-J.-A)^2/norm(D)^2)  # returns a tuple of (joint VE, indiv VE, residual VE) for a given dataset, where J is the joint structure, A is the individual structure, and D is the original data; this allows us to compare the variance explained by the joint and individual components in both our implementation and r.jive's implementation against the original data
# helper: joint subspace basis
jb(b...) = Matrix(qr(svd(vcat(b...)).Vt[1:2,:]').Q)[:,1:2]  # computes the joint subspace basis from the joint structures of both datasets; this allows us to compare the joint subspaces obtained from our implementation and r.jive's implementation by computing the canonical correlation between their joint subspace bases, which is a robust way to compare subspaces even if the ranks differ slightly due to estimation variability

# ================================================================
# PART A — GIVEN RANKS: similarity to r.jive
# ================================================================
println("="^66)
println("PART A — GIVEN RANKS (2, [3,3]): jive_rjive vs r.jive")
println("="^66)

resA = jive_rjive([X1,X2], rT, [r1T,r2T])
R"""
fitA <- jive(list(X1,X2), rankJ=2, rankA=c(3,3), method="given",
             scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE)
J1A<-fitA$joint[[1]]; J2A<-fitA$joint[[2]]
A1A<-fitA$individual[[1]]; A2A<-fitA$individual[[2]]
d1A<-fitA$data[[1]]; d2A<-fitA$data[[2]]
"""
@rget J1A J2A A1A A2A d1A d2A
JrA=[J1A,J2A]; ArA=[A1A,A2A]; DrA=[d1A,d2A]

println("\ninput match (scaled data, want 0):")
nelA=[p1*n, p2*n]; sumnA=sum(nelA)  # r.jive's scaling factor is the Frobenius norm of the full stacked data, which is sqrt(sum of squares of all elements) = sqrt(sum of (pᵢ*n) for i=1 to k) = sqrt(sumnA)
XcA = [ let Xi=X.-mean(X,dims=2); Xi./(norm(Xi)*sqrt(sumnA)); end for X in (X1,X2) ] # row-center + r.jive scaling of the original data, which is what r.jive uses as the input to its algorithm; we compare this to the scaled data that our algorithm uses internally to ensure that we are starting from the same point before the decomposition
for i in 1:2
    println("  $(nm[i]): ", round(norm(XcA[i].-DrA[i]),digits=10))
end
println("\nJ / A differences (want 0):")
for i in 1:2
    println("  $(nm[i]): ‖J diff‖=", round(norm(resA.J[i].-JrA[i]),digits=8),
            "  ‖A diff‖=", round(norm(resA.A[i].-ArA[i]),digits=8))
end
println("\nvariance explained:")
for i in 1:2
    println("  $(nm[i]): yours ", round.(ve(resA.J[i],resA.A[i],DrA[i]),digits=4),
            "  r.jive ", round.(ve(JrA[i],ArA[i],DrA[i]),digits=4))
end
println("\njoint subspace canon corr: ", round.(svd(jb(resA.J...)'*jb(JrA...)).S, digits=6))

# ================================================================
# PART B — GROUND TRUTH: does given-ranks recover planted structure?
# (use noiseless version for exact check)
# ================================================================
println("\n", "="^66)
println("PART B — GROUND TRUTH (noiseless, true ranks)")
println("="^66)
X1n = U1*S + W1*S1; X2n = U2*S + W2*S2          # noiseless
resB = jive_rjive([X1n,X2n], rT, [r1T,r2T]; scale=false)
Gc = [X1n .- mean(X1n,dims=2), X2n .- mean(X2n,dims=2)]
recon = sum(norm(Gc[i].-resB.J[i].-resB.A[i])^2 for i in 1:2)
ortho = norm(vcat(resB.J...)*resB.A[1]') + norm(vcat(resB.J...)*resB.A[2]')
Strue = S .- mean(S,dims=2)
Qt = Matrix(qr(Strue').Q)[:,1:rT]
Qm = Matrix(qr(svd(vcat(resB.J...)).Vt[1:rT,:]').Q)[:,1:rT]
println("  reconstruction ‖X−(J+A)‖² : ", round(recon, digits=10), "  (want ≈0)")
println("  orthogonality  ‖J·Aᵀ‖     : ", round(ortho, digits=10), "  (want ≈0)")
println("  joint subspace vs truth   : ", round.(svd(Qm'*Qt).S, digits=6), "  (want ≈1)")

# ================================================================
# PART C — AUTO RANKS: does permutation find the true ranks, and
#          do yours and r.jive agree on the estimated ranks?
# ================================================================
println("\n", "="^66)
println("PART C — AUTO RANKS via permutation")
println("="^66)
Random.seed!(2024)
resC = jive_rjive([X1,X2]; nperm=100)
println("\n  yours  estimated: joint ", resC.r, ", indiv ", resC.ri)
R"""
set.seed(2024)
fitC <- jive(list(X1,X2), method="perm", est=TRUE, orthIndiv=TRUE, showProgress=FALSE)
rJ_r <- fitC$rankJ; rA_r <- fitC$rankA
J1C<-fitC$joint[[1]]; J2C<-fitC$joint[[2]]
A1C<-fitC$individual[[1]]; A2C<-fitC$individual[[2]]
d1C<-fitC$data[[1]]; d2C<-fitC$data[[2]]
"""
@rget rJ_r rA_r J1C J2C A1C A2C d1C d2C
println("  r.jive estimated: joint ", Int(rJ_r), ", indiv ", Int.(rA_r))
println("  true ranks were : joint ", rT, ", indiv [", r1T, ",", r2T, "]")

# if both estimated the SAME ranks, compare their decompositions too
if resC.r == Int(rJ_r) && resC.ri == Int.(rA_r)
    println("\n  → same ranks estimated; comparing decompositions:")
    JrC=[J1C,J2C]; ArC=[A1C,A2C]; DrC=[d1C,d2C]
    for i in 1:2
        println("    $(nm[i]): ‖J diff‖=", round(norm(resC.J[i].-JrC[i]),digits=6),
                "  ‖A diff‖=", round(norm(resC.A[i].-ArC[i]),digits=6))
    end
    println("    joint subspace canon corr: ", round.(svd(jb(resC.J...)'*jb(JrC...)).S, digits=6))
else
    println("\n  → ranks differ (permutation is statistical; RNG differs across languages).")
    println("    comparing joint SUBSPACES instead (robust to rank diff):")
    println("    canon corr: ", round.(svd(jb(resC.J...)'*jb(J1C,J2C)).S, digits=6))
end

# ================================================================
# PART D — SPEED: both paths, vs r.jive
# ================================================================
println("\n", "="^66)
println("PART D — TIMING")
println("="^66)

println("\n[given ranks]")
print("  jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2], $rT, [$r1T,$r2T]);
R"""
library(microbenchmark)
mbG <- microbenchmark(
  jive(list(X1,X2), rankJ=2, rankA=c(3,3), method="given",
       scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=20)
cat("  r.jive (R)        :  median", round(median(mbG$time)/1e6,2), "ms\n")
"""

println("\n[auto ranks — permutation, fewer reps since it's slow]")
print("  jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2]; nperm=100) samples=3 evals=1;
R"""
mbP <- microbenchmark(
  jive(list(X1,X2), method="perm", est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=3)
cat("  r.jive (R)        :  median", round(median(mbP$time)/1e6,2), "ms\n")
"""




#=
==================================================================
PART A — GIVEN RANKS (2, [3,3]): jive_rjive vs r.jive
==================================================================

input match (scaled data, want 0):
  Dataset1: 0.0
  Dataset2: 0.0

J / A differences (want 0):
  Dataset1: ‖J diff‖=0.0  ‖A diff‖=0.0
  Dataset2: ‖J diff‖=0.0  ‖A diff‖=0.0

variance explained:
  Dataset1: yours (0.536, 0.4235, 0.0405)  r.jive (0.536, 0.4235, 0.0405)
  Dataset2: yours (0.4409, 0.502, 0.0571)  r.jive (0.4409, 0.502, 0.0571)

joint subspace canon corr: [1.0, 1.0]

==================================================================
PART B — GROUND TRUTH (noiseless, true ranks)
==================================================================
  reconstruction ‖X−(J+A)‖² : 618.1879298603  (want ≈0)
  orthogonality  ‖J·Aᵀ‖     : 1.39549e-5  (want ≈0)
  joint subspace vs truth   : [0.999436, 0.997297]  (want ≈1)

==================================================================
PART C — AUTO RANKS via permutation
==================================================================
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]

  yours  estimated: joint 2, indiv [3, 3]
  r.jive estimated: joint 2, indiv [3, 3]
  true ranks were : joint 2, indiv [3,3]

  → same ranks estimated; comparing decompositions:
    Dataset1: ‖J diff‖=0.0  ‖A diff‖=0.0
    Dataset2: ‖J diff‖=0.0  ‖A diff‖=0.0
    joint subspace canon corr: [1.0, 1.0]

==================================================================
PART D — TIMING
==================================================================

[given ranks]
  jive_rjive (Julia):   39.517 ms (4571 allocations: 52.11 MiB)
  r.jive (R)        :  median 618.81 ms

[auto ranks — permutation, fewer reps since it's slow]
  jive_rjive (Julia): Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
  714.210 ms (236660 allocations: 690.62 MiB)
  r.jive (R)        :  median 4439.36 ms
RObject{NilSxp}
NULL
=#














# test/TestJiveOpt.jl — full verification: optimized JIVE vs original vs r.jive
using BigRiverSchneider, BenchmarkTools, LinearAlgebra, Statistics, Random, RCall

# ============================================================================
# TEST DATA
# ============================================================================
Random.seed!(1234)
n = 100
X1 = randn(200, n); X2 = randn(180, n); X3 = randn(150, n)
Xs = [X1, X2, X3]
r, ri = 2, [10, 9, 8]

function compare_jive(a, b; label="")
    println("--- $label ---")
    k = length(a.J)
    for i in 1:k
        println("  J[$i] ‖diff‖     : ", norm(a.J[i] .- b.J[i]))
        println("  A[$i] ‖diff‖     : ", norm(a.A[i] .- b.A[i]))
    end
    println("  S ‖diff‖(abs)      : ", norm(abs.(a.S) .- abs.(b.S)))
    for i in 1:k
        println("  U[$i] ‖diff‖(abs): ", norm(abs.(a.U[i]) .- abs.(b.U[i])))
        println("  Si[$i] ‖diff‖(abs): ", norm(abs.(a.Si[i]) .- abs.(b.Si[i])))
        println("  Wi[$i] ‖diff‖(abs): ", norm(abs.(a.Wi[i]) .- abs.(b.Wi[i])))
    end
    println("  r match  : ", a.r == b.r)
    println("  ri match : ", a.ri == b.ri)
end

# ============================================================================
# PATH 1: GIVEN RANKS  (deterministic — should be bit-identical)
# ============================================================================
println("="^60)
println("PATH 1: GIVEN RANKS (jive_rjive vs jive_rjive_opt)")
println("="^60)
ref_given = jive_rjive(Xs, r, ri)
opt_given = jive_rjive_opt(Xs, r, ri)
compare_jive(ref_given, opt_given; label="given ranks r=$r ri=$ri")

# ============================================================================
# STRUCTURED DATA for meaningful auto-rank test (plant rank-2 joint + rank-3 individual)
# ============================================================================
Random.seed!(42)
Sjoint = randn(2, n)                                   # shared joint scores (rank 2)
X1s = randn(200,2)*Sjoint + randn(200,3)*randn(3,n)*0.5 + randn(200,n)*0.1
X2s = randn(180,2)*Sjoint + randn(180,3)*randn(3,n)*0.5 + randn(180,n)*0.1
X3s = randn(150,2)*Sjoint + randn(150,3)*randn(3,n)*0.5 + randn(150,n)*0.1
Xstruct = [X1s, X2s, X3s]

# ============================================================================
# PATH 2: AUTO RANKS — orig vs opt vs r.jive
# ============================================================================
println("\n", "="^60)
println("PATH 2: AUTO RANKS (orig vs opt vs r.jive) on STRUCTURED data")
println("="^60)

Random.seed!(999); ref_auto = jive_rjive(Xstruct)
Random.seed!(999); opt_auto = jive_rjive_opt(Xstruct)

# --- r.jive's own rank estimate on the SAME data ---
@rput X1s X2s X3s
R"""
suppressMessages(library(r.jive))
# r.jive expects a list of matrices with samples as COLUMNS (same orientation as ours)
dat <- list(t(t(X1s)), t(t(X2s)), t(t(X3s)))
set.seed(999)
fit <- jive(dat, method="perm", scale=TRUE, center=TRUE, conv=1e-6, maxiter=1000, showProgress=FALSE)
rjoint <- fit$rankJ
rindiv <- fit$rankA
"""
@rget rjoint rindiv
rjoint = Int(rjoint); rindiv = Int.(vec(rindiv))

println("\nEstimated ranks:")
println("  orig  : r=$(ref_auto.r), ri=$(ref_auto.ri)")
println("  opt   : r=$(opt_auto.r), ri=$(opt_auto.ri)")
println("  r.jive: r=$(rjoint), ri=$(rindiv)")
println("  orig == opt    : ", ref_auto.r==opt_auto.r && ref_auto.ri==opt_auto.ri)
println("  orig == r.jive : ", ref_auto.r==rjoint && ref_auto.ri==rindiv)
println("  opt  == r.jive : ", opt_auto.r==rjoint && opt_auto.ri==rindiv)

if ref_auto.r == opt_auto.r && ref_auto.ri == opt_auto.ri
    compare_jive(ref_auto, opt_auto; label="auto ranks orig vs opt (seeded)")
else
    println("  ⚠ orig/opt ranks differ — decomposition comparison skipped")
end

# ============================================================================
# BENCHMARKS
# ============================================================================
println("\n", "="^60)
println("BENCHMARK: time / allocations / memory")
println("="^60)

println("\n[given ranks]")
print("orig: "); @btime jive_rjive($Xs, $r, $ri);
print("opt : "); @btime jive_rjive_opt($Xs, $r, $ri);

println("\n[auto ranks — permutation, nperm=100, structured data]")
print("orig: "); @btime jive_rjive($Xstruct);
print("opt : "); @btime jive_rjive_opt($Xstruct);
#=
============================================================
PATH 1: GIVEN RANKS (jive_rjive vs jive_rjive_opt)
============================================================
--- given ranks r=2 ri=[10, 9, 8] ---
  J[1] ‖diff‖     : 2.614023243431123e-14
  A[1] ‖diff‖     : 5.948079138937322e-14
  J[2] ‖diff‖     : 2.644782324513928e-14
  A[2] ‖diff‖     : 7.084152062206963e-14
  J[3] ‖diff‖     : 2.9317013112982227e-14
  A[3] ‖diff‖     : 6.32646113649136e-14
  S ‖diff‖(abs)      : 3.3089856228793285e-11
  U[1] ‖diff‖(abs): 1.730305692050367e-14
  Si[1] ‖diff‖(abs): 1.0072814930791707e-10
  Wi[1] ‖diff‖(abs): 6.347388569932987e-14
  U[2] ‖diff‖(abs): 1.8448434689245185e-14
  Si[2] ‖diff‖(abs): 1.403601620662132e-10
  Wi[2] ‖diff‖(abs): 9.300263759721589e-14
  U[3] ‖diff‖(abs): 2.0133390668712735e-14
  Si[3] ‖diff‖(abs): 2.3141443455843017e-10
  Wi[3] ‖diff‖(abs): 1.513037261755589e-13
  r match  : true
  ri match : true

============================================================
PATH 2: AUTO RANKS (orig vs opt vs r.jive) on STRUCTURED data
============================================================
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]

Estimated ranks:
  orig  : r=2, ri=[3, 3, 3]
  opt   : r=2, ri=[3, 3, 3]
  r.jive: r=2, ri=[3, 3, 3]
  orig == opt    : true
  orig == r.jive : true
  opt  == r.jive : true
--- auto ranks orig vs opt (seeded) ---
  J[1] ‖diff‖     : 1.246174139643643e-16
  A[1] ‖diff‖     : 1.262348246019643e-15
  J[2] ‖diff‖     : 1.3067749401763986e-16
  A[2] ‖diff‖     : 1.0226924030486027e-15
  J[3] ‖diff‖     : 1.2609009384206562e-16
  A[3] ‖diff‖     : 1.026764831458663e-15
  S ‖diff‖(abs)      : 4.7093121307358476e-14
  U[1] ‖diff‖(abs): 3.436649887300215e-17
  Si[1] ‖diff‖(abs): 1.0809105343392978e-12
  Wi[1] ‖diff‖(abs): 2.44503072497833e-16
  U[2] ‖diff‖(abs): 4.054554934086248e-17
  Si[2] ‖diff‖(abs): 8.955813797573476e-13
  Wi[2] ‖diff‖(abs): 4.200176663034072e-16
  U[3] ‖diff‖(abs): 4.384879231373752e-17
  Si[3] ‖diff‖(abs): 8.979171037124809e-13
  Wi[3] ‖diff‖(abs): 6.761537137011926e-16
  r match  : true
  ri match : true

============================================================
BENCHMARK: time / allocations / memory
============================================================

[given ranks]
orig:   1.242 s (37731 allocations: 871.00 MiB)
opt :   878.527 ms (10235 allocations: 277.45 MiB)

[auto ranks — permutation, nperm=100, structured data]
orig: Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
  2.709 s (687904 allocations: 1.53 GiB)
opt : Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3, 3]
  2.552 s (28496 allocations: 559.46 MiB)
=#
