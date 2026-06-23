# verify_pmd_vs_r.jl — interactive check (NOT the formal test).
# (1) ground-truth: recover a planted sparse rank-2 structure
# (2) vs PMA::PMD (type="standard", explicit sumabsu & sumabsv)
# (3) benchmark Julia vs R
using BigRiverSchneider
using RCall, LinearAlgebra, Statistics, Random, BenchmarkTools

# ======================================================================
# PART 1 — GROUND TRUTH: recover a known sparse rank-2 decomposition
# ======================================================================
println("="^60)
println("PART 1 — GROUND TRUTH (planted sparse structure)")
println("="^60)

Random.seed!(42)
n, p = 80, 60
# two planted factors: sparse u (rows) and sparse v (cols), well separated
u1 = [randn(20); zeros(60)];      v1 = [randn(15); zeros(45)]
u2 = [zeros(40); randn(20); zeros(20)]; v2 = [zeros(30); randn(15); zeros(15)]
X = 6.0*(u1*v1') .+ 4.0*(u2*v2') .+ 0.3 .* randn(n, p)   # strong signal + noise
Xc = X .- mean(X)

# budgets that allow roughly the true support sizes
suf = 0.45 * sqrt(n)
svf = 0.45 * sqrt(p)
mg = pmd(Xc; sumabsu=suf, sumabsv=svf, K=2, center=false)

for (k, (ut, vt)) in enumerate(((u1,v1), (u2,v2)))
    # the planted factors are sparse; check the recovered factor aligns and
    # its support overlaps the true support.
    ua = abs(dot(mg.u[:,k] ./ norm(mg.u[:,k]), ut ./ norm(ut)))
    va = abs(dot(mg.v[:,k] ./ norm(mg.v[:,k]), vt ./ norm(vt)))
    trueu = Set(findall(!iszero, ut)); selu = Set(findall(!iszero, mg.u[:,k]))
    truev = Set(findall(!iszero, vt)); selv = Set(findall(!iszero, mg.v[:,k]))
    println("factor $k | u align=", round(ua,digits=3), "  v align=", round(va,digits=3),
            " | u recall=", round(length(intersect(selu,trueu))/length(trueu),digits=2),
            "  v recall=", round(length(intersect(selv,truev))/length(truev),digits=2))
end
println("(high alignment + recall ⇒ pmd recovers the planted sparse factors)")

# ======================================================================
# PART 2 — vs PMA::PMD  (explicit sumabsu & sumabsv; center=FALSE)
# ======================================================================
println("\n", "="^60)
println("PART 2 — Julia pmd vs PMA::PMD")
println("="^60)

Random.seed!(1234)
n2, p2 = 50, 40
X2 = randn(n2, p2)
K  = 3
su = 0.5 * sqrt(n2)
sv = 0.5 * sqrt(p2)

@rput X2 K su sv
R"""
suppressMessages(library(PMA))
Xc  <- X2 - mean(X2)                      # PMA centers by the GRAND mean
out <- PMD(Xc, type="standard", sumabsu=su, sumabsv=sv, K=K,
           center=FALSE, trace=FALSE)
u_r <- out$u; v_r <- out$v; d_r <- out$d
"""
@rget u_r v_r d_r

X2c = X2 .- mean(X2)
m   = pmd(X2c; sumabsu=su, sumabsv=sv, K=K, center=false)

println("  (n=$n2, p=$p2, K=$K, sumabsu=$(round(su,digits=3)), sumabsv=$(round(sv,digits=3)))")
for k in 1:K
    ua = abs(dot(m.u[:,k] ./ norm(m.u[:,k]), u_r[:,k] ./ norm(u_r[:,k])))
    va = abs(dot(m.v[:,k] ./ norm(m.v[:,k]), v_r[:,k] ./ norm(v_r[:,k])))
    nzuj = count(!iszero, m.u[:,k]); nzur = count(!iszero, u_r[:,k])
    nzvj = count(!iszero, m.v[:,k]); nzvr = count(!iszero, v_r[:,k])
    println("comp $k | u |cor|=", round(ua,digits=5), "  v |cor|=", round(va,digits=5),
            " | nz u: J=$nzuj R=$nzur  v: J=$nzvj R=$nzvr",
            " | d: J=", round(m.d[k],digits=4), " R=", round(d_r[k],digits=4))
end
println("\nd vector diff (‖d_j - d_r‖): ", round(norm(m.d .- d_r), sigdigits=3))

# max-budget sanity: both penalties off → rank-1 SVD on both sides
println("\n--- sumabsu=√n, sumabsv=√p (max budget → dense → ordinary SVD) ---")
sumax_u = sqrt(n2); sumax_v = sqrt(p2)
@rput sumax_u sumax_v
R"""
out2 <- PMD(X2 - mean(X2), type="standard", sumabsu=sumax_u, sumabsv=sumax_v,
            K=1, center=FALSE, trace=FALSE)
u_r1 <- out2$u[,1]; v_r1 <- out2$v[,1]
"""
@rget u_r1 v_r1
m1 = pmd(X2c; sumabsu=sumax_u, sumabsv=sumax_v, K=1, center=false)
F  = svd(X2c)
println("  Julia v vs SVD V1: ", round(abs(dot(m1.v[:,1], F.V[:,1])), digits=5),
        "   u vs SVD U1: ", round(abs(dot(m1.u[:,1], F.U[:,1])), digits=5))
println("  R     v vs SVD V1: ", round(abs(dot(v_r1 ./ norm(v_r1), F.V[:,1])), digits=5),
        "   u vs SVD U1: ", round(abs(dot(u_r1 ./ norm(u_r1), F.U[:,1])), digits=5))
println("  nz at max budget — Julia u:", count(!iszero,m1.u[:,1]), "/", n2,
        " v:", count(!iszero,m1.v[:,1]), "/", p2,
        "   R u:", count(!iszero,u_r1), "/", n2, " v:", count(!iszero,v_r1), "/", p2)

# ======================================================================
# PART 3 — BENCHMARK
# ======================================================================
println("\n", "="^60)
println("PART 3 — BENCHMARK")
println("="^60)

for (nb, pb) in ((50, 40), (200, 100))
    Random.seed!(7)
    Xb  = randn(nb, pb)
    Xbc = Xb .- mean(Xb)
    sub = 0.5 * sqrt(nb); svb = 0.5 * sqrt(pb)
    println("\n[n=$nb, p=$pb, K=3]")
    print("  Julia: ")
    @btime pmd($Xbc; sumabsu=$sub, sumabsv=$svb, K=3, center=false);
    @rput Xb sub svb
    R"""
    suppressMessages(library(microbenchmark))
    Xbc <- Xb - mean(Xb)
    mb <- microbenchmark(
        PMD(Xbc, type="standard", sumabsu=sub, sumabsv=svb, K=3,
            center=FALSE, trace=FALSE),
        times=10)
    cat("  R    :   median", round(median(mb$time)/1e6, 3), "ms\n")
    """
end

println("\n", "="^60)


#=
============================================================
PART 1 — GROUND TRUTH (planted sparse structure)
============================================================
factor 1 | u align=0.999  v align=0.999 | u recall=1.0  v recall=1.0
factor 2 | u align=0.999  v align=0.999 | u recall=1.0  v recall=1.0
(high alignment + recall ⇒ pmd recovers the planted sparse factors)

============================================================
PART 2 — Julia pmd vs PMA::PMD
============================================================
  (n=50, p=40, K=3, sumabsu=3.536, sumabsv=3.162)
comp 1 | u |cor|=1.0  v |cor|=1.0 | nz u: J=22 R=22  v: J=16 R=16 | d: J=11.1351 R=11.1351
comp 2 | u |cor|=1.0  v |cor|=1.0 | nz u: J=24 R=24  v: J=17 R=17 | d: J=10.3343 R=10.3343
comp 3 | u |cor|=1.0  v |cor|=1.0 | nz u: J=20 R=20  v: J=19 R=19 | d: J=10.5875 R=10.5875

d vector diff (‖d_j - d_r‖): 2.15e-14

--- sumabsu=√n, sumabsv=√p (max budget → dense → ordinary SVD) ---
  Julia v vs SVD V1: 1.0   u vs SVD U1: 1.0
  R     v vs SVD V1: 1.0   u vs SVD U1: 1.0
  nz at max budget — Julia u:50/50 v:40/40   R u:50/50 v:40/40

============================================================
PART 3 — BENCHMARK
============================================================

[n=50, p=40, K=3]
  Julia:   293.500 μs (44 allocations: 155.00 KiB)
  R    :   median 15.917 ms

[n=200, p=100, K=3]
  Julia:   1.807 ms (47 allocations: 984.28 KiB)
  R    :   median 28.676 ms

============================================================
=#
