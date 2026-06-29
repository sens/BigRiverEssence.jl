# benchmark_julia_from_r.R — R-hosted comparison of BigRiverSchneider (Julia) vs the
# native R packages, using JuliaCall to embed Julia. For each method: check the
# results are identical, then microbenchmark both implementations on ONE clock.
#
# Why this direction: hosting in R and timing both sides with the same microbenchmark
# call gives a fair head-to-head — unlike the Julia-hosted RCall version, where the
# two ran in separate runtimes. The tradeoff: a JuliaCall call from R carries the
# R->Julia->R marshalling cost, so the Julia timing here INCLUDES that boundary
# overhead (it's a "called from R" time, not Julia's standalone time).
#
# Prereqs:
#   install.packages(c("JuliaCall","microbenchmark","PMA","r.jive"))
#   # mixOmics from Bioconductor:  BiocManager::install("mixOmics")
#   # and BigRiverSchneider must be available to the Julia that JuliaCall starts.
#
# Run:  Rscript benchmark_julia_from_r.R

suppressMessages({
  library(JuliaCall)
  library(microbenchmark)
})

cat("Starting embedded Julia (first call compiles; this can take a minute)...\n")
julia_setup()                                  # boots the embedded Julia
julia_command("using BigRiverSchneider")
julia_command("const BRS = BigRiverSchneider")
julia_command("using Statistics, LinearAlgebra")

# sign-invariant column agreement: max over columns of 1 - |cos| (0 = identical)
colmisalign <- function(A, B) {
  stopifnot(ncol(A) == ncol(B))
  max(sapply(seq_len(ncol(A)), function(j) {
    a <- A[, j] / sqrt(sum(A[, j]^2)); b <- B[, j] / sqrt(sum(B[, j]^2))
    1 - abs(sum(a * b))
  }))
}
# selected-support sets equal per column?
sets_match <- function(A, B)
  all(sapply(seq_len(ncol(A)),
             function(j) setequal(which(A[, j] != 0), which(B[, j] != 0))))

verdict <- function(mis, sm = NA) {
  ok <- mis < 1e-2 && (is.na(sm) || sm)
  cat(sprintf("  result: %-11s  max misalignment = %.2e%s\n",
              if (ok) "IDENTICAL" else "DIFFER", mis,
              if (is.na(sm)) "" else if (sm) "  (sets match)" else "  (SETS DIFFER)"))
}

# time two expressions together, same clock; print medians in ms
bench2 <- function(label_jl, jl, label_r, r, times = 20L) {
  mb <- microbenchmark(julia = jl(), Rpkg = r(), times = times)
  med <- tapply(mb$time, mb$expr, median) / 1e6     # ns -> ms
  cat(sprintf("  %-22s %9.3f ms\n", label_jl, med[["julia"]]))
  cat(sprintf("  %-22s %9.3f ms   (same clock)\n", label_r, med[["Rpkg"]]))
}

line <- function() cat(strrep("=", 78), "\n")

# ===========================================================================
# 1. PMD  vs PMA::PMD
# ===========================================================================
line(); cat("  PMD  vs  PMA::PMD\n"); line()
suppressMessages(library(PMA))
set.seed(11)
n <- 60; p <- 40; sumabs <- 0.4; K <- 3L
X <- matrix(rnorm(n * p), n, p)

# native R
rj <- PMD(X, type = "standard", sumabs = sumabs, K = K, center = TRUE, trace = FALSE)
# Julia, called from R via JuliaCall (BRS.pmd centers internally)
mj <- julia_call("BRS.pmd", X, sumabs = sumabs, K = as.integer(K), center = TRUE)
v_jl <- julia_call("getfield", mj, as.symbol("v"))
u_jl <- julia_call("getfield", mj, as.symbol("u"))
verdict(max(colmisalign(v_jl, rj$v), colmisalign(u_jl, rj$u)))
bench2("BRS.pmd (via Julia)",
       function() julia_call("BRS.pmd", X, sumabs = sumabs, K = as.integer(K), center = TRUE),
       "PMA::PMD (native R)",
       function() PMD(X, type = "standard", sumabs = sumabs, K = K, center = TRUE, trace = FALSE))

# ===========================================================================
# 2. SPC  vs PMA::SPC
# ===========================================================================
line(); cat("  SPC  vs  PMA::SPC\n"); line()
set.seed(13)
n <- 80; p <- 30; sumabsv <- sqrt(p) / 2; K <- 3L
X <- matrix(rnorm(n * p), n, p)

rj <- SPC(scale(X, center = TRUE, scale = FALSE), sumabsv = sumabsv, K = K, trace = FALSE)
mj <- julia_call("BRS.spc", X, k = as.integer(K), c = sumabsv)
v_jl <- julia_call("getfield", mj, as.symbol("loadings"))
verdict(colmisalign(v_jl, rj$v), sets_match(v_jl, rj$v))
bench2("BRS.spc (via Julia)",
       function() julia_call("BRS.spc", X, k = as.integer(K), c = sumabsv),
       "PMA::SPC (native R)",
       function() SPC(scale(X, center = TRUE, scale = FALSE), sumabsv = sumabsv, K = K, trace = FALSE))

# ===========================================================================
# 3. SCCA  vs PMA::CCA
# ===========================================================================
line(); cat("  SCCA  vs  PMA::CCA\n"); line()
set.seed(17)
n <- 100; p1 <- 40; p2 <- 50; px <- 0.3; pz <- 0.3; K <- 2L; niter <- 15L
Xr <- matrix(rnorm(n * p1), n, p1)   # PMA layout: obs in rows
Zr <- matrix(rnorm(n * p2), n, p2)

rj <- CCA(Xr, Zr, typex = "standard", typez = "standard",
          penaltyx = px, penaltyz = pz, K = K, niter = niter, trace = FALSE)
# BRS.scca wants variables in rows ⇒ transpose in Julia via permutedims
mj <- julia_call("BRS.scca", t(Xr), t(Zr),
                 penaltyx = px, penaltyz = pz, K = as.integer(K), niter = as.integer(niter))
u_jl <- julia_call("getfield", mj, as.symbol("u"))
v_jl <- julia_call("getfield", mj, as.symbol("v"))
verdict(max(colmisalign(u_jl, rj$u), colmisalign(v_jl, rj$v)),
        sets_match(u_jl, rj$u) && sets_match(v_jl, rj$v))
bench2("BRS.scca (via Julia)",
       function() julia_call("BRS.scca", t(Xr), t(Zr),
                             penaltyx = px, penaltyz = pz, K = as.integer(K), niter = as.integer(niter)),
       "PMA::CCA (native R)",
       function() CCA(Xr, Zr, typex = "standard", typez = "standard",
                      penaltyx = px, penaltyz = pz, K = K, niter = niter, trace = FALSE))

# ===========================================================================
# 4a. JIVE  vs r.jive  (given ranks)
# ===========================================================================
line(); cat("  JIVE  vs  r.jive (given ranks)\n"); line()
suppressMessages(library(r.jive))
set.seed(2024)
n <- 80; rT <- 2L; r1 <- 3L; r2 <- 3L; p1 <- 60; p2 <- 50
S  <- matrix(rnorm(rT * n), rT, n)
U1 <- matrix(rnorm(p1 * rT), p1, rT); U2 <- matrix(rnorm(p2 * rT), p2, rT)
S1 <- matrix(rnorm(r1 * n), r1, n);   W1 <- matrix(rnorm(p1 * r1), p1, r1)
S2 <- matrix(rnorm(r2 * n), r2, n);   W2 <- matrix(rnorm(p2 * r2), p2, r2)
X1 <- U1 %*% S + W1 %*% S1 + 0.3 * matrix(rnorm(p1 * n), p1, n)
X2 <- U2 %*% S + W2 %*% S2 + 0.3 * matrix(rnorm(p2 * n), p2, n)

rj <- jive(list(X1, X2), rankJ = rT, rankA = c(r1, r2), method = "given",
           conv = 1e-6, maxiter = 1000, scale = TRUE, center = TRUE, showProgress = FALSE)

# assign the blocks into Julia, fit there, pull the joint back BY NAME (mj.J)
julia_assign("X1", X1); julia_assign("X2", X2)
julia_command(sprintf("mj = BRS.jive([X1, X2], %d, [%d, %d])", rT, r1, r2))
J1_jl <- julia_eval("reduce(vcat, mj.J)")        # stacked joint, extracted Julia-side

# subspace agreement via canonical correlation (decompositions aren't bit-identical)
Jr <- rbind(rj$joint[[1]], rj$joint[[2]])
qrz <- function(M) qr.Q(qr(M))[, seq_len(rT), drop = FALSE]
cc  <- svd(t(qrz(t(J1_jl))) %*% qrz(t(Jr)))$d
verdict(1 - min(cc))                             # smaller = more identical

bench2("BRS.jive (via Julia)",
       function() julia_eval(sprintf("BRS.jive([X1, X2], %d, [%d, %d])", rT, r1, r2)),
       "r.jive (native R)",
       function() jive(list(X1, X2), rankJ = rT, rankA = c(r1, r2), method = "given",
                       conv = 1e-6, maxiter = 1000, scale = TRUE, center = TRUE, showProgress = FALSE),
       times = 5L)

# ===========================================================================
# 4b. JIVE  vs r.jive  (ESTIMATED ranks — permutation test)
# ===========================================================================
line(); cat("  JIVE  vs  r.jive (estimated ranks, permutation)\n"); line()
# Strong, well-separated structure (mirrors your fixture generator): joint rank 2,
# individual rank 3 each. Build it exactly as generate_jive_reference.R does.
set.seed(99)
ns <- 80; ps1 <- 60; ps2 <- 50
Sj  <- matrix(rnorm(2 * ns), 2, ns)
i1  <- matrix(rnorm(ps1 * 3), ps1, 3) %*% matrix(rnorm(3 * ns), 3, ns)
i2  <- matrix(rnorm(ps2 * 3), ps2, 3) %*% matrix(rnorm(3 * ns), 3, ns)
X1s <- matrix(rnorm(ps1 * 2), ps1, 2) %*% Sj + 0.5 * i1 + 0.1 * matrix(rnorm(ps1 * ns), ps1, ns)
X2s <- matrix(rnorm(ps2 * 2), ps2, 2) %*% Sj + 0.5 * i2 + 0.1 * matrix(rnorm(ps2 * ns), ps2, ns)

# native R: r.jive estimates ranks itself (method="perm", no nperm argument)
set.seed(99)
rj <- jive(list(X1s, X2s), method = "perm", scale = TRUE, center = TRUE,
           orthIndiv = TRUE, showProgress = FALSE)
r_rank_J <- rj$rankJ
r_rank_A <- rj$rankA

# Julia: omit ranks ⇒ BRS.jive's permutation estimate; fix its RNG Julia-side
julia_assign("X1p", X1s); julia_assign("X2p", X2s)
julia_command("import Random; Random.seed!(999)")
julia_command("mjp = BRS.jive([X1p, X2p]; nperm = 100)")
jl_rank_J <- as.integer(julia_eval("mjp.r"))
jl_rank_A <- as.integer(julia_eval("collect(mjp.ri)"))

cat(sprintf("  Julia : joint = %d   indiv = [%s]\n",
            jl_rank_J, paste(jl_rank_A, collapse = ", ")))
cat(sprintf("  r.jive: joint = %d   indiv = [%s]\n",
            r_rank_J, paste(r_rank_A, collapse = ", ")))
ranks_match <- jl_rank_J == r_rank_J && all(jl_rank_A == r_rank_A)
cat(sprintf("  result: %-11s  (ranks %s)\n",
            if (ranks_match) "IDENTICAL" else "DIFFER",
            if (ranks_match) "match" else "DIFFER"))

# if the joint ranks agree, also compare the joint SUBSPACE
if (jl_rank_J == r_rank_J && jl_rank_J > 0) {
  J1_jl <- julia_eval("reduce(vcat, mjp.J)")
  Jr <- rbind(rj$joint[[1]], rj$joint[[2]])
  d  <- jl_rank_J
  qrz <- function(M) qr.Q(qr(M))[, seq_len(d), drop = FALSE]
  cc  <- svd(t(qrz(t(J1_jl))) %*% qrz(t(Jr)))$d
  cat(sprintf("  joint subspace: min canonical corr = %.5f%s\n",
              min(cc), if (min(cc) > 0.99) "  ✓" else "  ✗"))
}

# benchmark both auto-rank fits (permutation runs inside each ⇒ slow ⇒ few reps)
bench2("BRS.jive auto (via Julia)",
       function() julia_eval("BRS.jive([X1p, X2p]; nperm = 100)"),
       "r.jive perm (native R)",
       function() jive(list(X1s, X2s), method = "perm", scale = TRUE, center = TRUE,
                       orthIndiv = TRUE, showProgress = FALSE),
       times = 3L)

# ===========================================================================
# 5. SPLSDA  vs mixOmics::splsda
# ===========================================================================
line(); cat("  SPLSDA  vs  mixOmics::splsda\n"); line()
suppressMessages(library(mixOmics))
set.seed(123)
classes <- c("A", "B", "C"); n_per <- 20
y <- factor(rep(classes, each = n_per)); n <- length(y); p <- 200
X <- matrix(rnorm(n * p) * 0.5, n, p)
for (ci in seq_along(classes))
  X[y == classes[ci], 1:10] <- X[y == classes[ci], 1:10] + ci * 2.0
ncomp <- 2L; keepX <- c(10L, 10L)

rj <- splsda(X, y, ncomp = ncomp, keepX = keepX)
rlx <- rj$loadings$X
# BRS.splsda takes the label vector directly (as a Julia Vector of strings)
yc <- as.character(y)
julia_assign("Xspl", X); julia_assign("yspl", yc)
mj <- julia_eval(sprintf("BRS.splsda(Xspl, yspl, %d, [%d, %d])", ncomp, keepX[1], keepX[2]))
lx_jl <- julia_call("getfield", mj, as.symbol("loadings_X"))
verdict(colmisalign(lx_jl, rlx), sets_match(lx_jl, rlx))
bench2("BRS.splsda (via Julia)",
       function() julia_eval(sprintf("BRS.splsda(Xspl, yspl, %d, [%d, %d])", ncomp, keepX[1], keepX[2])),
       "mixOmics::splsda (native R)",
       function() splsda(X, y, ncomp = ncomp, keepX = keepX),
       times = 10L)

line()
cat("  Done. Timings are same-clock microbenchmark medians.\n")
cat("  NOTE: the Julia times INCLUDE the JuliaCall R->Julia->R marshalling cost,\n")
cat("  so they're 'called-from-R' times, not Julia's standalone speed.\n")
line()



