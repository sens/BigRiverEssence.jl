# smoke_test.jl — quick "does it run" check for pca, pmd, plskern, jive
using BigRiverSchneider
using LinearAlgebra, Statistics, Random
Random.seed!(1)

println("="^50)
println("PCA")
println("="^50)
X = randn(200, 30)
m = pca(X; k=5, method=:svd)
println("svd  : loadings ", size(m.loadings), "  vars ", round.(m.variances[1:3], digits=3))
m2 = pca(X; k=5, method=:cov)
println("cov  : loadings ", size(m2.loadings))
ms = pca(X; k=5, standardize=true)
println("std  : scale[1:3] ", round.(ms.scale[1:3], digits=3), " (should not be 1.0)")
sc = pca_transform(m, X)
Xr = pca_invtransform(pca(X; k=30), pca_transform(pca(X; k=30), X))
println("transform: scores ", size(sc), "   round-trip ‖Xr-X‖ ", round(norm(Xr .- X), digits=8))

println("\n", "="^50)
println("PMD (sparse PCA)")
println("="^50)
Xp = randn(150, 40)
mp = pmd(Xp; k=3, c=3.0)
println("pmd      : loadings ", size(mp.loadings), "  nonzeros/col ", [count(!iszero, mp.loadings[:,j]) for j in 1:3])
mpo = pmd_orth(Xp; k=3, c=3.0)
println("pmd_orth : loadings ", size(mpo.loadings), "  nonzeros/col ", [count(!iszero, mpo.loadings[:,j]) for j in 1:3])
# sanity: at c = sqrt(p), no sparsity → should match ordinary PC1 direction
mfull = pmd(Xp; k=1, c=sqrt(40))
v_ord = svd(Xp .- mean(Xp, dims=1)).V[:, 1]
println("c=√p check: |⟨v_pmd, v_pca⟩| = ", round(abs(dot(mfull.loadings[:,1], v_ord)), digits=4), " (want ≈ 1.0)")

println("\n", "="^50)
println("PLSKERN")
println("="^50)
Xl = randn(100, 20); Yl = randn(100, 3)
ml = plskern(Xl, Yl; nlv=4, method=:algo1)
println("algo1: W ", size(ml.W), "  T ", size(ml.T), "  Q ", size(ml.Q))
ml2 = plskern(Xl, Yl; nlv=4, method=:algo2)
println("algo2: W ", size(ml2.W))
yvec = randn(100)
mlv = plskern(Xl, yvec; nlv=4)              # Y as a vector
println("y-vector: Q ", size(mlv.Q), " (q should be 1)")
B, intc = plskerncoef(ml)
pred = plskernpredict(ml, Xl[1:5, :])
trans = plskerntransform(ml, Xl[1:5, :])
println("coef B ", size(B), "  predict ", size(pred), "  transform ", size(trans))

println("\n", "="^50)
println("JIVE")
println("="^50)
n = 50
X1 = randn(30, n); X2 = randn(40, n)
# fixed-rank path (skips permutation estimation)
jf = jive([X1, X2], 2, [1, 1]; maxiter=200)
println("fixed-rank: r=", jf.r, "  ri=", jf.ri)
println("  J blocks ", [size(J) for J in jf.J], "  S ", size(jf.S))
println("  A blocks ", [size(A) for A in jf.A])
# rank-estimation path (slower — small nperm for speed)
je = jive([X1, X2]; nperm=20, maxiter=200)
println("perm-estimated: r=", je.r, "  ri=", je.ri)


println("="^50)
println("SPLS-DA")
println("="^50)

# planted: 3 classes separated along a few features, rest noise
n_per = 40
k = 3
p = 100
nz = 15                                   # truly discriminative features
X = randn(n_per*k, p)
y = repeat(1:k, inner=n_per)
for (ci, c) in enumerate(1:k)             # shift the first nz features per class
    rows = (ci-1)*n_per+1 : ci*n_per
    X[rows, 1:nz] .+= (ci - 2) * 3.0      # class means spread on features 1:nz
end

ncomp = 2
keepX = [20, 20]                          # keep 20 features per component
m = splsda(X, y, ncomp, keepX)

println("struct fields:")
println("  variates_X ", size(m.variates_X), "  variates_Y ", size(m.variates_Y))
println("  loadings_X ", size(m.loadings_X), "  loadings_Y ", size(m.loadings_Y))
println("  ncomp ", m.ncomp, "  keepX ", m.keepX)
println("  Y_dummy ", size(m.Y_dummy), "  classes ", m.classes)

# sparsity contract: each X-loading column has ≤ keepX nonzeros
nzcols = [count(!iszero, m.loadings_X[:, c]) for c in 1:ncomp]
println("nonzeros per X-loading column: ", nzcols, "  (should be ≤ ", keepX, ")")

# did it pick the discriminative features? (first nz) — check overlap on comp 1
sel = Set(findall(!iszero, m.loadings_X[:, 1]))
hits = length(intersect(sel, Set(1:nz)))
println("comp-1 selected ", length(sel), " features, ", hits, "/", nz, " are truly discriminative")

# separation: class centroids in the X-variate space should be distinct
println("variate means by class (comp 1):")
for c in 1:k
    rows = (c-1)*n_per+1 : c*n_per
    println("  class $c : ", round(mean(m.variates_X[rows, 1]), digits=3))
end

println("\n--- binary case (k=2) ---")
y2 = repeat(1:2, inner=n_per)
X2 = randn(2*n_per, p); X2[1:n_per, 1:nz] .+= 3.0
m2 = splsda(X2, y2, 2, [15, 15])
println("  variates_Y ", size(m2.variates_Y), "  classes ", m2.classes, " (k=2)")

println("\n--- levels argument (custom class order) ---")
ys = repeat(["b", "a", "c"], inner=n_per)        # string labels
ml = splsda(X, ys, 1, [20]; levels=["a","b","c"])
println("  classes ", ml.classes, " (forced order a,b,c)")

println("\n--- no scaling ---")
mns = splsda(X, y, 1, [20]; scale=false)
println("  ran with scale=false, loadings_X ", size(mns.loadings_X))

println("\n--- argument check ---")
try
    splsda(X, y, 2, [20])                         # keepX length 1 ≠ ncomp 2
    println("  ERROR: should have thrown")
catch e
    println("  correctly threw: ", typeof(e))
end





println("="^50)
println("CCA")
println("="^50)

# planted: X and Y share latent factors → known canonical correlations.
# column-major: each COLUMN is an observation (dx×n, dy×n).
n  = 200
dx, dy = 8, 6
z1 = randn(n); z2 = randn(n)                       # two shared latent signals
Ax = randn(dx, 2); Ay = randn(dy, 2)
X = Ax * vcat(z1', z2') .+ 0.3 .* randn(dx, n)     # dx×n
Y = Ay * vcat(z1', z2') .+ 0.3 .* randn(dy, n)     # dy×n

m = cca(X, Y; method=:svd, outdim=4)
println("struct fields:")
println("  xproj ", size(m.xproj), "  yproj ", size(m.yproj))
println("  corrs ", round.(m.corrs, digits=4))
println("  nobs ", m.nobs)

# canonical correlations in [0,1], descending
println("corrs in [0,1] & descending: ",
        all(0 .<= m.corrs .<= 1 + 1e-9) && issorted(m.corrs; rev=true))
# two shared factors → first two canonical corrs should be high
println("top-2 canonical corrs: ", round.(m.corrs[1:2], digits=4), " (want ≈ high)")

println("\n--- :cov method ---")
mc = cca(X, Y; method=:cov, outdim=4)
println("  corrs ", round.(mc.corrs, digits=4), "  nobs ", mc.nobs, " (-1 expected for :cov)")
# :svd and :cov should agree on the canonical correlations
println("  |corrs_svd - corrs_cov| max: ", round(maximum(abs.(m.corrs .- mc.corrs)), sigdigits=3))

println("\n--- transform ---")
Xt = cca_transform(m, X, :x)
Yt = cca_transform(m, Y, :y)
println("  transformed X ", size(Xt), "  Y ", size(Yt), " (outdim × n)")
# canonical variates should correlate at exactly corrs[i] (the definition)
r1 = cor(Xt[1, :], Yt[1, :])
println("  cor(Xvariate1, Yvariate1) = ", round(abs(r1), digits=4),
        "  vs corrs[1] = ", round(m.corrs[1], digits=4), " (should match)")

println("\n--- argument checks ---")
for (desc, f) in [
    ("mismatched n",   () -> cca(X, randn(dy, n+5))),
    ("outdim too big", () -> cca(X, Y; outdim=99)),
    ("bad method",     () -> cca(X, Y; method=:bogus)),
    ("bad transform",  () -> cca_transform(m, X, :z)),
]
    try
        f(); println("  $desc: ERROR — should have thrown")
    catch e
        println("  $desc: correctly threw ", typeof(e))
    end
end





println("="^50)
println("SPARSE CCA (scca)")
println("="^50)

# planted: X and Y share a latent factor, but only the first nz features
# of each view load on it → sparse CCA should select those features.
# column-major: each COLUMN is an observation (dx×n, dy×n).
n  = 100
dx, dy = 500, 1000
nz1, nz2 = 25, 40                          # truly-loading features per view
lat = randn(n)                             # shared latent factor
Xr = randn(n, dx); Yr = randn(n, dy)       # rows=obs for construction
Xr[:, 1:nz1] .+= lat * (2.0 .* ones(nz1))'
Yr[:, 1:nz2] .+= lat * (2.0 .* ones(nz2))'
X = Matrix(transpose(Xr)); Y = Matrix(transpose(Yr))   # → dx×n, dy×n
truex = Set(1:nz1); truey = Set(1:nz2)

m = scca(X, Y; penaltyx=0.2, penaltyz=0.2, K=1, niter=15)

println("struct fields:")
println("  u ", size(m.u), "  v ", size(m.v))
println("  d ", round.(m.d, digits=4), "  cors ", round.(m.cors, digits=4))
println("  penaltyx ", m.penaltyx, "  penaltyz ", m.penaltyz, "  K ", m.K)

# sparsity: most loadings should be zero
selx = Set(findall(!iszero, m.u[:, 1]))
sely = Set(findall(!iszero, m.v[:, 1]))
println("\nselected X: ", length(selx), " / ", dx, " features")
println("selected Y: ", length(sely), " / ", dy, " features")

# ground-truth: selected features should be (almost) the planted ones
println("X precision: ", round(length(intersect(selx, truex))/max(1,length(selx)), digits=3),
        "   recall: ", round(length(intersect(selx, truex))/nz1, digits=3))
println("Y precision: ", round(length(intersect(sely, truey))/max(1,length(sely)), digits=3),
        "   recall: ", round(length(intersect(sely, truey))/nz2, digits=3))
println("canonical correlation: ", round(m.cors[1], digits=4), " (want high)")

println("\n--- K=3 (multiple factors / deflation) ---")
m3 = scca(X, Y; penaltyx=0.3, penaltyz=0.3, K=3, niter=15)
println("  u ", size(m3.u), "  v ", size(m3.v), "  cors ", round.(m3.cors, digits=4))
println("  nonzeros per u-col: ", [count(!iszero, m3.u[:,k]) for k in 1:3])
println("  nonzeros per v-col: ", [count(!iszero, m3.v[:,k]) for k in 1:3])

println("\n--- fastsvd init path (dx>n & dy>n) vs direct ---")
println("  dx>n: ", dx > n, "  dy>n: ", dy > n, " → used _fast_init_v")
# small case (dx,dy < n) should take the direct-svd branch and still run
Random.seed!(7)
ns = 200; dxs = 6; dys = 5
shared = randn(2, ns)
Xs = randn(dxs,2)*shared .+ 0.3*randn(dxs,ns)
Ys = randn(dys,2)*shared .+ 0.3*randn(dys,ns)
msmall = scca(Xs, Ys; penaltyx=0.8, penaltyz=0.8, K=2, niter=15)
println("  small case (dx,dy<n) ran: cors ", round.(msmall.cors, digits=4))

println("\n--- no standardize ---")
mns = scca(X, Y; penaltyx=0.2, penaltyz=0.2, K=1, standardize=false)
println("  ran with standardize=false: cor ", round(mns.cors[1], digits=4))

println("\n--- argument checks ---")
for (desc, f) in [
    ("mismatched n",   () -> scca(X, randn(dy, n+5))),
    ("penalty > 1",    () -> scca(X, Y; penaltyx=1.5)),
    ("penalty ≤ 0",    () -> scca(X, Y; penaltyz=0.0)),
    ("K out of range", () -> scca(X, Y; K=0)),
]
    try
        f(); println("  $desc: ERROR — should have thrown")
    catch e
        println("  $desc: correctly threw ", typeof(e))
    end
end





println("\n", "="^50)
println("ALL SMOKE TESTS RAN")
println("="^50)