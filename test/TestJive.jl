# test/TestJive.jl
# Validity checks for jive — self-contained, validates against the
# generative model (supplement §6.2) and the decomposition's defining properties.

using BigRiverSchneider
using LinearAlgebra, Statistics, Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Build data with KNOWN joint + individual structure (supplement §6.2 generative model):
#   X₁ = U₁S + W₁S₁ ,  X₂ = U₂S + W₂S₂   — shared scores S, individual scores Sᵢ.
# ---------------------------------------------------------------------------
n = 100
r, r1, r2 = 2, 3, 3            # true ranks: joint=2, individual=3 each
p1, p2 = 40, 30                # variables per dataset

S  = randn(r, n)               # shared JOINT scores (same across both datasets)
U1 = randn(p1, r); U2 = randn(p2, r)     # joint loadings
S1 = randn(r1, n); W1 = randn(p1, r1)    # individual to dataset 1
S2 = randn(r2, n); W2 = randn(p2, r2)    # individual to dataset 2

X1 = U1*S + W1*S1              # dataset 1 = joint + individual (no noise)
X2 = U2*S + W2*S2              # dataset 2

println("X1 is $p1×$n,  X2 is $p2×$n,  true ranks: joint=$r, indiv=[$r1,$r2]\n")

res = jive([X1, X2], r, [r1, r2]; standardize = false)

# ---------------------------------------------------------------------------
# TEST 1 — Exact reconstruction (supplement §6.2 anchor).
# No noise + true ranks ⇒ JIVE must recover the data to machine precision.
# ---------------------------------------------------------------------------
Xc1 = X1 .- mean(X1, dims = 2)         # compare against row-centered data
Xc2 = X2 .- mean(X2, dims = 2)
recon_err = norm(Xc1 .- (res.J[1] .+ res.A[1]))^2 +
            norm(Xc2 .- (res.J[2] .+ res.A[2]))^2
println("TEST 1  exact reconstruction (noiseless, true ranks)")
println("  ‖X − (J+A)‖²        : ", round(recon_err, digits = 12), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 2 — Orthogonality constraint  J Aᵢᵀ = 0  (the defining JIVE property).
# Joint and individual row spaces must be orthogonal (enforced by the I−VVᵀ projection).
# ---------------------------------------------------------------------------
Jfull = vcat(res.J[1], res.J[2])
ortho1 = norm(Jfull * res.A[1]')       # J Aᵢᵀ should be ≈ 0
ortho2 = norm(Jfull * res.A[2]')
println("TEST 2  orthogonality of joint & individual  (J Aᵢᵀ = 0)")
println("  ‖J·A₁ᵀ‖             : ", round(ortho1, digits = 10))
println("  ‖J·A₂ᵀ‖             : ", round(ortho2, digits = 10), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 3 — Ranks of the estimated structures match the requested ranks.
# ---------------------------------------------------------------------------
rank_tol = 1e-8
joint_rank = rank(Jfull, rank_tol)
indiv_rank1 = rank(res.A[1], rank_tol)
indiv_rank2 = rank(res.A[2], rank_tol)
println("TEST 3  estimated ranks match requested")
println("  joint rank          : ", joint_rank,  "  (want $r)")
println("  individual rank 1   : ", indiv_rank1, "  (want $r1)")
println("  individual rank 2   : ", indiv_rank2, "  (want $r2)\n")

# ---------------------------------------------------------------------------
# TEST 4 — Factorization consistency: J should equal U·S, Aᵢ should equal Wᵢ·Sᵢ.
# Checks jive_factorize / the returned scores & loadings reconstruct the matrices.
# ---------------------------------------------------------------------------
fac_err_joint = norm(res.J[1] .- res.U[1]*res.S) + norm(res.J[2] .- res.U[2]*res.S)
fac_err_ind   = norm(res.A[1] .- res.Wi[1]*res.Si[1]) + norm(res.A[2] .- res.Wi[2]*res.Si[2])
println("TEST 4  factorization consistency")
println("  ‖J − U·S‖           : ", round(fac_err_joint, digits = 10), "   (want ≈ 0)")
println("  ‖A − W·S_indiv‖     : ", round(fac_err_ind,   digits = 10), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 5 — Recovery of the JOINT SUBSPACE (the scientifically meaningful part).
# JIVE's joint scores S should span the same space as the true shared S.
# Compare via principal angles: the subspaces should align (canonical correlations ≈ 1).
# ---------------------------------------------------------------------------
Strue = S .- mean(S, dims = 2)          # center true joint scores for fair comparison
Sest  = res.S
# subspace alignment: singular values of (orthonormal basis of Strueᵀ)ᵀ (orthonormal basis of Sestᵀ)
Qtrue = Matrix(qr(Strue').Q)[:, 1:r]
Qest  = Matrix(qr(Sest').Q)[:, 1:r]
canon = svd(Qtrue' * Qest).S            # canonical correlations between the two joint subspaces
println("TEST 5  joint subspace recovery")
println("  canonical correlations: ", round.(canon, digits = 6), "   (want all ≈ 1.0)")