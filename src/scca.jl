# Abhisek Banerjee
# sparse_cca — Sparse Canonical Correlation Analysis via the Penalized Matrix
# Decomposition. Faithful transcription of PMA::CCA (typex=typez="standard",
# L1 penalties), with helpers from PMA's PMD.R (soft, l2n, BinarySearch).
#
# Witten, Tibshirani & Hastie (2009) Biostatistics 10(3):515-534;
# Witten & Tibshirani (2009) SAGMB 8(1):28.
#
# Column-major convention (consistent with cca.jl): each COLUMN of X (dx×n)
# and Y (dy×n) is an observation. Internally transposed to PMA's rows=obs layout.


struct SparseCcaResult{T}
    u::Matrix{T}        # X canonical vectors (dx × K), sparse
    v::Matrix{T}        # Y canonical vectors (dy × K), sparse
    d::Vector{T}        # d_k = (X u_k)·(Z v_k) for each factor
    cors::Vector{T}     # cor(X u_k, Z v_k) on the standardized data
    penaltyx::Float64   # [0,1] L1-bound fraction used for u
    penaltyz::Float64   # [0,1] L1-bound fraction used for v
    K::Int
end

# PMA helpers
# soft(x,d) = sign(x) * max(|x| - d, 0)
_softcca(a, d) = sign.(a) .* max.(abs.(a) .- d, 0.0)


# l2n: L2 norm, floored at 0.05 when zero 
function _l2n(vec)
    a = sqrt(sum(abs2, vec))
    return a == 0 ? 0.05 : a
end

# BinarySearch: find lam ≥ 0 so that ‖soft(argu,lam) / ‖soft(argu,lam)‖₂‖₁ = sumabs.
# Returns 0 if the unthresholded (normalized) vector already satisfies the L1 bound.
function _binary_search(argu, sumabs)
    if _l2n(argu) == 0 || sum(abs, argu ./ _l2n(argu)) <= sumabs
        return 0.0
    end
    lam1 = 0.0
    lam2 = maximum(abs, argu) - 1e-5
    iter = 1
    while iter < 150
        mid = (lam1 + lam2) / 2
        su = _softcca(argu, mid)
        if sum(abs, su ./ _l2n(su)) < sumabs
            lam2 = mid
        else
            lam1 = mid
        end
        (lam2 - lam1) < 1e-6 && return (lam1 + lam2) / 2
        iter += 1
    end
    return (lam1 + lam2) / 2     # PMA warns "Didn't quite converge" here
end


# single-factor sparse CCA (transcription of PMA's SparseCCA, standard/L1)
# x: n×p1, z: n×p2 (rows = observations).  v0: initial right vector (length p2).
function _sparse_cca_single(x, z, v0, penaltyx, penaltyz, niter)
    p1 = size(x, 2); p2 = size(z, 2)
    c1 = penaltyx * sqrt(p1)  # since c1 must like between 1 and sqrt(p1)
    c2 = penaltyz * sqrt(p2)  # since c2 must like between 1 and sqrt(p2)
    v = copy(v0)
    vold = randn(length(v))                 # R: vold <- rnorm(length(v))
    u = zeros(p1)
    for _ in 1:niter
        if sum(abs, vold .- v) > 1e-6
            #  update u from current v   argu = (z v)' x   (length p1)
            # z * v → X₂v. transpose(z*v) * x → (X₂v)ᵀX₁ → a 1×p₁ row. (X₂v)ᵀX₁ = vᵀX₂ᵀX₁ = (X₁ᵀX₂v)ᵀ, this equals X₁ᵀX₂v
            argu = vec(transpose(z * v) * x)  
            lamu = _binary_search(argu, c1)
            su = _softcca(argu, lamu)
            u = su ./ _l2n(su)
            #  update v   (set vold = v that produced this u, then update)
            vold = copy(v)
            # x * u  → X₁u.   transpose(x*u) * z → (X₁u)ᵀX₂ → a 1×p₂ row.  (X₁u)ᵀX₂ = uᵀX₁ᵀX₂ = (X₂ᵀX₁u)ᵀ,  this equals X₂ᵀX₁u
            argv = vec(transpose(x * u) * z)   # (x u)' z   (length p2)
            lamv = _binary_search(argv, c2)
            sv = _softcca(argv, lamv)
            v = sv ./ _l2n(sv)
        end
    end
    d = dot(x * u, z * v)
    return u, v, d
end

# public interface
"""
    scca(X, Y; penaltyx=0.3, penaltyz=0.3, K=1, niter=15, standardize=true)

Sparse CCA via the penalized matrix decomposition, matching PMA::CCA
(standard / L1 penalties). Each COLUMN of `X` (dx×n) and `Y` (dy×n) is an
observation; both must share the same number of columns.

- `penaltyx`, `penaltyz` ∈ (0,1]: L1-bound fractions. The L1 bound on a
  canonical vector is `penalty · √(num features)`; smaller ⇒ sparser.
- `K`: number of canonical vector pairs.
- `niter`: iterations per factor (PMA default 15).
- `standardize`: center+scale each feature to mean 0, sd 1 (PMA default true).

Returns a `SparseCcaResult` with sparse `u` (dx×K), `v` (dy×K), `d`, and
`cors` = cor(Xuₖ, Zvₖ).
"""
function scca(X::AbstractMatrix, Y::AbstractMatrix;
                    penaltyx::Real=0.3, penaltyz::Real=0.3,
                    K::Int=1, niter::Int=15, standardize::Bool=true)
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must share the number of columns (observations)."))
    dx >= 2 && dy >= 2 || throw(ArgumentError("need at least two features in each of X and Y"))
    (0 < penaltyx <= 1 && 0 < penaltyz <= 1) || throw(ArgumentError("penaltyx, penaltyz must be in (0,1]"))
    1 <= K <= min(dx, dy) || throw(ArgumentError("K must be in 1:min(dx,dy)"))

    # to PMA's rows=observations layout: x is n×dx, z is n×dy
    x = Matrix{Float64}(transpose(X))
    z = Matrix{Float64}(transpose(Y))

    if standardize
        sdx = std(x, dims=1; corrected=true)
        sdz = std(z, dims=1; corrected=true)
        any(sdx .== 0) && throw(ArgumentError("a column of X has zero std; cannot standardize"))
        any(sdz .== 0) && throw(ArgumentError("a column of Y has zero std; cannot standardize"))
        x = (x .- mean(x, dims=1)) ./ sdx
        z = (z .- mean(z, dims=1)) ./ sdz
    end

    # init v from the leading K right singular vectors of x'z (PMA's CheckVs)
    Vfull = svd(transpose(x) * z).V          # (dy × min(dx,dy))
    Vinit = Vfull[:, 1:K]

    U = zeros(dx, K); V = zeros(dy, K); D = zeros(K); C = zeros(K)

    # PMA deflation: augment the data matrices with previous (scaled) factors
    xres = copy(x); zres = copy(z)
    for k in 1:K
        u, v, d = _sparse_cca_single(xres, zres, Vinit[:, k], penaltyx, penaltyz, niter)
        U[:, k] = u; V[:, k] = v; D[k] = d
        # canonical correlation on the ORIGINAL standardized data
        if any(!iszero, u) && any(!iszero, v)
            C[k] = cor(x * u, z * v)
        end
        # deflate by appending sqrt(d)*u' to x and -sqrt(d)*v' to z (new rows)
        if k < K
            xres = vcat(xres, sqrt(d) .* transpose(u))
            zres = vcat(zres, -sqrt(d) .* transpose(v))
        end
    end

    return SparseCcaResult(U, V, D, C, Float64(penaltyx), Float64(penaltyz), K)
end