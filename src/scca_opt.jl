# Abhisek Banerjee
# scca — OPTIMIZED. Same algorithm/results as scca (PMA::CCA, standard/L1).
# Key optimization: fastsvd init (PMA's trick) — computes the leading K right
# singular vectors of x'z through the small n-dimension WITHOUT forming the
# (dx×dy) cross-product, when features ≫ observations. This is where almost all
# the runtime was (the full svd(x'z) computed a 1000×1000 V to keep 3 columns).
# Plus: in-place soft-threshold + allocation-free binary search, mul! for the
# factored cross-products, preallocated buffers reused across factors/iterations.
# Uses SparseCcaResult from scca.jl. Defines NEW names only (_opt suffix / !).



# in-place soft-threshold: out .= sign(a)*max(|a|-d, 0). No allocation.
function _softcca!(out, a, d)
    @inbounds @simd for i in eachindex(a)
        ai = a[i]
        out[i] = sign(ai) * max(abs(ai) - d, 0.0)
    end
    return out
end

# l2 norm, floored at 0.05 (PMA's l2n) — allocation-free
function _l2n_val(a)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += a[i] * a[i]
    end
    r = sqrt(s)
    return r == 0 ? 0.05 : r
end

# ‖soft(a,d) / ‖soft(a,d)‖₂‖₁ — one pass, no temp arrays
function _l1_of_norm_soft(a, d)
    s2 = 0.0; s1 = 0.0
    @inbounds @simd for i in eachindex(a)
        si = sign(a[i]) * max(abs(a[i]) - d, 0.0)
        s2 += si * si
        s1 += abs(si)
    end
    nrm = sqrt(s2)
    nrm == 0 && (nrm = 0.05)
    return s1 / nrm
end

# ‖a/‖a‖₂‖₁ for the early-exit check — no temp arrays
function _l1_of_norm(a)
    nrm = _l2n_val(a)
    s1 = 0.0
    @inbounds @simd for i in eachindex(a)
        s1 += abs(a[i])
    end
    return s1 / nrm
end

# allocation-free sum(abs, a - b)  (PMA's L1 convergence metric)
function _l1diff(a, b)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += abs(a[i] - b[i])
    end
    return s
end

# BinarySearch (PMA) — allocation-free via the one-pass helpers above.
function _binary_search_opt(argu, sumabs)
    (_l2n_val(argu) == 0 || _l1_of_norm(argu) <= sumabs) && return 0.0
    lam1 = 0.0
    lam2 = maximum(abs, argu) - 1e-5
    iter = 1
    while iter < 150
        mid = (lam1 + lam2) / 2
        if _l1_of_norm_soft(argu, mid) < sumabs
            lam2 = mid
        else
            lam1 = mid
        end
        (lam2 - lam1) < 1e-6 && return (lam1 + lam2) / 2
        iter += 1
    end
    return (lam1 + lam2) / 2
end

# symmetric PSD matrix square root via eigendecomposition (PMA's msqrt):
# A^(1/2) = V diag(sqrt(max(λ,0))) V'.  A is n×n (small).
function _matsqrt(A)
    E = eigen(Symmetric(A))
    vals = sqrt.(max.(E.values, 0.0))            # clamp tiny negatives (PMA: pmax(0,·))
    return E.vectors * Diagonal(vals) * transpose(E.vectors)
end

# fastsvd init (PMA): leading K RIGHT singular vectors of x'z, computed through
# the small n-dimension WITHOUT forming the dx×dy product. x: n×dx, z: n×dy.
function _fast_init_v(x, z, K)
    xx = x * transpose(x)                        # n×n  (small!)
    xx_sqrt = _matsqrt(xx)                       # n×n symmetric square root
    y = transpose(z) * xx_sqrt                   # dy×n
    F = svd(y)                                   # only n singular triplets (not dy)
    return F.U[:, 1:K]                           # right singular vectors of x'z
end

# single-factor sparse CCA, optimized. x: nr×p1, z: nr×p2 (rows = observations;
# nr may exceed the original n after augmentation deflation). Work buffers zv_buf,
# xu_buf are sized for the MAX possible rows; we @view the active nr-length prefix.
function _sparse_cca_single_opt!(u, v, x, z, v0, penaltyx, penaltyz, niter,
                                 vold, zv_buf, xu_buf, argu, argv, su, sv)
    nr = size(x, 1)                              # current row count (= n + k - 1)
    p1 = size(x, 2); p2 = size(z, 2)
    c1 = penaltyx * sqrt(p1)
    c2 = penaltyz * sqrt(p2)
    zv = @view zv_buf[1:nr]                      # active prefixes of the buffers
    xu = @view xu_buf[1:nr]
    copyto!(v, v0)
    @inbounds for i in eachindex(vold); vold[i] = randn(); end   # R: vold <- rnorm
    fill!(u, 0.0)
    for _ in 1:niter
        if _l1diff(vold, v) > 1e-6
            # update u:  argu = (z v)' x   (= X₁ᵀX₂v, factored through the data)
            mul!(zv, z, v)                       # zv = z*v
            mul!(argu, transpose(x), zv)         # argu = x'·zv
            lamu = _binary_search_opt(argu, c1)
            _softcca!(su, argu, lamu)
            su ./= _l2n_val(su)
            copyto!(u, su)
            # set vold = v (the v that produced this u), then update v
            copyto!(vold, v)
            mul!(xu, x, u)                        # xu = x*u
            mul!(argv, transpose(z), xu)         # argv = z'·xu
            lamv = _binary_search_opt(argv, c2)
            _softcca!(sv, argv, lamv)
            sv ./= _l2n_val(sv)
            copyto!(v, sv)
        end
    end
    mul!(zv, z, v); mul!(xu, x, u)
    return dot(xu, zv)                            # d = (x u)·(z v)
end

"""
    scca_opt(X, Y; penaltyx=0.3, penaltyz=0.3, K=1, niter=15, standardize=true)

Optimized sparse CCA, matching PMA::CCA (standard / L1). Same results as `scca`.
Each COLUMN of `X` (dx×n) and `Y` (dy×n) is an observation; both must share the
same number of columns.

- `penaltyx`, `penaltyz` ∈ (0,1]: L1-bound fractions (bound = penalty·√features).
- `K`: number of canonical vector pairs.
- `niter`: iterations per factor (PMA default 15).
- `standardize`: center+scale each feature to mean 0, sd 1.
"""
function scca_opt(X::Matrix{Float64}, Y::Matrix{Float64};
                  penaltyx::Real=0.3, penaltyz::Real=0.3,
                  K::Int=1, niter::Int=15, standardize::Bool=true)
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must share the number of columns (observations)."))
    dx >= 2 && dy >= 2 || throw(ArgumentError("need at least two features in each of X and Y"))
    (0 < penaltyx <= 1 && 0 < penaltyz <= 1) || throw(ArgumentError("penaltyx, penaltyz must be in (0,1]"))
    1 <= K <= min(dx, dy) || throw(ArgumentError("K must be in 1:min(dx,dy)"))

    # to PMA's rows = observations layout: x is n×dx, z is n×dy
    x = Matrix{Float64}(transpose(X))
    z = Matrix{Float64}(transpose(Y))

    if standardize
        sdx = std(x, dims=1; corrected=true)
        sdz = std(z, dims=1; corrected=true)
        any(sdx .== 0) && throw(ArgumentError("a column of X has zero std; cannot standardize"))
        any(sdz .== 0) && throw(ArgumentError("a column of Y has zero std; cannot standardize"))
        x .= (x .- mean(x, dims=1)) ./ sdx
        z .= (z .- mean(z, dims=1)) ./ sdz
    end

    # init v: leading K right singular vectors of x'z.
    # When features ≫ observations (dx>n AND dy>n), use PMA's fastsvd — computes
    # them through the small n-dimension without forming the dx×dy product.
    # Otherwise (small/tall data) the direct SVD is fine.
    Vinit = if dx > n && dy > n
        _fast_init_v(x, z, K)
    else
        svd(transpose(x) * z).V[:, 1:K]
    end

    U = zeros(dx, K); V = zeros(dy, K); D = zeros(K); C = zeros(K)

    # preallocated buffers reused across factors and inner iterations.
    # zv/xu must hold up to n+K-1 rows (xres/zres grow by 1 row per deflation).
    nmax = n + K - 1
    u    = Vector{Float64}(undef, dx)
    v    = Vector{Float64}(undef, dy)
    vold = Vector{Float64}(undef, dy)
    zv   = Vector{Float64}(undef, nmax)
    xu   = Vector{Float64}(undef, nmax)
    argu = Vector{Float64}(undef, dx)
    argv = Vector{Float64}(undef, dy)
    su   = Vector{Float64}(undef, dx)
    sv   = Vector{Float64}(undef, dy)

    xres = copy(x); zres = copy(z)               # augmented data (PMA deflation)
    for k in 1:K
        d = _sparse_cca_single_opt!(u, v, xres, zres, @view(Vinit[:, k]),
                                    penaltyx, penaltyz, niter,
                                    vold, zv, xu, argu, argv, su, sv)
        @views U[:, k] .= u
        @views V[:, k] .= v
        D[k] = d
        # canonical correlation on the ORIGINAL standardized data (n rows)
        if any(!iszero, u) && any(!iszero, v)
            C[k] = cor(x * u, z * v)
        end
        # deflate: append sqrt(d)*u' to x and -sqrt(d)*v' to z (PMA row augmentation)
        if k < K
            xres = vcat(xres, sqrt(d) .* transpose(u))
            zres = vcat(zres, -sqrt(d) .* transpose(v))
        end
    end

    return SparseCcaResult(U, V, D, C, Float64(penaltyx), Float64(penaltyz), K)
end