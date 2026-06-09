# Abhisek Banerjee
# 2024-06-17


# PMD — the "SPC" method of Witten, Tibshirani & Hastie (2009),


# Soft-thresholding operator  S(a, Δ) = sign(a) * (|a| - Δ)_+ 
# (defined p.519, just above Lemma 2.2).
soft(a, delta) = sign(a) * max(abs(a) - delta, zero(a))

# Given z, return v = normalize(S(z, Δ)) using the smallest Δ ≥ 0 that gets ‖v‖₁ down to (<=)c.
# This is the "Δ chosen so that ‖v‖₁ = c" clause of Algorithm 3, Step 2(b) (p.520);
# the paper says Δ is found by binary search (p.520), which works because ‖S(z,Δ)‖₁


function finding_v(z, c)
    v0 = z ./ norm(z)  # start with the normalized z; if it's already sparse enough, we're done; otherwise we need to find the right Δ to get it down to c.
    sum(abs, v0) <= c && return v0            # if v0 is already sparse enough, return it; otherwise we need to find the right Δ to get it down to c.
    lo, hi = zero(eltype(z)), maximum(abs, z)  # Δ must be between 0 and max|zⱼ|; start the binary search with these bounds
    v = v0
    for _ in 1:100                           # 100 bisections pins delta to machine precision
        delta  = (lo + hi) / 2   # bisect the interval [lo, hi] to get a candidate Δ
        s  = soft.(z, delta)       # apply the soft-thresholding operator to z with this Δ to get a candidate v
        ns = norm(s)              # compute the L2 norm of this candidate v
        v  = ns == 0 ? s : s ./ ns          # normalize this candidate v to get the final candidate v; if ns is zero, just use s (which is all zeros)
        sum(abs, v) < c ? (hi = delta) : (lo = delta)     # if this candidate v is too sparse (L1 norm < c), we need to decrease Δ (move hi down); if it's not sparse enough (L1 norm > c), we need to increase Δ (move lo up)
    end
    return v
end

# One sparse component. Solves   max uᵀXv  s.t. ‖u‖₂≤1, ‖v‖₂≤1, ‖v‖₁≤c   (eq. 3.3, p.523)
# by alternating the two updates until v settles. This is Algorithm 3 (p.519-520) with
# Step 2(a) simplified to the plain power-method step — no penalty on u (p.524, top).
function spca_component(X, c; tol = 1e-6, maxiter = 500)
    # init v with a few cheap power iterations toward the leading direction —
    # avoids a full svd(X), which is wasteful just for a starting vector.
    v = randn(eltype(X), size(X, 2))  # random start; we just need a nonzero vector to get the power method going; the choice of random normal is arbitrary, any nonzero vector would do.
    v ./= norm(v) # normalize the initial vector to have L2 norm 1, which is a common practice in power iteration to ensure numerical stability; the choice of normalization is arbitrary, but it helps to keep the iterates well-scaled.
    for _ in 1:5 # 5 power iterations is usually enough to get a decent starting vector; this is a heuristic choice, not a strict rule.
        v = X' * (X * v)        # power iteration step: v ← XᵀXv; this is the standard power method for finding the leading eigenvector of XᵀX, which is the same as the leading right singular vector of X; we do this a few times to get a good starting point for the sparse PCA iterations.
        v ./= norm(v)    # normalize after each power iteration to prevent numerical issues
    end
    u = X * v; u ./= norm(u) 
    for _ in 1:maxiter
        v_old = v
        u = X * v; u ./= norm(u)              # Algo 3 simplified Step 2(a):  u ← Xv / ‖Xv‖      (p.524)
        v = finding_v(X' * u, c)                 #Algo 3 simplified  Step 2(b):  v ← S(Xᵀu, Δ)/‖·‖  (p.520)
        norm(v - v_old) < tol && break        # "iterate until convergence" (Alg 3)
    end
    d = u' * X * v                            # Algo 3Step 3:    d ← uᵀXv            (p.520)
    return d, u, v
end

# Sparse PCA: k components by deflation (Algorithm 2, p.518).
#   k = number of components
#   c = sparsity budget in [1, √p];  smaller c → sparser,  c = √p → no sparsity at all
function pmd(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,  # default of c is from Sec 2.3, P 519
              tol = 1e-6, maxiter = 500)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c must be in [1, √p] = [1, $(sqrt(p))], you entered $c"))

    # center (and optionally scale) just like our pca function
    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'
    T  = eltype(Xc)

    R = copy(Xc)                              # residual matrix; X₁ = X   (Alg 2)
    V = zeros(T, p, k)                        # sparse loadings, one per column
    d = zeros(T, k)
    for j in 1:k
        dj, uj, vj = spca_component(R, c; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        R .-= dj .* (uj * vj')                # deflate: peel this component off (Alg 2, 2b)
    end

    SignConsistency!(V)                       # same sign convention as your pca

    # NOTE on variances: sparse components are NOT orthogonal, so a clean
    # "proportion of variance" needs the projection formula in the paper (p.525).
    # Here dⱼ²/(n-1) is a rough strength measure — treat propOFvar as approximate.
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pcaStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end






# One sparse component, but with u forced orthogonal to columns of U_prev.
# This is the Section 3.2 variant (eq. 3.13-3.17, p.526-527): instead of
# deflating, we project each new u away from the earlier u's via (I - UUᵀ).
function spca_component_orth(X, c, U_prev; tol = 1e-6, maxiter = 500)
    v = randn(eltype(X), size(X, 2))
    v ./= norm(v)  # same power iteration initialization as before, but now we have to be careful to create u in the function's scope so we can project it away from U_pre
    for _ in 1:5
        v = X' * (X * v)
        v ./= norm(v)
    end
    u = X * v; u ./= norm(u)        # this u is in the function's scope, so we can project it away from U_prev in the main loop; the first time through, U_prev is empty, so this does nothing; on subsequent iterations, this ensures that u is orthogonal to all previous u's
    for _ in 1:maxiter
        v_old = v
        u = X * v                   # now this reassigns the existing u, not a new local
        if !isempty(U_prev)         # if there are previous u's, project this u away from them to enforce orthogonality; this is the (I - UUᵀ)u step in eq. 3.13-3.17, p.526-527
            u .-= U_prev * (U_prev' * u)
        end
        u ./= norm(u)        # normalize this u to have L2 norm 1
        v = finding_v(X' * u, c)         # same v update as before, but now with this new u that is orthogonal to previous u's; this is the same Step 2(b) as before, but now with the new u that has been projected to be orthogonal to previous u's
        norm(v - v_old) < tol && break   # convergence check
    end
    d = u' * X * v                  #  same d update as before, but now with this new u and v; this is the same Step 3 as before, but now with the new u and v that have been computed with the orthogonality constraint on u
    return d, u, v
end

# Sparse PCA with orthogonal u's (Section 3.2). No deflation — every component
# is computed on the ORIGINAL Xc, with orthogonality enforced on the u side.
function pmd_orth(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                  tol = 1e-6, maxiter = 500)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c must be in [1, √p] = [1, $(sqrt(p))], you entered $c"))

    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'
    T  = eltype(Xc)

    V = zeros(T, p, k)
    d = zeros(T, k)
    U = zeros(T, n, 0)                          # collects the u's; starts empty
    for j in 1:k
        dj, uj, vj = spca_component_orth(Xc, c, U; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        U = hcat(U, uj)                         # add this new u to the collection of previous u's; this ensures that on the next iteration, the new u will be orthogonal to all previous u's
    end

    SignConsistency!(V)
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pcaStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end