

struct spcStructure{T}
    mean::Vector{T}
    scale::Vector{T}
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end

function l1_diff(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        s += abs(a[i] - b[i])
    end
    return s
end

function finding_v!(v, s, z, c)
    Tz = eltype(z)
    nz = norm(z); iszero(nz) && (nz = Tz(0.05))
    @. v = z / nz
    sum(abs, v) <= c && return v
    lo = zero(Tz)
    hi = maximum(abs, z) - Tz(1e-5)
    for _ in 1:150
        delta = (lo + hi) / 2
        @. s = sign(z) * max(abs(z) - delta, zero(Tz))
        ns = norm(s); iszero(ns) && (ns = Tz(0.05))
        @. v = s / ns
        sum(abs, v) < c ? (hi = delta) : (lo = delta)
        (hi - lo) < Tz(1e-6) && break
    end
    return v
end

# Top-k right singular vectors via Gram-matrix partial eigen (matches R's
# CheckPMDV). Avoids svd's n×p U factor; returns only the k vectors needed.
function init_rsv(Xc, k)
    n, p = size(Xc); T = eltype(Xc)
    if p <= n
        E = eigen(Symmetric(transpose(Xc) * Xc), p-k+1:p)   # top-k only
        return E.vectors[:, k:-1:1]                          # p×k, descending
    else
        E  = eigen(Symmetric(Xc * transpose(Xc)), n-k+1:n)
        Vk = transpose(Xc) * E.vectors[:, k:-1:1]            # p×k
        @inbounds for j in 1:k
            nrm = norm(@view Vk[:, j]); iszero(nrm) && (nrm = T(0.05))
            @views Vk[:, j] ./= nrm
        end
        return Vk
    end
end

# Closed-form cumulative PVE: ‖proj_k‖²_F = tr(Mₖ⁻¹ Bₖ), all k×k.
# Algebraically identical to projecting Xc, but never forms an n×p matrix.
function prop_var_explained(Xc, V)
    K = size(V, 2); T = eltype(Xc)
    totsq = sum(abs2, Xc)
    A = Xc * V              # n×K, once
    M = transpose(V) * V    # K×K
    B = transpose(A) * A    # K×K
    pve = Vector{T}(undef, K)
    @inbounds for k in 1:K
        pve[k] = tr(Symmetric(M[1:k, 1:k]) \ B[1:k, 1:k]) / totsq
    end
    return pve
end

# Buffer-based: nothing allocated inside; u/v live in caller-owned buffers.
function spca_component!(v, X, c, u, Xv, Xtu, s, vold; tol = 1e-7, niter = 20)
    T = eltype(v)
    for _ in 1:niter
        copyto!(vold, v)
        mul!(u, X, v); nu = norm(u); iszero(nu) && (nu = T(0.05)); u ./= nu
        mul!(Xtu, transpose(X), u)
        finding_v!(v, s, Xtu, c)
        l1_diff(v, vold) < tol && break
    end
    mul!(Xv, X, v)
    return dot(u, Xv)
end

function spca_component_orth!(v, X, c, U_prev, u, uold, Xv, Xtu, s, vold, proj;
                              tol = 1e-6, niter = 20)
    T = eltype(v); fill!(u, zero(T)); m = size(U_prev, 2)
    for _ in 1:niter
        copyto!(vold, v); copyto!(uold, u)
        mul!(u, X, v)
        if m > 0
            pj = view(proj, 1:m)
            mul!(pj, transpose(U_prev), u)
            mul!(u, U_prev, pj, -one(T), one(T))   # u -= U_prev (U_prevᵀu)
        end
        nu = norm(u); iszero(nu) && (nu = T(0.05)); u ./= nu
        mul!(Xtu, transpose(X), u)
        finding_v!(v, s, Xtu, c)
        (l1_diff(v, vold) < tol && l1_diff(u, uold) < tol) && break
    end
    mul!(Xv, X, v)
    return dot(u, Xv)
end

function spc(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                 tol = 1e-7, niter = 20)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c (=sumabsv) must be in [1, √p]=[1,$(sqrt(p))], got $c"))
    T  = eltype(float(X))
    means = T.(vec(mean(X, dims = 1)))                 # per-column mean (matches scale(center=TRUE))
    sigma = standardize ? T.(vec(std(X, dims = 1))) : ones(T, p)
    Xc = standardize ? (X .- means') ./ sigma' : T.(X .- means')

    Vinit = init_rsv(Xc, k)
    Rmat  = copy(Xc)                                   # deflation workspace
    V = zeros(T, p, k); d = zeros(T, k)
    u = Vector{T}(undef, n); Xv = Vector{T}(undef, n)  # buffers allocated ONCE
    Xtu = Vector{T}(undef, p); s = Vector{T}(undef, p)
    vold = Vector{T}(undef, p); v = Vector{T}(undef, p)
    for j in 1:k
        copyto!(v, view(Vinit, :, j))
        d[j] = spca_component!(v, Rmat, c, u, Xv, Xtu, s, vold; tol = tol, niter = niter)
        @views V[:, j] .= v
        BLAS.ger!(-d[j], u, v, Rmat)                   # deflate in place
    end
    SignConsistency_opt!(V)
    vars = d .^ 2 ./ (n - 1)
    return spcStructure{T}(means, sigma, V, vars, prop_var_explained(Xc, V))
end

function spc_orth(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                      tol = 1e-6, niter = 20)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c (=sumabsv) must be in [1, √p]=[1,$(sqrt(p))], got $c"))
    T  = eltype(float(X))
    means = T.(vec(mean(X, dims = 1)))                 # per-column mean
    sigma = standardize ? T.(vec(std(X, dims = 1))) : ones(T, p)
    Xc = standardize ? (X .- means') ./ sigma' : T.(X .- means')

    Vinit = init_rsv(Xc, k)
    V = zeros(T, p, k); d = zeros(T, k); U = Matrix{T}(undef, n, k)
    u = Vector{T}(undef, n); uold = Vector{T}(undef, n); Xv = Vector{T}(undef, n)
    Xtu = Vector{T}(undef, p); s = Vector{T}(undef, p)
    vold = Vector{T}(undef, p); v = Vector{T}(undef, p); proj = Vector{T}(undef, k)
    for j in 1:k
        copyto!(v, view(Vinit, :, j))
        Uprev = @view U[:, 1:j-1]
        d[j] = spca_component_orth!(v, Xc, c, Uprev, u, uold, Xv, Xtu, s, vold, proj;
                                    tol = tol, niter = niter)
        @views V[:, j] .= v
        @views U[:, j] .= u
    end
    SignConsistency_opt!(V)
    vars = d .^ 2 ./ (n - 1)
    return spcStructure{T}(means, sigma, V, vars, prop_var_explained(Xc, V))
end