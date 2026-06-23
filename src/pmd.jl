struct pmdStructure{T}
    mean::Vector{T}
    scale::Vector{T}
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end






function norm_diff(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        d = a[i] - b[i]
        s += d * d
    end
    return sqrt(s)
end


function finding_v!(v, s, z, c)
    nz = norm(z)
    @. v = z / nz                                  
    sum(abs, v) <= c && return v                  
    lo = zero(eltype(z)); hi = maximum(abs, z)
    for _ in 1:100                                 
        delta = (lo + hi) / 2
        @. s = sign(z) * max(abs(z) - delta, zero(eltype(z)))   
        ns = norm(s)
        if ns == 0
            @. v = s
        else
            @. v = s / ns                          
        end
        sum(abs, v) < c ? (hi = delta) : (lo = delta)
    end
    return v
end



function spca_component_opt(X, c; tol = 1e-6, maxiter = 500)
    n, p = size(X)
    T = eltype(X)
    v    = randn(T, p); v ./= norm(v)
    vold = similar(v)
    u    = Vector{T}(undef, n)
    Xv   = Vector{T}(undef, n)         
    Xtu  = Vector{T}(undef, p)         
    s    = Vector{T}(undef, p)        

    
    for _ in 1:5
        mul!(Xv, X, v)                  
        mul!(v, transpose(X), Xv)      
        v ./= norm(v)
    end

    for _ in 1:maxiter
        copyto!(vold, v)
        mul!(u, X, v); u ./= norm(u)            
        mul!(Xtu, transpose(X), u)              
        finding_v!(v, s, Xtu, c)                
        norm_diff(v, vold) < tol && break
    end

    mul!(Xv, X, v)
    d = dot(u, Xv)                             
    return d, u, v
end


function pmd(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                 tol = 1e-6, maxiter = 500)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c must be in [1, √p] = [1, $(sqrt(p))], you entered $c"))

    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'
    T  = eltype(Xc)

    R = copy(Xc)                                
    V = zeros(T, p, k)
    d = zeros(T, k)
    for j in 1:k
        dj, uj, vj = spca_component_opt(R, c; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        BLAS.ger!(-dj, uj, vj, R)               
    end

    SignConsistency_opt!(V)
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pmdStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end


function spca_component_orth_opt(X, c, U_prev; tol = 1e-6, maxiter = 500)
    n, p = size(X)
    T = eltype(X)
    v    = randn(T, p); v ./= norm(v)
    vold = similar(v)
    u    = Vector{T}(undef, n)
    Xv   = Vector{T}(undef, n)
    Xtu  = Vector{T}(undef, p)
    s    = Vector{T}(undef, p)
    proj = isempty(U_prev) ? nothing : Vector{T}(undef, size(U_prev, 2))  # buffer for U'u

    for _ in 1:5
        mul!(Xv, X, v)
        mul!(v, transpose(X), Xv)
        v ./= norm(v)
    end

    for _ in 1:maxiter
        copyto!(vold, v)
        mul!(u, X, v)                           
        if !isempty(U_prev)                      
            mul!(proj, transpose(U_prev), u)     
            mul!(u, U_prev, proj, -1.0, 1.0)     
        end
        u ./= norm(u)
        mul!(Xtu, transpose(X), u)
        finding_v!(v, s, Xtu, c)
        norm_diff(v, vold) < tol && break
    end

    mul!(Xv, X, v)
    d = dot(u, Xv)
    return d, u, v
end



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
    U = Matrix{T}(undef, n, k)                  
    for j in 1:k
        Uprev = @view U[:, 1:j-1]               
        dj, uj, vj = spca_component_orth_opt(Xc, c, Uprev; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        U[:, j] = uj                           
    end

    SignConsistency_opt!(V)
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pmdStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end