struct sccaStructure{T}
    u::Matrix{T}        
    v::Matrix{T}        
    d::Vector{T}        
    cors::Vector{T}     
    penaltyx::Float64   
    penaltyz::Float64   
    K::Int
end





function _softcca!(out, a, d)
    @inbounds @simd for i in eachindex(a)
        ai = a[i]
        out[i] = sign(ai) * max(abs(ai) - d, 0.0)
    end
    return out
end


function _l2n_val(a)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += a[i] * a[i]
    end
    r = sqrt(s)
    return r == 0 ? 0.05 : r
end




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


function _l1_of_norm(a)
    nrm = _l2n_val(a)
    s1 = 0.0
    @inbounds @simd for i in eachindex(a)
        s1 += abs(a[i])
    end
    return s1 / nrm
end


function _l1diff(a, b)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += abs(a[i] - b[i])
    end
    return s
end

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


function _matsqrt(A)
    E = eigen(Symmetric(A))
    vals = sqrt.(max.(E.values, 0.0))            
    return E.vectors * Diagonal(vals) * transpose(E.vectors)
end


function _fast_init_v(x, z, K)
    xx = x * transpose(x)                       
    xx_sqrt = _matsqrt(xx)                      
    y = transpose(z) * xx_sqrt                   
    F = svd(y)                                   
    return F.U[:, 1:K]                           
end


function _sparse_cca_single_opt!(u, v, x, z, v0, penaltyx, penaltyz, niter,
                                 vold, zv_buf, xu_buf, argu, argv, su, sv)
    nr = size(x, 1)                              
    p1 = size(x, 2); p2 = size(z, 2)
    c1 = penaltyx * sqrt(p1)
    c2 = penaltyz * sqrt(p2)
    zv = @view zv_buf[1:nr]                     
    xu = @view xu_buf[1:nr]
    copyto!(v, v0)
    @inbounds for i in eachindex(vold); vold[i] = randn(); end   
    fill!(u, 0.0)
    for _ in 1:niter
        if _l1diff(vold, v) > 1e-6
            mul!(zv, z, v)                       
            mul!(argu, transpose(x), zv)        
            lamu = _binary_search_opt(argu, c1)
            _softcca!(su, argu, lamu)
            su ./= _l2n_val(su)
            copyto!(u, su)
            copyto!(vold, v)
            mul!(xu, x, u)                        
            mul!(argv, transpose(z), xu)         
            lamv = _binary_search_opt(argv, c2)
            _softcca!(sv, argv, lamv)
            sv ./= _l2n_val(sv)
            copyto!(v, sv)
        end
    end
    mul!(zv, z, v); mul!(xu, x, u)
    return dot(xu, zv)                           
end







function scca(X::Matrix{Float64}, Y::Matrix{Float64};
                  penaltyx::Real=0.3, penaltyz::Real=0.3,
                  K::Int=1, niter::Int=15, standardize::Bool=true)
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must share the number of columns (observations)."))
    dx >= 2 && dy >= 2 || throw(ArgumentError("need at least two features in each of X and Y"))
    (0 < penaltyx <= 1 && 0 < penaltyz <= 1) || throw(ArgumentError("penaltyx, penaltyz must be in (0,1]"))
    1 <= K <= min(dx, dy) || throw(ArgumentError("K must be in 1:min(dx,dy)"))

    
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

    Vinit = if dx > n && dy > n
        _fast_init_v(x, z, K)
    else
        svd(transpose(x) * z).V[:, 1:K]
    end

    U = zeros(dx, K); V = zeros(dy, K); D = zeros(K); C = zeros(K)

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

    xres = copy(x); zres = copy(z)               
    for k in 1:K
        d = _sparse_cca_single_opt!(u, v, xres, zres, @view(Vinit[:, k]),
                                    penaltyx, penaltyz, niter,
                                    vold, zv, xu, argu, argv, su, sv)
        @views U[:, k] .= u
        @views V[:, k] .= v
        D[k] = d
        if any(!iszero, u) && any(!iszero, v)
            C[k] = cor(x * u, z * v)
        end
        if k < K
            xres = vcat(xres, sqrt(d) .* transpose(u))
            zres = vcat(zres, -sqrt(d) .* transpose(v))
        end
    end

    return sccaStructure(U, V, D, C, Float64(penaltyx), Float64(penaltyz), K)
end