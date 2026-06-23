



struct pmdStructure{T}
    u::Matrix{T}         
    v::Matrix{T}         
    d::Vector{T}         
    sumabsu::Float64
    sumabsv::Float64
    K::Int
    meanx::Float64       
end


_pmd_soft(a, λ) = sign(a) * max(abs(a) - λ, 0.0)


function _pmd_l2n(a)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += a[i] * a[i]
    end
    r = sqrt(s)
    return r == 0 ? 0.05 : r
end


function _pmd_l1_norm_soft(a, λ)
    s2 = 0.0; s1 = 0.0
    @inbounds @simd for i in eachindex(a)
        si = sign(a[i]) * max(abs(a[i]) - λ, 0.0)
        s2 += si * si; s1 += abs(si)
    end
    nrm = sqrt(s2); nrm == 0 && (nrm = 0.05)
    return s1 / nrm
end


function _pmd_l1_norm(a)
    nrm = _pmd_l2n(a); s1 = 0.0
    @inbounds @simd for i in eachindex(a)
        s1 += abs(a[i])
    end
    return s1 / nrm
end


function _pmd_l1diff(a, b)
    s = 0.0
    @inbounds @simd for i in eachindex(a)
        s += abs(a[i] - b[i])
    end
    return s
end


function _pmd_binary_search(argu, sumabs)
    (_pmd_l2n(argu) == 0 || _pmd_l1_norm(argu) <= sumabs) && return 0.0
    lam1 = 0.0
    lam2 = maximum(abs, argu) - 1e-5
    iter = 1
    while iter < 150
        mid = (lam1 + lam2) / 2
        _pmd_l1_norm_soft(argu, mid) < sumabs ? (lam2 = mid) : (lam1 = mid)
        (lam2 - lam1) < 1e-6 && return (lam1 + lam2) / 2
        iter += 1
    end
    return (lam1 + lam2) / 2
end


function _pmd_soft_normalize!(out, arg, λ)
    nrm = 0.0
    @inbounds @simd for i in eachindex(arg)
        si = sign(arg[i]) * max(abs(arg[i]) - λ, 0.0)
        out[i] = si
        nrm += si * si
    end
    nrm = sqrt(nrm); nrm == 0 && (nrm = 0.05)
    @inbounds @simd for i in eachindex(out)
        out[i] /= nrm
    end
    return out
end


function _pmd_check_v(x, K)
    n, p = size(x)
    if p > n
        F = svd(x * transpose(x))                 
        V = transpose(x) * F.V[:, 1:K]            
        for j in 1:K
            V[:, j] ./= _pmd_l2n(@view V[:, j])
        end
        return V
    else
        F = svd(transpose(x) * x)                
        return F.V[:, 1:K]
    end
end



function _pmd_smd!(x, v0, sumabsu, sumabsv, niter, u, v, vold, argu, argv)
    copyto!(v, v0)
    @inbounds for i in eachindex(vold); vold[i] = randn(); end   # PMA loop primer
    fill!(u, 0.0)
    for _ in 1:niter
        if _pmd_l1diff(vold, v) > 1e-7
            copyto!(vold, v)
            mul!(argu, x, v)                          
            lamu = _pmd_binary_search(argu, sumabsu)
            _pmd_soft_normalize!(u, argu, lamu)       
            mul!(argv, transpose(x), u)               
            lamv = _pmd_binary_search(argv, sumabsv)
            _pmd_soft_normalize!(v, argv, lamv)       
        end
    end
    mul!(argu, x, v)                                  
    return dot(u, argu)                               
end


function pmd(X::Matrix{Float64}; sumabs::Real=0.4,
             sumabsu::Union{Nothing,Real}=nothing,
             sumabsv::Union{Nothing,Real}=nothing,
             K::Int=1, niter::Int=20, center::Bool=true)
    n, p = size(X)
    1 <= K <= min(n, p) || throw(ArgumentError("K must be in 1:min(n,p)"))


    if sumabsu === nothing || sumabsv === nothing
        0 < sumabs <= 1 || throw(ArgumentError("sumabs must be in (0,1]"))
        su = sqrt(n) * sumabs
        sv = sqrt(p) * sumabs
    else
        su = Float64(sumabsu); sv = Float64(sumabsv)
    end
    1 <= su <= sqrt(n) || throw(ArgumentError("sumabsu must be in [1,√n]=[1,$(sqrt(n))]"))
    1 <= sv <= sqrt(p) || throw(ArgumentError("sumabsv must be in [1,√p]=[1,$(sqrt(p))]"))

    meanx = center ? mean(X) : NaN                
    Xc = center ? X .- meanx : copy(X)

    U = zeros(n, K); V = zeros(p, K); D = zeros(K)
    Vinit = _pmd_check_v(Xc, K)

 
    u    = Vector{Float64}(undef, n)
    v    = Vector{Float64}(undef, p)
    vold = Vector{Float64}(undef, p)
    argu = Vector{Float64}(undef, n)
    argv = Vector{Float64}(undef, p)

    R = copy(Xc)                                 
    for k in 1:K
        d = _pmd_smd!(R, @view(Vinit[:, k]), su, sv, niter, u, v, vold, argu, argv)
        @views U[:, k] .= u
        @views V[:, k] .= v
        D[k] = d
        BLAS.ger!(-d, u, v, R)                    
    end

    return pmdStructure(U, V, D, su, sv, K, center ? meanx : NaN)
end