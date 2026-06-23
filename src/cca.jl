



struct ccaStructure{T}
    xmean::Vector{T}    
    ymean::Vector{T}     
    xproj::Matrix{T}     
    yproj::Matrix{T}     
    corrs::Vector{T}     
    nobs::Int            
end




function cca_transform(M::ccaStructure, Z::AbstractMatrix, c::Symbol)
    if c === :x
        return transpose(M.xproj) * (Z .- M.xmean)
    elseif c === :y
        return transpose(M.yproj) * (Z .- M.ymean)
    else
        throw(ArgumentError("component must be :x or :y"))
    end
end





function _cca_svd_opt(Zx, Zy, xmean, ymean, p::Int)
    n = size(Zx, 2)

    Sx = svd!(Zx)                             
    Sy = svd!(Zy)

    inner = Sx.Vt * transpose(Sy.Vt)
    S = svd!(inner)

    ord = sortperm(S.S; rev=true)
    si  = ord[1:p]

    scale = sqrt(n - 1)
    rmul!(Sx.U, Diagonal(scale ./ Sx.S))      
    rmul!(Sy.U, Diagonal(scale ./ Sy.S))
    Px = Sx.U * @view S.U[:, si]
    Py = Sy.U * @view S.V[:, si]

    corrs = S.S[si]
    return ccaStructure(xmean, ymean, Px, Py, corrs, n)
end



function _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, p::Int)
    dx = size(Cxx, 1)
    dy = size(Cyy, 1)

    if dx <= dy
        G  = cholesky(Symmetric(Cyy)) \ transpose(Cxy)    
        A  = Cxy * G                                       
        E  = eigen(Symmetric(A), Symmetric(Cxx))
        ord  = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Px = E.vectors[:, ord]                             
        Py = G * Px                                        
        _qnormalize!(Py, Cyy)                              
    else
        
        H  = cholesky(Symmetric(Cxx)) \ Cxy                
        A  = transpose(Cxy) * H                            
        E  = eigen(Symmetric(A), Symmetric(Cyy))
        ord  = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Py = E.vectors[:, ord]
        Px = H * Py
        _qnormalize!(Px, Cxx)
    end

    corrs = sqrt.(clamp.(eigs, 0.0, Inf))    
    return ccaStructure(xmean, ymean, Px, Py, corrs, -1)
end


function _qnormalize!(P, C)
    d, p = size(P)
    cp = Vector{eltype(P)}(undef, d)          
    @inbounds for j in 1:p
        pj = @view P[:, j]
        mul!(cp, C, pj)                       
        s = sqrt(dot(pj, cp))
        pj ./= s
    end
    return P
end



function cca(X::Matrix{Float64}, Y::Matrix{Float64};
                 method::Symbol=:svd, outdim::Int=min(size(X,1), size(Y,1)))
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must have the same number of columns."))
    1 <= outdim <= min(dx, dy) || throw(ArgumentError("outdim must be in 1:min(dx,dy)"))
    (n > dx && n > dy) || @warn "CCA unstable when n ≤ dx or n ≤ dy (n=$n, dx=$dx, dy=$dy)."

    xmean = vec(mean(X, dims=2))
    ymean = vec(mean(Y, dims=2))
    Zx = X .- xmean                           
    Zy = Y .- ymean

    if method === :svd
        return _cca_svd_opt(Zx, Zy, xmean, ymean, outdim)
    elseif method === :cov
        Cxx = rmul!(Zx * transpose(Zx), 1.0 / (n - 1))
        Cyy = rmul!(Zy * transpose(Zy), 1.0 / (n - 1))
        Cxy = rmul!(Zx * transpose(Zy), 1.0 / (n - 1))
        return _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, outdim)
    else
        throw(ArgumentError("method must be :svd or :cov"))
    end
end