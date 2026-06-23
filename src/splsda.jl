
struct splsdaStructure{T}
    variates_X::Matrix{T}     
    variates_Y::Matrix{T}     
    loadings_X::Matrix{T}    
    loadings_Y::Matrix{T}     
    ncomp::Int
    keepX::Vector{Int}
    Y_dummy::Matrix{T}       
    classes::Vector           
end


function _center_scale(M::AbstractMatrix; scale=true)
    Mc = M .- mean(M, dims=1)
    if scale
        s = std(M, dims=1; corrected=true)        
        s[s .== 0] .= 1.0                           
        Mc = Mc ./ s
        zerocols = vec(std(M, dims=1; corrected=true) .== 0)
        Mc[:, zerocols] .= 0.0
    end
    return Mc
end


function _unmap(y::AbstractVector; levels=nothing)
    classes = levels === nothing ? sort(unique(y)) : collect(levels)  
    length(classes) == length(unique(y)) ||
        throw(ArgumentError("`levels` must list each class exactly once (got $(length(classes)) levels for $(length(unique(y))) classes)"))
    k = length(classes)
    Yd = zeros(Float64, length(y), k)
    for (i, yi) in enumerate(y)
        idx = findfirst(==(yi), classes) 
        idx === nothing && throw(ArgumentError("class $(yi) not found in supplied `levels`"))
        Yd[i, idx] = 1.0
    end
    return Yd, classes
end

function _soft_threshold_L1!(out, x, nx::Int, absx, ord, ranks)
    p = length(x)
    if nx <= 0
        copyto!(out, x)
        return out
    end
    @. absx = abs(x)
    sortperm!(ord, absx)                      
    fill!(ranks, 0)
    i = 1
    while i <= p
        j = i
        while j < p && absx[ord[j+1]] == absx[ord[i]]
            j += 1
        end
        for m in i:j; ranks[ord[m]] = j; end   
        i = j + 1
    end
    lambda = 0.0; anydrop = false
    @inbounds for t in 1:p
        if ranks[t] <= nx
            anydrop = true
            absx[t] > lambda && (lambda = absx[t])
        end
    end
    if !anydrop                                
        copyto!(out, x)
        return out
    end
    @inbounds for t in 1:p
        out[t] = ranks[t] > nx ? sign(x[t]) * (absx[t] - lambda) : 0.0
    end
    return out
end

function _sqdiff(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        d = a[i] - b[i]; s += d * d
    end
    return s
end

function splsda(X::Matrix{Float64}, y::Vector, ncomp::Int, keepX::Vector{Int};
                    scale=true, tol=1e-6, max_iter=100, levels=nothing)
    n, p = size(X)
    Yd, classes = _unmap(y; levels=levels)
    k = size(Yd, 2)
    length(keepX) == ncomp || throw(ArgumentError("keepX must have length ncomp"))

    Xc = _center_scale(Matrix{Float64}(X); scale=scale)
    Yc = _center_scale(Yd; scale=scale)

    TX = zeros(n, ncomp); TY = zeros(n, ncomp)
    PX = zeros(p, ncomp); PY = zeros(k, ncomp)

    R  = copy(Xc)                              
    Ry = copy(Yc)                              

    M     = Matrix{Float64}(undef, p, k)
    uh    = Vector{Float64}(undef, p); uh_old = Vector{Float64}(undef, p)
    vh    = Vector{Float64}(undef, k); vh_old = Vector{Float64}(undef, k)
    tX    = Vector{Float64}(undef, n)
    tY    = Vector{Float64}(undef, n)
    uraw  = Vector{Float64}(undef, p)         
    pX    = Vector{Float64}(undef, p)
    absx  = Vector{Float64}(undef, p)
    ord   = Vector{Int}(undef, p)
    ranks = Vector{Int}(undef, p)

    for comp in 1:ncomp
        mul!(M, transpose(R), Ry)
        F = svd(M)
        copyto!(uh, @view F.U[:, 1])
        copyto!(vh, @view F.V[:, 1])
        copyto!(uh_old, uh); copyto!(vh_old, vh)

        iter = 1
        while true
            mul!(tY, Ry, vh)                          
            mul!(uraw, transpose(R), tY)             
            _soft_threshold_L1!(uh, uraw, p - keepX[comp], absx, ord, ranks)
            uh ./= sqrt(sum(abs2, uh))                
            mul!(tX, R, uh)                           
            mul!(vh, transpose(Ry), tX)              
            vh ./= sqrt(sum(abs2, vh))                

            dX = _sqdiff(uh, uh_old)
            dY = _sqdiff(vh, vh_old)
            (max(dX, dY) < tol || iter > max_iter) && break
            copyto!(uh_old, uh); copyto!(vh_old, vh)
            iter += 1
        end

        mul!(tX, R, uh); mul!(tY, Ry, vh)
        @views TX[:, comp] .= tX; @views TY[:, comp] .= tY
        @views PX[:, comp] .= uh; @views PY[:, comp] .= vh

        mul!(pX, transpose(R), tX)
        pX ./= dot(tX, tX)
        BLAS.ger!(-1.0, tX, pX, R)                    
        
    end

    return splsdaStructure(TX, TY, PX, PY, ncomp, keepX, Yd, classes)
end