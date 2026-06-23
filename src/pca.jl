
struct pcaStructure{T}
    mean::Vector{T}
    scale::Vector{T}   
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end








function pca_transform(m::pcaStructure, X)
    Xc = (X .- m.mean') ./ m.scale'    
    return Xc * m.loadings             
end


function pca_invtransform(m::pcaStructure, scores)
    Xc = scores * m.loadings'          
    return Xc .* m.scale' .+ m.mean'   
end






function pca(X; k = minimum(size(X)), standardize = false, method = :svd)
    n, p = size(X)
    1 <= k <= min(n, p) || throw(ArgumentError("$k you chose is not in range. Please reselect $k"))
    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'                 
    T  = eltype(Xc)
    total = sum(abs2, Xc) / (n - 1)

    if method === :cov
        C = Symmetric(Xc'Xc)                      
        F = eigen(C)                             
        idx = p:-1:(p-k+1)                         
        vars = @view(F.values[idx]) ./ (n - 1)    
        V = F.vectors[:, idx]                      
        vars = collect(vars)
    elseif method === :svd
        F = svd(Xc; full = false)                  
        vars = @view(F.S[1:k]) .^ 2 ./ (n - 1)
        V = F.V[:, 1:k]
        vars = collect(vars)
    else
        error("unknown method :$method")
    end

    SignConsistency_opt!(V)
    return pcaStructure{T}(T.(means), T.(sigma), Matrix(V), vars, vars ./ total)
end