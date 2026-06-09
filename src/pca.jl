# Abhisek Banerjee
# Principal Component Analysis function



# Two packages: LinearAlgebra for matrix operations, and Statistics for mean and std.
# using LinearAlgebra, Statistics --- IGNORE --- (moved to BRMB.jl)

# A structure to hold the PCA model parameters: mean, scale, loadings (principal directions), variances, and proportion of variance explained.
struct pcaStructure{T}
    mean::Vector{T}
    scale::Vector{T}   # standard deviations used for scaling (or ones if no scaling)
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end

# Now assume X is an n×p matrix of observations (rows) and features (columns).
# Now we create a function pca which will compute the PCA model parameters based on the specified method.
# The method can be svd or cov. Here I have set the default to svd.
# method = :svd or :cov.
function pca(X; k = minimum(size(X)), standardize = false, method = :svd)
    n, p = size(X) # number of observations and features
    1 <= k <= min(n, p) || throw(ArgumentError("$k you chose is not in range. Please reselect $k")) # sanity check on k
    means = vec(mean(X, dims = 1)) # compute column means (p-vector)
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p) # compute column standard deviations (p-vector)
    Xc = (X .- means') ./ sigma'                 # center (and scale) in one fused pass
    T  = eltype(Xc)            # Type of the centered or scaled  data
    total = sum(abs2, Xc) / (n - 1)     # total variance in Xc (after centering and scaling) Basically XcXc'/(n-1) is the covariance matrix of the centered data
    if method === :cov
        vals, vecs = eigen(Symmetric(Xc'Xc ./ (n - 1))) # eigen decomposition of the covariance matrix (p×p) to get eigenvalues and eigenvectors
        ord  = sortperm(vals, rev = true)[1:k]   # indices of the top k eigenvalues (sorted in descending order)
        vars = vals[ord]                     # variances explained by the top k components (the eigenvalues)
        V    = vecs[:, ord]            # loadings (the eigenvectors corresponding to the top k eigenvalues)
    elseif method === :svd
        F    = svd(Xc)                     # singular value decomposition of the centered (and scaled) data matrix Xc
        vars = F.S[1:k] .^ 2 ./ (n - 1)     # variances explained by the top k components (the squared singular values divided by n-1)
        V    = F.V[:, 1:k]                 # loadings (the right singular vectors corresponding to the top k singular values)
    else
        error("unknown method :$method")     # if user chose an invalid method, throw an error
    end

    SignConsistency!(V)                        # since sign of a PC is arbitrary we need to make the sign consistent 
    return pcaStructure{T}(T.(means), T.(sigma), Matrix(V), vars, vars ./ total)
end

# Since the sign of a principal component is arbitrary, 
# we can make it consistent by ensuring that the largest absolute value in each loading vector is positive. 
# This function modifies the loadings matrix V in place to achieve this consistency.

function SignConsistency!(V)
    for c in eachcol(V)
        c .*= sign(c[argmax(abs.(c))])
    end
    return V
end

# project data onto the components 
function pca_transform(m::pcaStructure, X)
    Xc = (X .- m.mean') ./ m.scale'    # center and scale using the stored stats
    return Xc * m.loadings             # project onto the principal directions
end

# rough reconstruction back in the original units
function pca_invtransform(m::pcaStructure, scores)
    Xc = scores * m.loadings'          # back to full feature width (centered space)
    return Xc .* m.scale' .+ m.mean'   # undo the scaling, then undo the centering
end










