struct plskernStructure{T}
    W::Matrix{T}      
    P::Matrix{T}     
    Q::Matrix{T}      
    R::Matrix{T}      
    T::Matrix{T}      
    xmeans::Vector{T}
    xscales::Vector{T}
    ymeans::Vector{T}
    yscales::Vector{T}
end



function plskern(X, Y; nlv = 2, standardize = false, method = :algo1)
    X = Matrix{Float64}(X)
    Y = Y isa AbstractVector ? reshape(Float64.(Y), :, 1) : Matrix{Float64}(Y)
    n, p = size(X)
    q    = size(Y, 2)
    nlv  = min(nlv, n, p)
    method in (:algo1, :algo2) || throw(ArgumentError("method must be :algo1 or :algo2, you entered :$method"))
    T_ = Float64

    
    xmeans  = vec(mean(X, dims = 1))
    ymeans  = vec(mean(Y, dims = 1))
    xscales = standardize ? vec(std(X, dims = 1)) : ones(T_, p)
    yscales = standardize ? vec(std(Y, dims = 1)) : ones(T_, q)
    Xc = (X .- xmeans') ./ xscales'
    Yc = (Y .- ymeans') ./ yscales'

    
    XtY = Xc' * Yc                                   
    XtX = method === :algo2 ? Xc' * Xc : nothing    

    # result matrices
    W  = zeros(T_, p, nlv)
    P  = zeros(T_, p, nlv)
    Q  = zeros(T_, q, nlv)
    R  = zeros(T_, p, nlv)
    Tt = zeros(T_, n, nlv)

    
    w    = Vector{T_}(undef, p)
    r    = Vector{T_}(undef, p)
    t    = Vector{T_}(undef, n)
    pbuf = Vector{T_}(undef, p)
    qbuf = Vector{T_}(undef, q)

    Xct = transpose(Xc)                               

    for a in 1:nlv
        if q == 1
            @views w .= XtY[:, 1]                     
        else
            w .= svd(XtY).U[:, 1]                      
        end
        w ./= norm(w)                                  

    
        copyto!(r, w)
        for j in 1:(a - 1)
            r .-= dot(@view(P[:, j]), w) .* @view(R[:, j])
        end

       
        if method === :algo1
            mul!(t, Xc, r)                             
            tt = dot(t, t)
            mul!(pbuf, Xct, t); pbuf ./= tt            
            @views Tt[:, a] .= t
        else
            mul!(pbuf, XtX, r)                         
            tt = dot(r, pbuf)                          
            pbuf ./= tt                                
            mul!(t, Xc, r); @views Tt[:, a] .= t       
        end
        mul!(qbuf, transpose(XtY), r); qbuf ./= tt     

        
        BLAS.ger!(-tt, pbuf, qbuf, XtY)                

       
        @views W[:, a] .= w
        @views P[:, a] .= pbuf
        @views Q[:, a] .= qbuf
        @views R[:, a] .= r
    end

    return plskernStructure{T_}(W, P, Q, R, Tt, xmeans, xscales, ymeans, yscales)
end



function plskerncoef(m::plskernStructure; nlv = size(m.R, 2))
    nlv = min(nlv, size(m.R, 2))
    B = (m.R[:, 1:nlv] ./ m.xscales) * (m.Q[:, 1:nlv]') .* m.yscales'   
    intercept = m.ymeans' .- m.xmeans' * B 
    return B, intercept
end


function plskernpredict(m::plskernStructure, Xnew; nlv = size(m.R, 2))
    Xnew = Matrix{Float64}(Xnew)
    B, intercept = plskerncoef(m; nlv = nlv)
    return intercept .+ Xnew * B    
end


function plskerntransform(m::plskernStructure, Xnew; nlv = size(m.R, 2))
    Xnew = Matrix{Float64}(Xnew)              
    nlv  = min(nlv, size(m.R, 2))            
    Xc   = (Xnew .- m.xmeans') ./ m.xscales'  
    return Xc * m.R[:, 1:nlv]                 
end