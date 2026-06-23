
function SignConsistency_opt!(V)
    @inbounds for c in eachcol(V)
        mi = 1; mv = abs(c[1])
        for i in 2:length(c)    
            a = abs(c[i])
            if a > mv; mv = a; mi = i; end
        end
        s = sign(c[mi])
        s != 0 && (c .*= s)  
    end
    return V
end


