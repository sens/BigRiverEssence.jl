# Test/spc_test.jl — formal tests for SPC (Witten sparse PCA): spc + spc_orth.

@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, k = 80, 60, 3
    X = randn(n, p)
    m = spc(X; k=k, c=0.6*sqrt(p))

    @test m isa spcStructure
    @test size(m.loadings) == (p, k)
    @test length(m.variances) == k
    @test length(m.propOFvar) == k
    @test length(m.mean) == p
    @test length(m.scale) == p

    # each loading column unit-norm (or zero)
    for j in 1:k
        nv = norm(m.loadings[:, j])
        @test isapprox(nv, 1.0; atol=1e-6) || nv == 0.0
    end
    @test all(isfinite, m.variances)
    @test all(isfinite, m.propOFvar)
    # cumulative PVE is nondecreasing and within [0, 1]
    @test all(0 .<= m.propOFvar .<= 1 + 1e-9)
    @test all(diff(m.propOFvar) .>= -1e-9)
    # column means recorded (default standardize=false ⇒ unit scale)
    @test isapprox(m.mean, vec(mean(X, dims=1)); atol=1e-10)
    @test all(isapprox.(m.scale, 1.0; atol=1e-12))
end

@testset "max budget reduces to ordinary PCA (theorem anchor)" begin
    # At c = √p the L1 penalty doesn't bind → no soft-thresholding → the loading
    # is the ordinary top PC of the column-centered data, fully dense.
    Random.seed!(7)
    n, p = 60, 40
    X = randn(n, p)
    Xc = X .- mean(X, dims=1)
    m = spc(X; k=1, c=sqrt(p))
    F = svd(Xc)
    @test abs(dot(m.loadings[:, 1], F.V[:, 1])) > 0.999
    @test count(!iszero, m.loadings[:, 1]) == p          # fully dense
end

@testset "sparsity contract (c controls sparsity)" begin
    Random.seed!(11)
    n, p = 80, 60
    X = randn(n, p)
    dense = spc(X; k=1, c=sqrt(p))
    mid   = spc(X; k=1, c=0.5*sqrt(p))
    tight = spc(X; k=1, c=1.5)
    @test count(!iszero, dense.loadings[:,1]) == p
    # smaller c ⇒ fewer (monotone) nonzeros
    @test count(!iszero, tight.loadings[:,1]) <= count(!iszero, mid.loadings[:,1]) <= p
    @test count(!iszero, tight.loadings[:,1]) < p        # tight actually zeros some
end

@testset "ground-truth: recovers planted sparse loading" begin
    # plant a sparse rank-1 signal; a tight budget gives high PRECISION (every
    # selected variable is real) — recall is budget-limited, so we assert on
    # precision + direction, not recall.
    Random.seed!(42)
    n, p = 200, 300
    u_true = [randn(50); zeros(n-50)]
    v_true = [randn(75); zeros(p-75)]
    X = u_true * v_true' .+ randn(n, p)
    m = spc(X; k=1, c=4.0)
    vh = m.loadings[:, 1]
    est = Set(findall(!iszero, vh)); truth = Set(1:75)
    prec = length(intersect(est, truth)) / length(est)
    @test prec > 0.95                                    # selected vars are real
    @test abs(dot(vh ./ norm(vh), v_true ./ norm(v_true))) > 0.6
    # loosening the budget improves recovery (recall + direction climb)
    m2 = spc(X; k=1, c=8.0)
    rec2 = length(intersect(Set(findall(!iszero, m2.loadings[:,1])), truth)) / 75
    @test rec2 > 0.9
end

@testset "spc_orth: scores are orthonormal" begin
    # the defining property of the orth variant: UᵀU ≈ I. spcStructure doesn't
    # store U, so recompute scores T = Xc·V; for the orthogonal variant the score
    # directions are orthogonal (off-diagonal correlations ≈ 0), unlike deflation.
    Random.seed!(5)
    n, p, k = 100, 50, 4
    X = randn(n, p)
    Xc = X .- mean(X, dims=1)
    mo = spc_orth(X; k=k, c=2.0)
    md = spc(X;      k=k, c=2.0)
    offdiag(C) = maximum(abs(C[i,j]) for i in 1:size(C,1) for j in 1:size(C,2) if i != j)
    Co = cor(Xc * mo.loadings)
    Cd = cor(Xc * md.loadings)
    # orthogonal variant has lower off-diagonal score correlation than deflation
    @test offdiag(Co) <= offdiag(Cd) + 1e-9
    # NOTE: a direct UᵀU ≈ I assertion needs spc_orth to store U. If you add a
    # `scores` field, replace the above with:
    #   @test maximum(abs.(mo.scores'mo.scores - I)) < 1e-10
end

@testset "multiple components, shapes" begin
    Random.seed!(5)
    n, p, k = 60, 50, 4
    X = randn(n, p)
    m = spc(X; k=k, c=2.0)
    @test size(m.loadings) == (p, k)
    @test length(m.variances) == k
    for j in 1:k
        @test count(!iszero, m.loadings[:,j]) < p
    end
end

@testset "standardize=true scales columns" begin
    Random.seed!(9)
    n, p = 60, 40
    X = randn(n, p) .* (1:p)'            # wildly unequal column scales
    m = spc(X; k=1, c=0.5*sqrt(p), standardize=true)
    @test isapprox(m.scale, vec(std(X, dims=1)); atol=1e-8)
    md = spc(X; k=1, c=0.5*sqrt(p), standardize=false)
    @test all(isapprox.(md.scale, 1.0; atol=1e-12))
end

@testset "determinism (SVD init, no random dependence)" begin
    Random.seed!(99)
    n, p = 60, 40
    X = randn(n, p)
    a = spc(X; k=2, c=0.5*sqrt(p))
    b = spc(X; k=2, c=0.5*sqrt(p))
    for j in 1:2
        @test abs(dot(a.loadings[:,j] ./ norm(a.loadings[:,j]),
                      b.loadings[:,j] ./ norm(b.loadings[:,j]))) > 0.999
        @test isapprox(a.variances[j], b.variances[j]; rtol=1e-6)
    end
end

@testset "argument validation" begin
    Random.seed!(0)
    n, p = 100, 16
    X = randn(n, p)
    @test_throws ArgumentError spc(X; k=1, c=0.5)            # c < 1
    @test_throws ArgumentError spc(X; k=1, c=sqrt(p)+1)      # c > √p
    @test_throws ArgumentError spc_orth(X; k=1, c=0.5)
    @test_throws ArgumentError spc_orth(X; k=1, c=sqrt(p)+1)
end

@testset "matches R PMA::SPC (offline reference fixtures)" begin
    refdir = joinpath(@__DIR__, "Data", "SPC")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "SPC R-reference fixtures not found; skipping. Run spc.R to create them."
    else
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart=1)
            row = findfirst(==("PMA_version"), sm[:, 1])
            row !== nothing && @info "SPC fixtures generated against PMA $(sm[row, 2])"
        end

        X    = readdlm(joinpath(refdir, "X.csv"),          ',', Float64; skipstart=1)
        v_r  = readdlm(joinpath(refdir, "v_spc.csv"),      ',', Float64; skipstart=1)
        d_r  = vec(readdlm(joinpath(refdir, "d_spc.csv"),  ',', Float64; skipstart=1))
        vo_r = readdlm(joinpath(refdir, "v_spc_orth.csv"), ',', Float64; skipstart=1)
        do_r = vec(readdlm(joinpath(refdir, "d_spc_orth.csv"), ',', Float64; skipstart=1))
        meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart=1)
        n = Int(meta[1]); K = Int(meta[3]); sv = meta[4]

        # X.csv is RAW (uncentered); spc column-centers internally to match R.
        m  = spc(X;      k=K, c=sv)
        mo = spc_orth(X; k=K, c=sv)
        d_jl  = sqrt.(max.(m.variances,  0) .* (n - 1))   # spcStructure has no d field
        do_jl = sqrt.(max.(mo.variances, 0) .* (n - 1))

        for k in 1:K
            # orth=FALSE
            @test abs(dot(m.loadings[:,k] ./ norm(m.loadings[:,k]),
                          v_r[:,k] ./ norm(v_r[:,k]))) > 0.999
            @test Set(findall(!iszero, m.loadings[:,k])) == Set(findall(!iszero, v_r[:,k]))
            # orth=TRUE
            @test abs(dot(mo.loadings[:,k] ./ norm(mo.loadings[:,k]),
                          vo_r[:,k] ./ norm(vo_r[:,k]))) > 0.999
            @test Set(findall(!iszero, mo.loadings[:,k])) == Set(findall(!iszero, vo_r[:,k]))
        end
        @test isapprox(d_jl,  d_r;  rtol=1e-5)
        @test isapprox(do_jl, do_r; rtol=1e-5)
    end
end