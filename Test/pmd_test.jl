# Test/pmd_test.jl — formal tests for the two-sided PMD(L1,L1).


@testset "output structure & invariants" begin
    Random.seed!(1)
    n, p, K = 80, 60, 3
    X = randn(n, p)
    m = pmd(X; sumabs=0.4, K=K)

    @test m isa pmdStructure
    @test size(m.u) == (n, K)
    @test size(m.v) == (p, K)
    @test length(m.d) == K
    @test m.K == K

    # both factors unit-norm (or zero) per component
    for k in 1:K
        nu = norm(m.u[:, k]); nv = norm(m.v[:, k])
        @test isapprox(nu, 1.0; atol=1e-6) || nu == 0.0
        @test isapprox(nv, 1.0; atol=1e-6) || nv == 0.0
    end
    # d values are real and finite
    @test all(isfinite, m.d)
    # budgets were resolved into the recorded range
    @test 1 <= m.sumabsu <= sqrt(n) + 1e-9
    @test 1 <= m.sumabsv <= sqrt(p) + 1e-9
    # grand mean recorded when centering
    @test isapprox(m.meanx, mean(X); atol=1e-10)
end

@testset "max budget reduces to rank-1 SVD (theorem anchor)" begin
    # At sumabsu=√n, sumabsv=√p neither penalty binds → no soft-thresholding →
    # plain power iteration → the rank-1 SVD of the centered data.
    Random.seed!(7)
    n, p = 60, 40
    X = randn(n, p)
    Xc = X .- mean(X)
    m = pmd(X; sumabsu=sqrt(n), sumabsv=sqrt(p), K=1)
    F = svd(Xc)
    @test abs(dot(m.u[:, 1], F.U[:, 1])) > 0.999     # u → U₁
    @test abs(dot(m.v[:, 1], F.V[:, 1])) > 0.999     # v → V₁
    # and both factors are fully dense at max budget
    @test count(!iszero, m.u[:, 1]) == n
    @test count(!iszero, m.v[:, 1]) == p
end

@testset "two-sided sparsity contract" begin
    Random.seed!(11)
    n, p = 80, 60
    X = randn(n, p)
    # smaller sumabs → sparser on BOTH factors; max budget → dense
    dense  = pmd(X; sumabsu=sqrt(n), sumabsv=sqrt(p), K=1)
    mid    = pmd(X; sumabs=0.5, K=1)
    tight  = pmd(X; sumabs=0.25, K=1)
    @test count(!iszero, dense.u[:,1]) == n
    @test count(!iszero, dense.v[:,1]) == p
    # tightening drives nonzeros down (monotone) on both sides
    @test count(!iszero, tight.u[:,1]) <= count(!iszero, mid.u[:,1]) <= n
    @test count(!iszero, tight.v[:,1]) <= count(!iszero, mid.v[:,1]) <= p
    @test count(!iszero, tight.v[:,1]) < p           # tight actually zeros some
end

@testset "ground-truth: recovers planted sparse factors" begin
    # plant two well-separated sparse rank-1 factors (sparse u AND sparse v).
    # strong signal + SVD init ⇒ unique optimum ⇒ recovery is clean.
    Random.seed!(42)
    n, p = 80, 60
    u1 = [randn(20); zeros(60)];            v1 = [randn(15); zeros(45)]
    u2 = [zeros(40); randn(20); zeros(20)]; v2 = [zeros(30); randn(15); zeros(15)]
    X = 6.0*(u1*v1') .+ 4.0*(u2*v2') .+ 0.3 .* randn(n, p)
    m = pmd(X; sumabsu=0.45*sqrt(n), sumabsv=0.45*sqrt(p), K=2, center=true)

    for (k, (ut, vt)) in enumerate(((u1,v1),(u2,v2)))
        @test abs(dot(m.u[:,k] ./ norm(m.u[:,k]), ut ./ norm(ut))) > 0.9
        @test abs(dot(m.v[:,k] ./ norm(m.v[:,k]), vt ./ norm(vt))) > 0.9
        # recovered support overlaps the true support
        selu = Set(findall(!iszero, m.u[:,k])); trueu = Set(findall(!iszero, ut))
        selv = Set(findall(!iszero, m.v[:,k])); truev = Set(findall(!iszero, vt))
        @test length(intersect(selu, trueu)) / length(trueu) > 0.8
        @test length(intersect(selv, truev)) / length(truev) > 0.8
    end
end

@testset "multiple components (deflation)" begin
    Random.seed!(5)
    n, p, K = 60, 50, 4
    X = randn(n, p)
    m = pmd(X; sumabs=0.4, K=K)
    @test size(m.u) == (n, K)
    @test size(m.v) == (p, K)
    @test length(m.d) == K
    # every component sparse at this budget
    for k in 1:K
        @test count(!iszero, m.v[:,k]) < p
    end
end

@testset "sumabsu/sumabsv override sumabs" begin
    Random.seed!(15)
    n, p = 70, 50
    X = randn(n, p)
    # explicit budgets are recorded directly (ignoring sumabs)
    su = 0.3*sqrt(n); sv = 0.6*sqrt(p)
    m = pmd(X; sumabs=0.9, sumabsu=su, sumabsv=sv, K=1)
    @test isapprox(m.sumabsu, su; atol=1e-10)
    @test isapprox(m.sumabsv, sv; atol=1e-10)
end

@testset "center=false skips centering" begin
    Random.seed!(9)
    n, p = 50, 40
    X = randn(n, p) .+ 5.0                # nonzero mean
    m = pmd(X; sumabs=0.5, K=1, center=false)
    @test isnan(m.meanx)                  # no mean recorded
end

@testset "determinism (SVD init, no random dependence)" begin
    # despite the randn loop-primer, the SVD init pins the result → repeat calls
    # converge to the same factors (up to the primer not changing the fixed point).
    Random.seed!(99)
    n, p = 60, 40
    X = randn(n, p)
    a = pmd(X; sumabs=0.4, K=2)
    b = pmd(X; sumabs=0.4, K=2)
    for k in 1:2
        @test abs(dot(a.v[:,k] ./ norm(a.v[:,k]), b.v[:,k] ./ norm(b.v[:,k]))) > 0.999
        @test isapprox(a.d[k], b.d[k]; rtol=1e-6)
    end
end

@testset "argument validation" begin
    Random.seed!(0)
    n, p = 100, 16
    X = randn(n, p)
    # K out of range
    @test_throws ArgumentError pmd(X; K=0)
    @test_throws ArgumentError pmd(X; K=min(n,p)+1)
    # sumabs out of (0,1]
    @test_throws ArgumentError pmd(X; sumabs=0.0)
    @test_throws ArgumentError pmd(X; sumabs=1.5)
    # explicit budgets out of range
    @test_throws ArgumentError pmd(X; sumabsu=0.5, sumabsv=2.0)        # sumabsu < 1
    @test_throws ArgumentError pmd(X; sumabsu=2.0, sumabsv=sqrt(p)+1)  # sumabsv > √p
end


@testset "matches R PMA::PMD (offline reference fixtures)" begin
    # Reads precomputed R outputs from Test/Data/PMD/ (generated by pmd.R).
    refdir = joinpath(@__DIR__, "Data", "PMD")
    if !isfile(joinpath(refdir, "X.csv"))
        @info "PMD R-reference fixtures not found; skipping. Run pmd.R to create them."
    else
        # provenance: report which PMA version these were generated against
        smfile = joinpath(refdir, "session_meta.csv")
        if isfile(smfile)
            sm = readdlm(smfile, ',', String; skipstart=1)
            row = findfirst(==("PMA_version"), sm[:, 1])
            row !== nothing && @info "PMD fixtures generated against PMA $(sm[row, 2])"
        end

        X   = readdlm(joinpath(refdir, "X.csv"),     ',', Float64; skipstart=1)
        u_r = readdlm(joinpath(refdir, "u_pmd.csv"), ',', Float64; skipstart=1)
        v_r = readdlm(joinpath(refdir, "v_pmd.csv"), ',', Float64; skipstart=1)
        d_r = vec(readdlm(joinpath(refdir, "d_pmd.csv"), ',', Float64; skipstart=1))
        meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart=1)
        K  = Int(meta[3]); su = meta[4]; sv = meta[5]

        # X.csv is already grand-mean-centered by R → run pmd with center=false
        m = pmd(X; sumabsu=su, sumabsv=sv, K=K, center=false)

        for k in 1:K
            # both factors align up to sign (bit-identical → ≈ 1.0)
            @test abs(dot(m.u[:,k] ./ norm(m.u[:,k]), u_r[:,k] ./ norm(u_r[:,k]))) > 0.999
            @test abs(dot(m.v[:,k] ./ norm(m.v[:,k]), v_r[:,k] ./ norm(v_r[:,k]))) > 0.999
            # selected supports match exactly (same features zeroed on both sides)
            @test Set(findall(!iszero, m.u[:,k])) == Set(findall(!iszero, u_r[:,k]))
            @test Set(findall(!iszero, m.v[:,k])) == Set(findall(!iszero, v_r[:,k]))
        end
        # singular values match to machine precision
        @test isapprox(m.d, d_r; rtol=1e-6)
    end
end