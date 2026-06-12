











# test/CompareJiveRjive.jl — jive_rjive vs r.jive: results + speed
using BigRiverSchneider, RCall, BenchmarkTools
using LinearAlgebra, Statistics

# load real BRCA data (the actual target)
R"""
library(r.jive); data(BRCA_data)
X1 <- Data[[1]]; X2 <- Data[[2]]; X3 <- Data[[3]]
"""
@rget X1 X2 X3
rJ, rA = 2, [27, 26, 25]

# ---- fit yours ----
res = jive_rjive([X1,X2,X3], rJ, rA)

# ---- fit r.jive (same settings your function replicates) ----
@rput X1 X2 X3
R"""
fit <- jive(list(X1,X2,X3), rankJ=2, rankA=c(27,26,25),
            method="given", scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE,
            showProgress=FALSE)
"""
R"J1<-fit$joint[[1]]; J2<-fit$joint[[2]]; J3<-fit$joint[[3]]"
R"A1<-fit$individual[[1]]; A2<-fit$individual[[2]]; A3<-fit$individual[[3]]"
R"d1<-fit$data[[1]]; d2<-fit$data[[2]]; d3<-fit$data[[3]]"
@rget J1 J2 J3 A1 A2 A3 d1 d2 d3
Jr = [J1,J2,J3]; Ar = [A1,A2,A3]; Dr = [d1,d2,d3]

println("="^60)
println("RESULTS: jive_rjive vs r.jive")
println("="^60)

# 1. same input? (your scaled data should equal r.jive's fit$data)
nm = ["Expression","Methylation","miRNA"]
# rebuild your scaled data the way jive_rjive does, to compare against r.jive's
nel = [size(X,1)*size(X,2) for X in (X1,X2,X3)]; sum_n = sum(nel)
Xc = [ let Xi = X .- mean(X,dims=2); Xi ./ (norm(Xi)*sqrt(sum_n)); end for X in (X1,X2,X3) ]
println("\ninput match (your scaled data vs r.jive fit\$data):")
for i in 1:3
    println("  $(nm[i]): ‖diff‖ = ", round(norm(Xc[i] .- Dr[i]), digits=10))
end

# 2. joint & individual matrices
println("\nJ / A matrix differences:")
for i in 1:3
    println("  $(nm[i]): ‖J diff‖=", round(norm(res.J[i].-Jr[i]),digits=6),
            "  ‖A diff‖=", round(norm(res.A[i].-Ar[i]),digits=6))
end

# 3. variance explained (the interpretable, robust metric)
ve(J,A,D) = (norm(J)^2/norm(D)^2, norm(A)^2/norm(D)^2, norm(D.-J.-A)^2/norm(D)^2)
println("\nvariance explained (joint/indiv/resid):")
for i in 1:3
    println("  $(nm[i]):")
    println("    yours : ", round.(ve(res.J[i],res.A[i],Dr[i]), digits=4))
    println("    r.jive: ", round.(ve(Jr[i],Ar[i],Dr[i]), digits=4))
end

# 4. joint subspace agreement
jb(b...) = Matrix(qr(svd(vcat(b...)).Vt[1:rJ,:]').Q)[:,1:rJ]
println("\njoint subspace canonical correlations: ",
        round.(svd(jb(res.J...)' * jb(Jr...)).S, digits=6), "  (want ≈1)")

# ---- SPEED ----
println("\n", "="^60); println("TIMING"); println("="^60)
print("jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2,$X3], $rJ, $rA);
#  20.110 s (33806 allocations: 7.70 GiB)
R"""
library(microbenchmark)
mb <- microbenchmark(
  jive(list(X1,X2,X3), rankJ=2, rankA=c(27,26,25), method="given",
       scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=10)
cat("r.jive (R)        :  median", round(median(mb$time)/1e6,2), "ms\n")
"""
# r.jive (R)  :  median 77056.34 ms















# test/VignetteFigures.jl — faithful reproduction of all r.jive vignette figures
using BigRiverSchneider, RCall, Plots, Clustering, Distances, StatsPlots
using LinearAlgebra, Statistics

R"""
library(r.jive); data(BRCA_data)
X1<-Data[[1]]; X2<-Data[[2]]; X3<-Data[[3]]; cl<-clusts
"""
@rget X1 X2 X3 cl
nm = ["Expression","Methylation","miRNA"]
res = jive_rjive([X1,X2,X3], 2, [27,26,25])

# scaled data exactly as r.jive/jive_rjive computes it (for variance & heatmaps)
nel=[size(X,1)*size(X,2) for X in (X1,X2,X3)]; sum_n=sum(nel)
Dat = [ let Xi=X.-mean(X,dims=2); Xi./(norm(Xi)*sqrt(sum_n)); end for X in (X1,X2,X3) ]

# ============================================================
# FIGURE 1 — showVarExplained: ‖J‖²/‖data‖² stacked, grayscale
# ============================================================
VarJ = [norm(res.J[i])^2/norm(Dat[i])^2 for i in 1:3]
VarI = [norm(res.A[i])^2/norm(Dat[i])^2 for i in 1:3]
VarR = 1 .- VarJ .- VarI

fig1 = groupedbar(nm, [VarR VarI VarJ];              # reversed: residual, individual, joint
    bar_position=:stack,
    label=["Residual" "Individual" "Joint"],         # labels match column order
    color=[RGB(0.65,0.65,0.65) RGB(0.43,0.43,0.43) RGB(0.2,0.2,0.2)],  # light→dark, matching
    title="Variation Explained", legend=:outertopright,
    lw=0.3, linecolor=:black, bar_width=0.7, ylims=(0,1))
savefig(fig1, "vig_VarExplained.png"); println("Fig 1 done")
# ============================================================
# FIGURE 2 — showHeatmaps: Data = Joint + Individual + Noise per source.
# Matches r.jive: bluered (:bwr) colormap, mean±3sd clipping, hclust ordering.
# ============================================================
using Clustering, Distances

# column ordering: hclust on Euclidean distances of stacked-joint columns
Jstack = vcat(res.J...)
col_order = hclust(pairwise(Euclidean(), Jstack, dims=2), linkage=:complete).order
# per-source row orderings: hclust on each source's joint rows
row_orders = [hclust(pairwise(Euclidean(), res.J[i], dims=1), linkage=:complete).order for i in 1:3]

# layout widths: four heatmaps (0.22) + three symbol columns (0.04) = 1.0 exactly
W = [0.22, 0.04, 0.22, 0.04, 0.22, 0.04, 0.22]

# one clipped, saturated heatmap panel (r.jive's show.image: clip at mean±3sd, bluered)
function panel(M, ro; title="", ylabel="")
    Mo = M[ro, col_order]
    m = mean(Mo); s = std(Mo); lo, hi = m - 3s, m + 3s
    heatmap(clamp.(Mo, lo, hi);
        c = :bwr, clim = (lo, hi),
        title = title, ylabel = ylabel,
        cbar = false,  xticks = false, yticks = false,
        titlefontsize = 11, guidefontsize = 9)
end

# small text panel for =, +, and column headers
function txt(s; fs=20)
    p = plot(framestyle=:none, xlims=(0,1), ylims=(0,1), legend=false)
    annotate!(p, 0.5, 0.5, text(s, fs))
    return p
end

# one source's row: Data = Joint + Individual + Noise
function source_row(i)
    ro = row_orders[i]
    D = panel(Dat[i], ro; ylabel = nm[i])
    J = panel(res.J[i], ro)
    A = panel(res.A[i], ro)
    N = panel(Dat[i] .- res.J[i] .- res.A[i], ro)
    plot(D, txt("="), J, txt("+"), A, txt("+"), N;
        layout = grid(1, 7, widths = W), size = (1500, 300))
end

# header row of column titles
header = plot(
    txt("Data"; fs=16), txt(""), txt("Joint"; fs=16), txt(""),
    txt("Individual"; fs=16), txt(""), txt("Noise"; fs=16);
    layout = grid(1, 7, widths = W), size = (1500, 60))

# assemble: header + three source rows (heights sum to 1.0)
fig2 = plot(header, source_row(1), source_row(2), source_row(3);
    layout = grid(4, 1, heights = [0.08, 0.307, 0.307, 0.306]),
    size = (1500, 950))
savefig(fig2, "vig_Heatmaps.png")
println("Fig 2 → vig_Heatmaps.png")

# ============================================================
# FIGURE 3 — Joint PCA scatter (showPCA, n_joint=2), colored by cluster.
# r.jive's exact diag(d)*t(v) formula, true scores (matches current r.jive
# to 5 sig figs). NO rescaling — this is the actual ±1e-4 magnitude both produce.
# ============================================================
Fj  = svd(vcat(res.J...))
PCs = Diagonal(Fj.S[1:2]) * Fj.Vt[1:2, :]      # diag(d) * t(v)  — r.jive's showPCA formula

clusters = Int.(cl)
pal = [:black, :green, :purple]
colors = [pal[c] for c in clusters]

fig3 = scatter(PCs[2, :], PCs[1, :];            # x = Joint 2, y = Joint 1 (r.jive's order)
    color           = colors,
    markershape     = :circle,
    markersize      = 4,
    markerstrokewidth = 1.2,
    markerstrokecolor = colors,                 # open circles (colored ring, white fill)
    markercolor     = :white,
    xlabel          = "Joint 2  (×10⁻⁴)",
    ylabel          = "Joint 1  (×10⁻⁴)",
    title           = "Joint Structure, colored by cluster",
    legend          = false,
    size            = (550, 500),
    tickfontsize    = 9,
    guidefontsize   = 11,
    xformatter = x -> string(round(x * 1e4, digits = 1)),   # show value × 10⁴ (readable)
    yformatter = y -> string(round(y * 1e4, digits = 1)))
savefig(fig3, "vig_JointPCA.png")
println("Fig 3 → vig_JointPCA.png")


# FIGURE 4 — showPCA(n_joint=1, n_indiv=c(1,1,1)): scatterplot matrix.
# 4 components: Joint1, Expr-Indiv1, Meth-Indiv1, miRNA-Indiv1.
# Natural units (true diag(d)*t(v) scores), lower-triangular grid, open circles.
# ============================================================
Fj = svd(vcat(res.J...))
j1 = (Diagonal(Fj.S[1:1]) * Fj.Vt[1:1, :])[1, :]           # Joint 1
ind1 = [ (let F=svd(res.A[i]); (Diagonal(F.S[1:1])*F.Vt[1:1,:])[1,:]; end) for i in 1:3 ]

comps  = [j1, ind1[1], ind1[2], ind1[3]]                   # natural units, no scaling
names4 = ["Joint 1", "Expression Indiv 1", "Methylation Indiv 1", "miRNA Indiv 1"]

clusters = Int.(cl); pal = [:black, :green, :purple]
colors = [pal[c] for c in clusters]

# readable tick labels: show value × 10⁴ (the scores are ~1e-4)
fmt = v -> string(round(v * 1e4, digits = 1))

nPC = 4
panels = Matrix{Any}(undef, nPC-1, nPC-1)
for i in 2:nPC, j in 1:(nPC-1)
    r = i-1; c = j
    if j >= i
        panels[r,c] = plot(framestyle=:none)               # empty upper triangle
    else
        panels[r,c] = scatter(comps[i], comps[j];          # x = comp i, y = comp j (natural units)
            color=colors, markershape=:circle, markersize=3,
            markerstrokewidth=1.0, markerstrokecolor=colors, markercolor=:white,
            xlabel="$(names4[i])  (×10⁻⁴)", ylabel="$(names4[j])  (×10⁻⁴)",
            xformatter=fmt, yformatter=fmt,
            legend=false, tickfontsize=7, guidefontsize=8)
    end
end

fig4 = plot(permutedims(panels)...; layout=(nPC-1, nPC-1), size=(1100, 1100))
savefig(fig4, "vig_MorePCA.png")
println("Fig 4 → vig_MorePCA.png")