# test/VignetteFiguresOpt.jl — BRCA vignette figures using jive_rjive_opt
using BigRiverSchneider, RCall, Plots, StatsPlots, Clustering, Distances
using LinearAlgebra, Statistics

# load real BRCA data + cluster labels
R"""
library(r.jive); data(BRCA_data)
X1<-Data[[1]]; X2<-Data[[2]]; X3<-Data[[3]]; cl<-clusts
"""
@rget X1 X2 X3 cl
nm = ["Expression","Methylation","miRNA"]

# ---- fit with the OPTIMIZED function ----
println("Fitting jive_rjive_opt with permutation rank selection")
res = jive_rjive_opt([X1,X2,X3])             # ranks estimated (optimized path)
rJ  = res.r
println("Estimated ranks: joint=$rJ, individual=$(res.ri)\n")

# scaled data exactly as r.jive/jive_rjive computes it
nel = [size(X,1)*size(X,2) for X in (X1,X2,X3)]; sum_n = sum(nel)
Dat = [ let Xi = X .- mean(X,dims=2); Xi ./ (norm(Xi)*sqrt(sum_n)); end for X in (X1,X2,X3) ]

clusters = Int.(cl)
pal = [:black, :green, :purple]
colors = [pal[c] for c in clusters]

# FIGURE 1 — showVarExplained
VarJ = [norm(res.J[i])^2/norm(Dat[i])^2 for i in 1:3]
VarI = [norm(res.A[i])^2/norm(Dat[i])^2 for i in 1:3]
VarR = 1 .- VarJ .- VarI

fig1 = groupedbar(nm, [VarR VarI VarJ];
    bar_position = :stack,
    label = ["Residual" "Individual" "Joint"],
    color = [RGB(0.65,0.65,0.65) RGB(0.43,0.43,0.43) RGB(0.2,0.2,0.2)],
    title = "Variation Explained", legend = :outertopright,
    lw = 0.3, linecolor = :black, bar_width = 0.7, ylims = (0,1))
savefig(fig1, "vigopt_VarExplained.png"); println("Fig 1 → vigopt_VarExplained.png")

# FIGURE 2 — showHeatmaps
Jstack = vcat(res.J...)
col_order = hclust(pairwise(Euclidean(), Jstack, dims=2), linkage=:complete).order
row_orders = [hclust(pairwise(Euclidean(), res.J[i], dims=1), linkage=:complete).order for i in 1:3]

W = [0.22, 0.04, 0.22, 0.04, 0.22, 0.04, 0.22]

function panel(M, ro; title="", ylabel="")
    Mo = M[ro, col_order]
    m = mean(Mo); s = std(Mo); lo, hi = m - 3s, m + 3s
    heatmap(clamp.(Mo, lo, hi);
        c = :bwr, clim = (lo, hi), title = title, ylabel = ylabel,
        cbar = false, xticks = false, yticks = false,
        titlefontsize = 11, guidefontsize = 9)
end

function txt(s; fs=20)
    p = plot(framestyle=:none, xlims=(0,1), ylims=(0,1), legend=false)
    annotate!(p, 0.5, 0.5, text(s, fs))
    return p
end

function source_row(i)
    ro = row_orders[i]
    D = panel(Dat[i], ro; ylabel = nm[i])
    J = panel(res.J[i], ro)
    A = panel(res.A[i], ro)
    N = panel(Dat[i] .- res.J[i] .- res.A[i], ro)
    plot(D, txt("="), J, txt("+"), A, txt("+"), N;
        layout = grid(1, 7, widths = W), size = (1500, 300))
end

header = plot(
    txt("Data"; fs=16), txt(""), txt("Joint"; fs=16), txt(""),
    txt("Individual"; fs=16), txt(""), txt("Noise"; fs=16);
    layout = grid(1, 7, widths = W), size = (1500, 60))

fig2 = plot(header, source_row(1), source_row(2), source_row(3);
    layout = grid(4, 1, heights = [0.08, 0.307, 0.307, 0.306]),
    size = (1500, 950))
savefig(fig2, "vigopt_Heatmaps.png"); println("Fig 2 → vigopt_Heatmaps.png")

# FIGURE 3 — Joint PCA scatter
Fj = svd(vcat(res.J...))
fmt = v -> string(round(v * 1e4, digits = 1))

if rJ >= 2
    PCs = Diagonal(Fj.S[1:rJ]) * Fj.Vt[1:rJ, :]
    fig3 = scatter(PCs[2, :], PCs[1, :];
        color = colors, markershape = :circle, markersize = 4,
        markerstrokewidth = 1.2, markerstrokecolor = colors, markercolor = :white,
        xlabel = "Joint 2  (×10⁻⁴)", ylabel = "Joint 1  (×10⁻⁴)",
        title = "Joint Structure, colored by cluster",
        legend = false, size = (550, 500), tickfontsize = 9, guidefontsize = 11,
        xformatter = fmt, yformatter = fmt)
    savefig(fig3, "vigopt_JointPCA.png"); println("Fig 3 → vigopt_JointPCA.png")
else
    PCs = Diagonal(Fj.S[1:1]) * Fj.Vt[1:1, :]
    fig3 = scatter(1:size(PCs,2), PCs[1, :];
        color = colors, markershape = :circle, markersize = 4,
        markerstrokewidth = 1.2, markerstrokecolor = colors, markercolor = :white,
        xlabel = "Sample", ylabel = "Joint 1  (×10⁻⁴)",
        title = "Joint Structure (1 component), colored by cluster",
        legend = false, size = (550, 500), yformatter = fmt)
    savefig(fig3, "vigopt_JointPCA.png"); println("Fig 3 → vigopt_JointPCA.png (1 joint comp)")
end

# FIGURE 4 — showPCA scatterplot matrix
j1   = (Diagonal(Fj.S[1:1]) * Fj.Vt[1:1, :])[1, :]
ind1 = [ (let F = svd(res.A[i]); (Diagonal(F.S[1:1])*F.Vt[1:1,:])[1,:]; end) for i in 1:length(res.A) ]

comps  = [j1, ind1...]
names4 = ["Joint 1"; ["$(nm[i]) Indiv 1" for i in 1:length(ind1)]]

nPC = length(comps)
panels = Matrix{Any}(undef, nPC-1, nPC-1)
for i in 2:nPC, j in 1:(nPC-1)
    r = i-1; c = j
    if j >= i
        panels[r,c] = plot(framestyle=:none)
    else
        panels[r,c] = scatter(comps[i], comps[j];
            color = colors, markershape = :circle, markersize = 3,
            markerstrokewidth = 1.0, markerstrokecolor = colors, markercolor = :white,
            xlabel = "$(names4[i])  (×10⁻⁴)", ylabel = "$(names4[j])  (×10⁻⁴)",
            xformatter = fmt, yformatter = fmt,
            legend = false, tickfontsize = 7, guidefontsize = 8)
    end
end
fig4 = plot(permutedims(panels)...; layout = (nPC-1, nPC-1), size = (1100, 1100))
savefig(fig4, "vigopt_MorePCA.png"); println("Fig 4 → vigopt_MorePCA.png")

# ============================================================
# BENCHMARK — jive_rjive_opt vs jive_rjive vs r.jive on real BRCA
# ============================================================
using BenchmarkTools
println("\n", "="^60); println("TIMING — permutation rank selection on BRCA"); println("="^60)

print("jive_rjive_opt (Julia): ")
@btime jive_rjive_opt([$X1,$X2,$X3]) samples=3 evals=1;

print("jive_rjive     (Julia): ")
@btime jive_rjive([$X1,$X2,$X3]) samples=3 evals=1;

R"""
library(microbenchmark)
mbP <- microbenchmark(
  jive(list(X1,X2,X3), method="perm", est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=3)
cat("r.jive (R)            :  median", round(median(mbP$time)/1e6, 2), "ms\n")
"""