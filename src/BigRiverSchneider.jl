module BigRiverSchneider




using LinearAlgebra, Statistics   

include("pca.jl")
export pca, pcaStructure, pca_transform, pca_invtransform

include("pmd.jl") 
export  pmd, pmd_orth

include("pls.jl") 
export  pls, Plsr, plscoef, plspredict, plstransform




end 
