# This opens a module named BRMB. A module is a self-contained namespace — a named container 
#that holds definitions (functions, types, variables) and keeps them separate from everything else.



module BRMB     # This is the main module file for your package. It should be named BRMB.jl, and it should be located in the src/ folder of your package directory.


# This pulls in the two packages your PCA code depends on, making their functions available inside the module.
using LinearAlgebra, Statistics   # We need these packages for matrix operations and statistical functions. They are part of the Julia standard library, so we don't need to add them as dependencies in Project.toml.

# include your code using a path relative to THIS file.
# @__DIR__ is the folder BRMB.jl lives in (src/), so this always finds src/pca.jl.
include(joinpath(@__DIR__, "pca.jl")) 
include(joinpath(@__DIR__, "pmd.jl")) 

# export the names you want usable after `using .BRMB`
export pca, pcaStructure, pca_transform, pca_invtransform, pmd, pmd_orth

end










