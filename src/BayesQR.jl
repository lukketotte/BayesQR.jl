module BayesQR

# Write your package code here.
using Distributions, LinearAlgebra, SpecialFunctions, ForwardDiff
using StatsModels
using MCMCChains, DataFrames

import Distributions: @check_args
import Base: rand

include("bqr.jl")
include("bcqr.jl")

export bqr, bcqr

end
