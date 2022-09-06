module BayesQR

# Write your package code here.
using Distributions, LinearAlgebra, SpecialFunctions
using StatsModels
using MCMCChains, DataFrames

import Distributions: @check_args
import Base: rand

include("bqr.jl")

export bqr

end
