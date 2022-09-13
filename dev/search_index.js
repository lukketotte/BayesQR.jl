var documenterSearchIndex = {"docs":
[{"location":"#BayesQR.jl","page":"Home","title":"BayesQR.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Bayesian quantile regression (BQR) models in Julia.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pkg.add(\"BayesQR\")","category":"page"},{"location":"#Function-Documentation","page":"Home","title":"Function Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"bqr","category":"page"},{"location":"#BayesQR.bqr","page":"Home","title":"BayesQR.bqr","text":"bqr(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, τ::Real, niter::Int, burn::Int)\n\nRuns the Bayesian quantile regression with dependent variable y and covariates X for quantile τ. Priors currently implemented are the Normal and Laplace.\n\nArguments\n\nσᵦ::Real: variance of π(β)\nprior::String : \"Normal\" or \"Laplace\"\n\n\n\n\n\nbqr(f::FormulaTerm, df::DataFrame, τ::Real, niter::Int, burn::Int)\n\nRuns the Bayesian quantile regression with dependent variable y and covariates X constructed from f and df.\n\n\n\n\n\n","category":"function"},{"location":"#Fitting-BayesQR-models","page":"Home","title":"Fitting BayesQR models","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two methods can be used to fit a BQR: bqr(formula, data, τ, niter, burn) and bqr(y, X, τ, niter, burn). Their arguments must be: -formula: a StatsModels.jl Formula object referring to columns in data.","category":"page"},{"location":"","page":"Home","title":"Home","text":"data: a table in the Tables.jl definition, e.g. a data frame; NAs are dropped\nX a matrix holding values of the independent variable(s) in columns\ny a vector holding values of the dependent variable","category":"page"},{"location":"","page":"Home","title":"Home","text":"Both method returns a MCMCChains.jl Chains object","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"}]
}
