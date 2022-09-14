# BayesQR.jl
Bayesian quantile regression (BQR) models in Julia.

## Installation
```julia
Pkg.add("BayesQR")
```

## Function Documentation

```@docs
bqr
```

## Fitting BayesQR models
Two methods can be used to fit a BQR:
`bqr(formula, data, τ, niter, burn)` and `bqr(y, X, τ, niter, burn)`.
Their arguments must be:
- `formula`: a [StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/) referring to columns in `data`.
- `data`: a table in the Tables.jl definition, e.g. a data frame; NAs are dropped
- `X` a matrix holding values of the independent variable(s) in columns
- `y` a vector holding values of the dependent variable

Both method returns a [MCMCChains.jl `Chains` object](https://beta.turing.ml/MCMCChains.jl/dev/chains/)
