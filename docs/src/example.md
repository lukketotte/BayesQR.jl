# Boston housing data
The data is obtained from [RDatasets.jl](https://juliapackages.com/p/rdatasets)

```@example 1
using BayesQR, Plots, DataFrames, MCMCChains, RDatasets

dat = dataset("MASS", "Boston")
y = log.(dat[:, :MedV])
X = dat[:, Not(["MedV"])] |> Matrix
nothing
```

Generate 3000 samples from the posterior and discard the first 1000 as burn-in for quantile levels 0.1 and 0.9.
```@example 1
b_01 = bqr(y, X, 0.1, 3000, 1000)
b_09 = bqr(y, X, 0.9, 3000, 1000)

plot(b_01[:β4], label = "τ = 0.1")
plot!(b_09[:β4], label = "τ = 0.9")
savefig("f-plot.svg"); nothing # hide
```
![](f-plot.svg)
