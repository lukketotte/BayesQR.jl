using BayesQR
using Test
using Distributions, DataFrames, StatsModels, LinearAlgebra, MCMCChains

@testset "BayesQR.jl" begin
    n,p = 100, 3;
    X = rand(MultivariateNormal(zeros(p), I), n) |> x -> reshape(x, n, p)
    y = 2.1 .+  X * ones(p) .+ rand(Normal(), n)

    df = DataFrame(hcat(y, X), :auto)
    f1 = @formula(x1 ~ 1 + a + x3 + x4)
    f = @formula(x1 ~ 1 + x2 + x3 + x4)

    @test_throws ArgumentError bqr(f1, df, 0.5, 100, 1)
    @test_throws DimensionMismatch bqr(y, X[1:(n-2),:], 0.5, 100, 1)
    @test_throws DomainError bqr(y, X, -0.1, 100, 1)
    @test_throws DomainError bqr(f, df, -0.1, 100, 1)
    @test_throws ArgumentError bqr(f, df, 0.1, 1, 10)
    @test_throws ArgumentError bqr(y, X, 0.1, 1, 10)
    @test typeof(bqr(f, df, 0.5, 10, 1)) <: Chains

    @test_throws DomainError bqr(f, df, 0.1, 100, 1, -1, "laplace")
    @test_throws ArgumentError bqr(y, X, 0.1, 1, 10, 10, "beta")
end
