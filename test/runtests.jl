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

    τ =  [0.1, 0.25, 0.5, 0.75, 0.9]

    @test_throws ArgumentError bcqr(f1, df, τ, 100, 1)
    @test_throws ArgumentError bcqr(f, df, [0.1, 0.1], 100, 1)
    @test_throws ArgumentError bcqr(f, df, [-0.1, 0.25, 0.5], 100, 1)
    @test_throws ArgumentError bcqr(y, X, [0.1, 0.1], 100, 1)
    @test_throws ArgumentError bcqr(y, X, [-0.1, 0.25, 0.5], 100, 1)
    @test typeof(bcqr(f, df, τ, 10, 1)) <: Tuple{<:Chains, <:AbstractVector}
    @test typeof(bcqr(f, df, τ, 10, 1, false)) <: Chains
    @test_throws DimensionMismatch bcqr(y, X[1:(n-2),:], τ, 100, 1)
end
