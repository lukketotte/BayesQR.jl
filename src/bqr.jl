function rInvGauss(μ::Real, λ::Real)
    χ = rand(Chisq(1))
    f = μ/(2*λ) * (2*λ + μ * χ - √(4*λ*μ*χ + μ^2*χ^2))
    rand(Uniform()) < μ/(μ + f) ? f : μ^2/f
end

function sampleβ(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, σ::Real, θ::Real, ω::Real)
    Σ = ((broadcast(/, X, sqrt.(v.*σ.*ω)) |> x -> x'x) + I / 10)^(-1)
    μ = Σ*(X'*((y .- θ.*v) ./ (ω*σ.*v)))
    rand(MvNormal(μ, Symmetric(Σ)))
end

function sampleV(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real}, θ::Real, ω::Real, σ::Real)
    v = similar(y)
    λ = 2/σ + θ^2/(σ*ω)
    μ = sqrt.(λ ./ ((y - X*β).^2 ./ (ω*σ)))
    for i ∈ 1:length(v)
        v[i] = 1/rInvGauss(μ[i], λ)
    end
    v
end

function sampleσ(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real},
    v::AbstractVector{<:Real}, θ::Real, ω::Real)
    shape = (2 + 3*length(y)) /2
    scale = (2 + 2*sum(v) + sum((y - X*β - θ.*v).^2 ./ (ω.*v)))/2
    rand(InverseGamma(shape, scale))
end

function bqr(f::FormulaTerm, df::DataFrame, τ::Real, niter::Int, burn::Int; kwargs...)
    τ > 0 && τ < 1 || throw(DomainError(τ,"τ must be on (0,1)"))
    niter > burn || throw(ArgumentError("niter must be larger than burn"))
    mf = ModelFrame(f, df)
    y = response(mf)::Vector{Float64}
    X = modelmatrix(mf)::Matrix{Float64}
    n,p = size(X)
    θ, ω = (1-2*τ)/(τ*(1-τ)), 2/(τ*(1-τ))
    β = zeros((niter, p))
    kwargs = Dict(kwargs)
    if haskey(kwargs, :β)
        β[1,:] = kwargs[:β]
    end
    if haskey(kwargs, :σ)
        σ = kwargs[:σ] > 0 ? kwargs[:σ] : 1
    else
        σ = 1
    end
    v = ones(n)

    for i ∈ 2:niter
        v = sampleV(y, X, β[i-1,:], θ, ω, σ)
        β[i,:] = sampleβ(y, X, v, σ, θ, ω)
        σ = sampleσ(y, X, β[i,:], v, θ, ω)
    end

    Chains(β[burn:end,:], ["β"*string(i) for i in 1:p])
end

function bqr(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, τ::Real, niter::Int, burn::Int; kwargs...)
    τ > 0 && τ < 1 || throw(DomainError(τ,"τ must be on (0,1)"))
    niter > burn || throw(ArgumentError("niter must be larger than burn"))
    n,p = size(X)
    n == length(y) || throw(DimensionMismatch("Mismatching dimensions of y and X"))
    θ, ω = (1-2*τ)/(τ*(1-τ)), 2/(τ*(1-τ))
    β = zeros((niter, p))
    kwargs = Dict(kwargs)
    if haskey(kwargs, :β)
        β[1,:] = kwargs[:β]
    end
    if haskey(kwargs, :σ)
        σ = kwargs[:σ] > 0 ? kwargs[:σ] : 1
    else
        σ = 1
    end
    v = ones(n)

    for i ∈ 2:niter
        v = sampleV(y, X, β[i-1,:], θ, ω, σ)
        β[i,:] = sampleβ(y, X, v, σ, θ, ω)
        σ = sampleσ(y, X, β[i,:], v, θ, ω)
    end

    Chains(β[burn:end,:], ["β"*string(i) for i in 1:p])
end
