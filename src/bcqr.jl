function sampleτ(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    C::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, θ::AbstractVector{<:Real}, a::Real, m::Real)
    k,n = size(C)
    sumterm = 0
    for i ∈ 1:n
        for j ∈ 1:k
            ξ₁, ξ₂ = (1-2*θ[j])/(θ[j]*(1-θ[j])), (2/(θ[j]*(1-θ[j])))
            sumterm += C[k, i] * (y[i] - b[k] - X[i,:] ⋅ β - ξ₁*v[i])^2 / (ξ₂ * v[i])
        end
    end
    rand(Gamma(3*n/2 + a,1/(0.5*sumterm + sum(v) + m)))
end

function samplev(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real},
    C::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, θ::AbstractVector{<:Real}, τ::Real)
    k,n = size(C)
    retV = zeros(n)
    for i ∈ 1:n
        ξ₁, ξ₂ = (1 .- 2 .* θ)./(θ.*(1 .-θ)), sqrt.(2 ./(θ .* (1 .-θ)))
        λ = C[:,i] ⋅ (ξ₁.^2 ./ ξ₂.^2)
        μ = (C[:,i] ⋅ ((ξ₁.^2 + 2*ξ₂.^2) ./ (y[i] .- b .- X[i,:]⋅β).^2))
        retV[i] = 1/rInvGauss(√μ, (λ + 2)*τ)
    end
    retV
end

function sampleβ(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, v::AbstractVector{<:Real},
    C::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, θ::AbstractVector{<:Real}, τ::Real, σβ::Real = 1.)
    n,p = size(X)
    k = length(b)
    retβ = zeros(p)
    ξ₁, ξ₂ = (1 .- 2 .* θ)./(θ.*(1 .-θ)), 2 ./(θ .* (1 .-θ))
    μ = zeros(p)
    Σ = zeros((p,p))
    for i in 1:n
        for j in 1:k
            μ += X[i,:] * C[j, i] * (y[i] - b[j] - ξ₁[j]*v[i]) * τ /(ξ₂[j]*v[i])
            Σ += τ/(ξ₂[j]*v[i]) * C[j,i] *(X[i,:] * X[i,:]')
        end
    end
    η = rand(Exponential(2/(1/σβ)^2), p)
    Σ = Symmetric((Σ + diagm(η))^(-1))
    rand(MvNormal(Σ*μ, Σ))
end

function sampleb(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real},  v::AbstractVector{<:Real},
    C::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, τ::Real)
    k,n = size(C)
    ξ₁, ξ₂ = (1 .- 2 .* θ)./(θ.*(1 .-θ)), 2 ./(θ .* (1 .-θ))
    retb = zeros(k)
    for i ∈ 1:k
        if sum(C[i,:]) > 0
            μ = sum((C[i,:] .* ((y - X*β) - (ξ₁[i].* v)) ) ./ (ξ₂[i] .* v))
            σ = sum(C[i,:] ./ (ξ₂[i] .* v))
            retb[i] = rand(Normal(μ/σ, 1/(τ*σ)))
        else
            retb[i] = rand(Normal(0, 1/τ))
        end
    end
    retb
end

function sampleW(C::AbstractMatrix{<:Real}, α::AbstractVector{<:Real})
    n = vec(sum(C, dims = 2))
    rand(Dirichlet(α + n))
end

function sampleC(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real},  v::AbstractVector{<:Real},
    b::AbstractVector{<:Real}, w::AbstractVector{<:Real}, θ::AbstractVector{<:Real}, τ::Real)
    n, k = length(y), length(θ)
    retC = zeros((k,n))
    ξ₁, ξ₂ = (1 .- 2 .* θ)./(θ.*(1 .-θ)), 2 ./(θ .* (1 .-θ))
    z = y - X*β
    for i ∈ 1:n
        probs = zeros(k)
        for j ∈ 1:k
            μ = z[i] - b[j]
            probs[j] = w[j] / √ξ₂[j] * exp(-(μ - ξ₁[j]*v[i])^2*τ / (2*ξ₂[j] * v[i]))
            #probs[j] = (z[i] - b[j] - ξ₁[j]*v[i])^2*τ / (2*ξ₂[j] * v[i])
        end
        #probs = (w ./ sqrt.(ξ₂)) .* exp.(.-(probs .+ mean(probs)))

        retC[:,i] = rand(Multinomial(1, probs./sum(probs)))
    end
    retC
end

function bcqr(f::FormulaTerm, df::DataFrame, τ::AbstractVector{<:Real}, niter::Int, burn::Int,
    probs::Bool = true, σβ::Real = 1.; kwargs...)
    allunique(τ) && all(τ .> 0) && all(τ .< 1) || throw(ArgumentError("all elements of τ must be on (0,1) and unique"))
    niter > burn || throw(ArgumentError("niter must be larger than burn"))

    mf = ModelFrame(f, df)
    y = response(mf)::Vector{Float64}
    X = modelmatrix(mf)::Matrix{Float64}
    n,p = size(X)
    k = length(τ)
    β, b = zeros((niter, p)), zeros((niter, k))
    C = zeros(Int64, k, n, niter); C[:,:,1] = rand(Multinomial(1, ones(k)/k), n);

    kwargs = Dict(kwargs)
    if haskey(kwargs, :β)
        β[1,:] = kwargs[:β]
    end
    if haskey(kwargs, :b)
        b[1,:] = kwargs[:b]
    end
    if haskey(kwargs, :σ)
        σ = kwargs[:σ] > 0 ? kwargs[:σ] : 1
    else
        σ = 1
    end
    v = ones(n)

    for i ∈ 2:niter
        σ = sampleτ(y, X, v, β[i-1,:], C[:,:,i-1], b[i-1,:], τ, 1, 1)
        v = samplev(y, X, β[i-1,:], C[:,:,i-1], b[i-1,:], τ, σ)
        β[i,:] = sampleβ(y, X, v, C[:,:,i-1], b[i-1,:], τ, σ)
        b[i,:] = sampleb(y, X, β[i,:], v, C[:,:,i-1], τ, σ)
        w = sampleW(C[:,:,i-1], [0.1 for i ∈ 1:length(τ)])
        C[:,:,i] = sampleC(y, X, β[i,:], v, b[i,:], w, τ, σ)
    end

    if probs
        (Chains(β[burn:niter,:], ["β"*string(i) for i in 1:p]), vec(mean(C[:,:,:], dims = [2,3])))
    else
        Chains(β[burn:niter,:], ["β"*string(i) for i in 1:p])
    end
end

function bcqr(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, τ::AbstractVector{<:Real}, niter::Int,
    burn::Int, probs::Bool = true, σβ::Real = 1.; kwargs...)
    allunique(τ) && all(τ .> 0) && all(τ .< 1) || throw(ArgumentError("all elements of τ must be on (0,1) and unique"))
    niter > burn || throw(ArgumentError("niter must be larger than burn"))
    n,p = size(X)
    n == length(y) || throw(DimensionMismatch("Mismatching dimensions of y and X"))
    k = length(τ)
    β, b = zeros((niter, p)), zeros((niter, k))
    C = zeros(Int64, k, n, niter); C[:,:,1] = rand(Multinomial(1, ones(k)/k), n);
    kwargs = Dict(kwargs)
    if haskey(kwargs, :β)
        β[1,:] = kwargs[:β]
    end
    if haskey(kwargs, :b)
        b[1,:] = kwargs[:b]
    end
    if haskey(kwargs, :σ)
        σ = kwargs[:σ] > 0 ? kwargs[:σ] : 1
    else
        σ = 1
    end
    v = ones(n)

    for i ∈ 2:niter
        σ = sampleτ(y, X, v, β[i-1,:], C[:,:,i-1], b[i-1,:], τ, 1, 1)
        v = samplev(y, X, β[i-1,:], C[:,:,i-1], b[i-1,:], τ, σ)
        β[i,:] = sampleβ(y, X, v, C[:,:,i-1], b[i-1,:], τ, σ)
        b[i,:] = sampleb(y, X, β[i,:], v, C[:,:,i-1], τ, σ)
        w = sampleW(C[:,:,i-1], [0.1 for i ∈ 1:length(τ)])
        C[:,:,i] = sampleC(y, X, β[i,:], v, b[i,:], w, τ, σ)
    end

    if probs
        (Chains(β[burn:niter,:], ["β"*string(i) for i in 1:p]), vec(mean(C[:,:,:], dims = [2,3])))
    else
        Chains(β[burn:niter,:], ["β"*string(i) for i in 1:p])
    end
end
