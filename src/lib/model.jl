import Statistics

using Functors
using LinearAlgebra
using LogExpFunctions

function load_model()
    ann = Lux.Chain(
        Normalize(Float32[100, 300]),
        Lux.BranchLayer(
            Lux.Chain(
                Lux.SelectDim(1, 1),
                Lux.ReshapeLayer((1,)),
                Lux.Dense(1, 12, Lux.swish), # was smooth_relu
                Lux.Parallel(vcat, 
                    Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32),
                    Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
                )
            ),
            Lux.Chain(
                Lux.SelectDim(1, 2),
                Lux.ReshapeLayer((1,)),
                Lux.Dense(1, 12, Lux.swish),
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
            )
        ),
        Combine(1 => [1, 2], 2 => [1]),
        AddGlobalParameters(4, [3, 4]; activation=Lux.softplus) # Fixed effects on Q and V2
    )

    return DCM(two_comp!, ann, CombinedError(; init = [1., 0.1]))
end


function setup(rng::Random.AbstractRNG, dcm::DeepCompartmentModel{T}, population::Population; omega = 0.1) where T
    ps_theta, st_theta = Lux.setup(rng, dcm.model)
    
    Ω_init = Symmetric(collect(Diagonal(T.(omega) .* ones(T, 2))))

    ps = (
        theta = ps_theta,
        error = dcm.error.init_f(rng, dcm.error, MeanSqrt()),
        omega = Ω_init,
        phi = (
            μ = [zeros(T, 2) for _ in eachindex(population)],
            L = [cholesky(Ω_init).L for _ in eachindex(population)]
        )
    )

    st = (
        theta = st_theta,
        phi = (
            epsilon = [randn(rng, T, 2) for _ in eachindex(population)],
            mask = DeepCompartmentModels.indicator(4, [1,2], T)
        )
    )

    return fmap(Base.Fix1(_convert, T), ps), fmap(Base.Fix1(_convert, T), st)
end

_convert(::Type{T}, x::Real) where T = T(x)
_convert(::Type{T}, x::AbstractArray) where T = T.(x)
_convert(::Type{T}, x::Cholesky) where T = Cholesky(T.(x.factors), x.uplo, x.info)
_convert(::Type{T}, x::Symmetric) where T = Symmetric(T.(x))

Statistics.std(::CombinedError, ŷ::AbstractVector{T}, ps) where T<:Real = 
    sqrt.(softplus(ps.error.σ[1]).^2 .+ ŷ.^2 .* softplus.(ps.error.σ[2])^2)

DeepCompartmentModels.make_dist(error::CombinedError, ŷ::AbstractVector{T}, ps) where T<:Real = 
    MvNormal(ŷ, std(error, ŷ, ps))

function DeepCompartmentModels.predict(dcm::DeepCompartmentModel, population::Population, ps, st)
    ζ, _ = dcm.model(get_x(population), ps.theta, st.theta)
    η = ps.phi.μ + ps.phi.L .* st.phi.epsilon
    z = ζ + exp.(st.phi.mask * reduce(hcat, η))
    p = construct_p(z, population)
    return forward_ode_with_dv(dcm, population, p)
end

function loglikelihood(dcm::DeepCompartmentModel, population::Population, ps, st)
    ŷ = predict(dcm, population, ps, st)
    dists = make_dist(dcm, ŷ, ps)
    return logpdf.(dists, get_y(population))
end

function logprior(::DeepCompartmentModel{T}, ps, st) where T
    η = ps.phi.μ + ps.phi.L .* st.phi.epsilon
    prior = MvNormal(zeros(T, size(ps.omega, 1)), ps.omega)
    return logpdf(prior, η)
end

function logq(::DeepCompartmentModel{T}, ps, st) where T
    η = ps.phi.μ + ps.phi.L .* st.phi.epsilon
    qΣ = Symmetric.(ps.phi.L .* transpose.(ps.phi.L))
    qs = MvNormal.(ps.phi.μ, qΣ)
    return logpdf.(qs, η)
end

elbo(dcm::DeepCompartmentModel, population::Population, ps, st) =
    loglikelihood(dcm, population, ps, st) + logprior(dcm, ps, st) - logq(dcm, ps, st)

objective(dcm, population, ps, st) = -sum(elbo(dcm, population, ps, st))

function gradient(dcm, population, ps, st)
    grad = Zygote.gradient(ps) do p
        objective(dcm, population, p, st)
    end
    return first(grad)
end

update_epsilon!(rng::Random.AbstractRNG, st) = fmap_with_path(st; cache = nothing) do kp, x
    if :epsilon in kp
        x .= randn(rng, eltype(x), size(x))
    else
        return x
    end
end