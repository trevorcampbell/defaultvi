# Duchi et al
# Adaptive subgradient methods for online learning and stochastic optimization
# JMLR 2011

mutable struct AdaGrad
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	v::Vector{Float64}
	x::Vector{Float64}
end

AdaGrad(x::Vector{Float64}, stepsz, seed) = AdaGrad(Xoshiro(seed), 0, 0, 0, stepsz, zero(x), copy(x))
AdaGrad(dimension::Int64, stepsz, seed) = AdaGrad(zeros(dimension), stepsz, seed)

function step!(ag::AdaGrad, obj)
	g = gradient_estimate(obj, ag.x, 1, ag.rng)
	ag.ng += 1
	ag.iter += 1
	ag.v += g.^2
	ag.x -= ag.γ*(g./sqrt.(ag.v))
end
