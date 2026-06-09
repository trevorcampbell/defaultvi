# DoWG (Khaled et al) + Momentum

mutable struct DoWGMom
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	β::Float64
	dmax::Float64
	v::Float64
	x::Vector{Float64}
	x0::Vector{Float64}
	m::Vector{Float64}
end

DoWGMom(x::Vector{Float64}, stepsz, seed) = DoWGMom(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 1e-6, 0.0, copy(x), copy(x), zero(x))
DoWGMom(dimension::Int64, stepsz, seed) = DoWGMom(zeros(dimension), stepsz, seed)

function step!(dowg::DoWGMom, obj)
	g = gradient_estimate(obj, dowg.x, 1, dowg.rng)
	dowg.ng += 1
	dowg.iter += 1
	dowg.dmax = max(sqrt(sum((dowg.x-dowg.x0).^2)), dowg.dmax)
	dowg.v += dowg.dmax^2*sum(g.^2)
	dowg.m = dowg.β*dowg.m + (1-dowg.β)*g
	η = dowg.γ*dowg.dmax^2/sqrt(dowg.v)
	dowg.x -= η*dowg.m
end
