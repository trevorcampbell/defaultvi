# Dog (Ivgi et al) + Momentum

mutable struct DoGMom
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

DoGMom(x::Vector{Float64}, stepsz, seed) = DoGMom(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 1e-6, 0.0, copy(x), copy(x), zero(x))
DoGMom(dimension::Int64, stepsz, seed) = DoGMom(zeros(dimension), stepsz, seed)

function step!(dog::DoGMom, obj)
	g = gradient_estimate(obj, dog.x, 1, dog.rng)
	dog.ng += 1
	dog.iter += 1
	dog.dmax = max(sqrt(sum((dog.x-dog.x0).^2)), dog.dmax)
	dog.v += sum(g.^2)
	dog.m = dog.β*dog.m + (1-dog.β)*g
	η = dog.γ*dog.dmax/sqrt(dog.v)
	dog.x -= η*dog.m
end
