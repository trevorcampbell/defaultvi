# Orabona and Tommasi
# Training Deep Networks without Learning Rates Through COCOB Betting
# COCOB-Backprop (Algorithm 2)
# (because we don't know bounds on gradients)

mutable struct COCOB
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	α::Float64
	L::Vector{Float64}
	G::Vector{Float64}
	R::Vector{Float64}
	θ::Vector{Float64}
	x0::Vector{Float64}
	x::Vector{Float64}
end

COCOB(x::Vector{Float64}, stepsz, seed) = COCOB(Xoshiro(seed), 0, 0, 0, 100.0, zero(x), zero(x), zero(x), zero(x), copy(x), copy(x))
COCOB(dimension::Int64, stepsz, seed) = COCOB(zeros(dimension), stepsz, seed)

function step!(ccb::COCOB, obj)
	ccb.iter += 1
	g = gradient_estimate(obj, ccb.x, 1, ccb.rng) # note, paper Alg 2 uses g = -grad, here g = grad
	ccb.ng += 1
	ccb.L = max.(ccb.L, abs.(g))
	ccb.G += abs.(g)
	ccb.R = max.(0, ccb.R - (ccb.x - ccb.x0) .* g)
	ccb.θ -= g
	ccb.x = ccb.x0 + ccb.θ ./ (ccb.L .* max.(ccb.G + ccb.L, ccb.α*ccb.L)) .* (ccb.L + ccb.R)
end
