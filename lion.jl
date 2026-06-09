# Symbolic Discovery of Optimization Algorithms
# Chen et al
# NeurIPS 2023

mutable struct Lion
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	β1::Float64
	β2::Float64
	m::Vector{Float64}
	x::Vector{Float64}
end

Lion(x::Vector{Float64}, stepsz, seed) = Lion(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 0.999, zero(x), copy(x))
Lion(dimension::Int64, stepsz, seed) = Lion(zeros(dimension), stepsz, seed)

function step!(li::Lion, obj)
	li.iter += 1
	g = gradient_estimate(obj, li.x, 1, li.rng)
	li.ng += 1
	li.x -= li.γ*sign.(li.β1*li.m + (1-li.β1)*g)
	li.m = li.β2*li.m +(1-li.β2)*g
end
