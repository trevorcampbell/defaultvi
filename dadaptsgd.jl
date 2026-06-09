# Learning-rate-free learning by d-adaptation
# Defazio and Mishchenko
# 2023
# Algorithm 4

mutable struct DAdaptSGD
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	d::Float64
	β::Float64
	G::Float64
	s::Vector{Float64}
	z::Vector{Float64}
	λgs::Float64
	γ::Float64
	x::Vector{Float64}
end

DAdaptSGD(x::Vector{Float64}, stepsz, seed) = DAdaptSGD(Xoshiro(seed), 0, 0, 0, 1e-6, 0.9, 0.0, zero(x), copy(x), 0.0, stepsz, copy(x))
DAdaptSGD(dimension::Int64, stepsz, seed) = DAdaptSGD(zeros(dimension), stepsz, seed)

function step!(sgd::DAdaptSGD, obj)
	g = gradient_estimate(obj, sgd.x, 1, sgd.rng)
	sgd.ng += 1
	sgd.iter += 1
	if sgd.iter == 1
		sgd.G = sqrt(sum(g.^2))
	end
	λ = sgd.d*sgd.γ/sgd.G
	sgd.λgs += λ*sum(g.*sgd.s)
	sgd.s = sgd.s + λ*g
	sgd.z = sgd.z - λ*g
	sgd.x = sgd.β*sgd.x + (1-sgd.β)*sgd.z
	sgd.d = max(sgd.d, 2*sgd.λgs/sqrt(sum(sgd.s.^2)))
end


