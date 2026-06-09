# Khaled et al
# DoWG Unleashed: An Efficient universal parameter-free gradient descent method
# NeurIPS 2023

mutable struct DoWG
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	dmax::Float64
	v::Float64
	x::Vector{Float64}
	x0::Vector{Float64}
end

DoWG(x::Vector{Float64}, stepsz, seed) = DoWG(Xoshiro(seed), 0, 0, 0, stepsz, 1e-6, 0.0, copy(x), copy(x))
DoWG(dimension::Int64, stepsz, seed) = DoWG(zeros(dimension), stepsz, seed)

function step!(dowg::DoWG, obj)
	g = gradient_estimate(obj, dowg.x, 1, dowg.rng)
	dowg.ng += 1
	dowg.iter += 1
	dowg.dmax = max(sqrt(sum((dowg.x-dowg.x0).^2)), dowg.dmax)
	dowg.v += dowg.dmax^2*sum(g.^2)
	η = dowg.γ*dowg.dmax^2/sqrt(dowg.v)
	dowg.x -= η*g
end
