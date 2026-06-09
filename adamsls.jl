# Adaptive Gradient Methods Converge Faster with Over-Parametrization (but you should do a line-search)
# Vaswani et al
# Optimization for Machine Learning NeurIPS Workshop 2020

mutable struct AdamSLS
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	η::Float64
	c::Float64
	β1::Float64
	β2::Float64
	ϵ::Float64
	γ::Float64
	γls::Float64
	m::Vector{Float64}
	v::Vector{Float64}
	x::Vector{Float64}
end

AdamSLS(x::Vector{Float64}, stepsz, seed) = AdamSLS(Xoshiro(seed), 0, 0, 0, 0.5, 2.0, 0.9, 0.999, 1e-8, stepsz, 1e-6, zero(x), zero(x), copy(x))
AdamSLS(dimension::Int64, stepsz, seed) = AdamSLS(zeros(dimension), stepsz, seed)

function step!(asls::AdamSLS, obj)
	asls.iter += 1
	crng = copy(asls.rng)
	g = gradient_estimate(obj, asls.x, 1, asls.rng)
	asls.ng += 1

	# adam update of momentum/2nd moment
	asls.m = asls.β1*asls.m + (1-asls.β1)*g
	asls.v = asls.β2*asls.v + (1-asls.β2)*(g.^2)
	m̂ = asls.m/(1-asls.β1^asls.iter)
	v̂ = asls.v/(1-asls.β2^asls.iter)

	# sls step size in gradient direction
	f = function_estimate(obj, asls.x, 1, copy(crng))
	fp = function_estimate(obj, asls.x - asls.γls*g, 1, copy(crng))
	gs = sum(g.^2)
	asls.nf += 2
	while fp > f - asls.η*asls.γls*gs
		asls.γls /= asls.c
		fp = function_estimate(obj, asls.x - asls.γls*g, 1, copy(crng))
		asls.nf += 1
	end

	# step
	asls.x -= asls.γ*asls.γls * m̂./(sqrt.(v̂) .+ asls.ϵ)

	# step size reset
	asls.γls *= asls.c
end
