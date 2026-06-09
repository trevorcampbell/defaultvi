# Adaptive Gradient Methods Converge Faster with Over-Parametrization (but you should do a line-search)
# Vaswani et al
# Optimization for Machine Learning NeurIPS Workshop 2020

mutable struct AdaSLS
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	η::Float64
	c::Float64
	γ::Float64
	γls::Float64
	v::Vector{Float64}
	x::Vector{Float64}
end

AdaSLS(x::Vector{Float64}, stepsz, seed) = AdaSLS(Xoshiro(seed), 0, 0, 0, 0.5, 2.0, stepsz, 1e-6, zero(x), copy(x))
AdaSLS(dimension::Int64, stepsz, seed) = AdaSLS(zeros(dimension), stepsz, seed)

function step!(asls::AdaSLS, obj)
	asls.iter += 1
	crng = copy(asls.rng)
	g = gradient_estimate(obj, asls.x, 1, asls.rng)
	asls.ng += 1
	asls.v += g.^2
	gs = sum(g.^2)
	# update step size
	f = function_estimate(obj, asls.x, 1, copy(crng))
	fp = function_estimate(obj, asls.x - asls.γls*g, 1, copy(crng))
	asls.nf += 2
	while fp > f - asls.η*asls.γls*gs
		asls.γls /= asls.c
		fp = function_estimate(obj, asls.x - asls.γls*g, 1, copy(crng))
		asls.nf += 1
	end
	asls.x = asls.x - asls.γ*asls.γls*g./sqrt.(asls.v)
	asls.γls *= asls.c
end
