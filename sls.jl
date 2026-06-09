# Vaswani et al
# Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates
# NeurIPS 2019
# With resetting

mutable struct SLS
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	η::Float64
	c::Float64
	γ::Float64
	γls::Float64
	x::Vector{Float64}
end

SLS(x::Vector{Float64}, stepsz, seed) = SLS(Xoshiro(seed), 0, 0, 0, 0.5, 2.0, stepsz, 1e-6, copy(x))
SLS(dimension::Int64, stepsz, seed) = SLS(zeros(dimension), stepsz, seed)

function step!(sls::SLS, obj)
	# compute the optimal step incrementally
	rng = copy(sls.rng)
	sls.iter += 1
	f = function_estimate(obj, sls.x, 1, sls.rng) # step the rng forward
	g = gradient_estimate(obj, sls.x, 1, copy(rng))
	gs = sum(g.^2)
	f2 = function_estimate(obj, sls.x-sls.γls*g, 1, copy(rng))
	sls.ng += 1
	sls.nf += 2
	while f2 > f - sls.η*sls.γls*gs
		sls.γls /= sls.c
		f2 = function_estimate(obj, sls.x-sls.γls*g, 1, copy(rng))
		sls.nf += 1
	end
	sls.x = sls.x - sls.γ*sls.γls*g
	sls.γls *= sls.c # reset
end


