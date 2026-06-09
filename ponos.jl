# Galli et al
# Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parametrized Models
# NeurIPS 2023
# With Zhang Xi = 1
# No Polyak initial step sizes (since we have no estimate of f*), but using the nonmonotone bound for armijo

mutable struct PoNoS
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	η::Float64
	c::Float64
	C::Float64
	γ::Float64
	γls::Float64
	x::Vector{Float64}
end

PoNoS(x::Vector{Float64}, stepsz, seed) = PoNoS(Xoshiro(seed), 0, 0, 0, 0.5, 2.0, 0.0, stepsz, 1e-6, copy(x))
PoNoS(dimension::Int64, stepsz, seed) = PoNoS(zeros(dimension), stepsz, seed)

function step!(ps::PoNoS, obj)
	rng = copy(ps.rng)
	ps.iter += 1
	f = function_estimate(obj, ps.x, 1, ps.rng) # step the rng forward
	g = gradient_estimate(obj, ps.x, 1, copy(rng))
	ps.nf += 1
	ps.ng += 1
	if ps.iter == 1
		ps.C = f
	end
	C̃ = ((ps.iter-1)*ps.C + f)/(ps.iter + 1)
	ps.C = max(C̃, f)
	gs = sum(g.^2)
	f2 = function_estimate(obj, ps.x-ps.γls*g, 1, copy(rng))
	ps.nf += 1
	while f2 > ps.C - ps.η*ps.γls*gs
		ps.γls /= ps.c
		f2 = function_estimate(obj, ps.x-ps.γls*g, 1, copy(rng))
		ps.nf += 1
	end
	ps.γls *= ps.c
	ps.x = ps.x - ps.γ*ps.γls*g
end


