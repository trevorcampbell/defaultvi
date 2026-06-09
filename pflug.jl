# On the determination of the step size in stochastic quasigradient methods
# George Pflug
# 1983
# Algorithm 4.1

mutable struct Pflug
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	n0::Int64
	c::Float64
	g::Vector{Float64}
	δ::Float64
	ξ::Float64
	γ::Float64
	x::Vector{Float64}
end

Pflug(x::Vector{Float64}, stepsz, seed) = Pflug(Xoshiro(seed), 0, 0, 0, 0, 2.0, zero(x), 0.0, 0.0, stepsz, copy(x))
Pflug(dimension::Int64, stepsz, seed) = Pflug(zeros(dimension), stepsz, seed)

function step!(pf::Pflug, obj)
	pf.iter += 1
	ĝ1 = gradient_estimate(obj, pf.x, 1, pf.rng)
	ĝ2 = gradient_estimate(obj, pf.x, 1, pf.rng)
	pf.ng += 2
	ĝ = 0.5(ĝ1+ĝ2)
	d̂ = 0.5(ĝ1-ĝ2)

	x̃ = pf.x - pf.γ*d̂
	pf.x = pf.x - pf.γ*ĝ
	g̃ = gradient_estimate(obj, x̃, 1, pf.rng)
	pf.ng += 1

	pf.δ += sum(d̂ .* g̃)
	pf.ξ += sum(ĝ .* pf.g)
	pf.g = ĝ

	if pf.iter % 2 == 0
		if pf.ξ/(pf.iter-pf.n0) <= 3/2 * pf.δ/(pf.iter-pf.n0+1)
			pf.γ /= pf.c
			pf.n0 = pf.iter
			pf.g = zero(pf.x)
			pf.δ = 0.0
			pf.ξ = 0.0
		end
	end
end
