# Cutkosky et al
# Mechanic: A learning rate tuner
# NeurIPS 2023
# Mechanic is appropriate to apply to 
# any optimizer that queries a single gradient per call

mutable struct Mechanic{Alg}
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	alg::Alg
	x::Vector{Float64}
	x0::Vector{Float64}
	xbuf::Vector{Float64}
	Δ::Vector{Float64}
	s::Vector{Float64}
	sinit::Float64
	β::Vector{Float64}
	λ::Float64
	ϵ::Float64
	v::Vector{Float64}
	r::Vector{Float64}
	m::Vector{Float64}
end

Mechanic{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg} = Mechanic{Alg}(Xoshiro(seed), 0, 0, 0, Alg(copy(x), stepsz, seed), copy(x), copy(x), copy(x), zero(x), 
											zeros(6), 1e-8, [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], 0.01, 1e-8, zeros(6), zeros(6), zeros(6))
Mechanic{Alg}(dimension::Int64, stepsz, seed) where {Alg} = Mechanic{Alg}(zeros(dimension), stepsz, seed)

# used to trick alg into taking a step with a g that is known to mechanic
struct MechanicFakeObjective
	g::Vector{Float64}
end

function gradient_estimate(obj::MechanicFakeObjective, x, n, rng)
	return obj.g
end

function step!(mch::Mechanic, obj)
	mch.iter += 1
	# mechanic queries the gradient at the meta-alg's state xt
	g = gradient_estimate(obj, mch.x, 1, mch.rng)
	mch.ng += 1

	# send g to the base algorithm and take a step
	mobj = MechanicFakeObjective(g)
	mch.xbuf .= mch.alg.x
	step!(mch.alg, mobj)
	# receive the update
	u = mch.alg.x - mch.xbuf

	h = sum( mch.Δ .* (g + mch.λ*sum(mch.s)*sqrt(sum(g.^2))*mch.x/sqrt(sum(mch.x.^2))) )
	mch.Δ += u
	mch.m = max.(mch.β .* mch.m, h)
	mch.v = mch.β.^2 .* mch.v .+ h^2
	mch.r = mch.β .* mch.r - mch.s*h
	W = mch.sinit*mch.m/length(mch.m) + mch.r
	mch.s = W ./ (sqrt.(mch.v) .+ mch.ϵ)
	mch.x = mch.x0 + sum(mch.s)*mch.Δ
end
