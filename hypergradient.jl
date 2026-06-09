# Baydin et al
# Online learning rate adaptation with hypergradient descent
# ICLR 2018
# with the multiplicative step size update

mutable struct HyperGradient{Alg,Rate}
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	alg::Alg
	β::Float64
	x::Vector{Float64}
	xbuf::Vector{Float64}
	uprev::Vector{Float64}
end

HyperGradient{Alg,Rate}(x::Vector{Float64}, stepsz, seed) where {Alg,Rate} = HyperGradient{Alg,Rate}(Xoshiro(seed), 0, 0, 0, Alg(copy(x), stepsz, seed), 10.0 .^(-Rate), copy(x), zero(x), zero(x))
HyperGradient{Alg,Rate}(dimension::Int64, stepsz, seed) where {Alg,Rate} = HyperGradient{Alg,Rate}(zeros(dimension), stepsz, seed)

# used to trick alg into taking a step with a g that is known to hypergradient
struct HyperGradientFakeObjective
	g::Vector{Float64}
end

function gradient_estimate(obj::HyperGradientFakeObjective, x, n, rng)
	return obj.g
end

function step!(hgd::HyperGradient, obj)
	hgd.iter += 1
	g = gradient_estimate(obj, hgd.x, 1, hgd.rng)
	hgd.ng += 1

	# step size update
	# additive rule
	#hgd.alg.γ -= hgd.β*h
	# multiplicative rule (scale free)
	if sum(g.^2)*sum(hgd.uprev .^2) > 0
		hgd.alg.γ *= 1-hgd.β*sum(g .* hgd.uprev)/sqrt(sum(g.^2)*sum(hgd.uprev.^2))
	end

	# have the inner algorithm take a step using its new step size
	hgobj = HyperGradientFakeObjective(g)
	hgd.xbuf .= hgd.alg.x
	step!(hgd.alg, hgobj)

	# update uprev to store the step just taken
	hgd.uprev = (hgd.alg.x - hgd.xbuf)/hgd.alg.γ

	# copy the inner alg state to the meta-alg state
	hgd.x .= hgd.alg.x 
end
