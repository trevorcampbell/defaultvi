# A minor modification of Adam to use average of squared gradient norms in denominator (not exponential average)
# As described in
# Welandawe et al 
# A Framework for Improving the Reliability of Black-box Variational Inference
# JMLR 25(219):1−71, 2024.
# https://arxiv.org/abs/2203.15945

mutable struct AdamAvg
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	β::Float64
	ϵ::Float64
	m::Vector{Float64}
	v::Vector{Float64}
	x::Vector{Float64}
end

AdamAvg(x::Vector{Float64}, stepsz, seed) = AdamAvg(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 1e-8, zero(x), zero(x), copy(x))
AdamAvg(dimension::Int64, stepsz, seed) = AdamAvg(zeros(dimension), stepsz, seed)

function step!(ad::AdamAvg, obj)
	g = gradient_estimate(obj, ad.x, 1, ad.rng)
	ad.ng += 1
	ad.iter += 1
	ad.m = ad.β*ad.m + (1-ad.β)*g
	ad.v = (1.0 - 1/ad.iter)*ad.v + (1/ad.iter)*(g.^2)
	m̂ = ad.m/(1-ad.β^ad.iter)
	v̂ = ad.v # no bias correction on denominator
	ad.x -= ad.γ * m̂./(sqrt.(v̂) .+ ad.ϵ)
end
