# Adam: A Method for Stochastic Optimization
# Kingma and Ba
# ICLR 2014

mutable struct Adam
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	β1::Float64
	β2::Float64
	ϵ::Float64
	m::Vector{Float64}
	v::Vector{Float64}
	x::Vector{Float64}
end

Adam(x::Vector{Float64}, stepsz, seed) = Adam(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 0.999, 1e-8, zero(x), zero(x), copy(x))
Adam(dimension::Int64, stepsz, seed) = Adam(zeros(dimension), stepsz, seed)

function step!(ad::Adam, obj)
	g = gradient_estimate(obj, ad.x, 1, ad.rng)
	ad.ng += 1
	ad.iter += 1
	ad.m = ad.β1*ad.m + (1-ad.β1)*g
	ad.v = ad.β2*ad.v + (1-ad.β2)*(g.^2)
	m̂ = ad.m/(1-ad.β1^ad.iter)
	v̂ = ad.v/(1-ad.β2^ad.iter)
	ad.x -= ad.γ * m̂./(sqrt.(v̂) .+ ad.ϵ)
end
