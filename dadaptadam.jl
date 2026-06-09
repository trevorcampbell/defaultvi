# Learning-rate-free learning by d-adaptation
# Defazio and Mishchenko
# 2023
# Algorithm 5

mutable struct DAdaptAdam
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	d::Float64
	β1::Float64
	β2::Float64
	ϵ::Float64
	s::Vector{Float64}
	m::Vector{Float64}
	v::Vector{Float64}
	r::Float64
	γ::Float64
	x::Vector{Float64}
end

DAdaptAdam(x::Vector{Float64}, stepsz, seed) = DAdaptAdam(Xoshiro(seed), 0, 0, 0, 1e-6, 0.9, 0.999, 1e-8, zero(x), zero(x), zero(x), 0.0, stepsz, copy(x))
DAdaptAdam(dimension::Int64, stepsz, seed) = DAdaptAdam(zeros(dimension), stepsz, seed)

function step!(ad::DAdaptAdam, obj)
	g = gradient_estimate(obj, ad.x, 1, ad.rng)
	ad.ng += 1
	ad.iter += 1
	ad.m = ad.β1*ad.m + (1-ad.β1)*ad.d*ad.γ*g
	ad.v = ad.β2*ad.v + (1-ad.β2)*(g.^2)
	A = sqrt.(ad.v) .+ ad.ϵ
	ad.x -= ad.m ./ A
	ad.r = sqrt(ad.β2)*ad.r + (1-sqrt(ad.β2))*ad.d*ad.γ*sum(g.*ad.s./A)
	ad.s = sqrt(ad.β2)*ad.s + (1-sqrt(ad.β2))*ad.d*ad.γ*g
	ad.d = max(ad.d, ad.r/((1-sqrt(ad.β2))*sum(abs.(ad.s))))
end
