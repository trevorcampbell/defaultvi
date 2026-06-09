# On the convergence of AMSGrad and beyond
# Reddi et al
# ICLR 2018

mutable struct AMSGrad
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
	v̂::Vector{Float64}
	x::Vector{Float64}
end

AMSGrad(x::Vector{Float64}, stepsz, seed) = AMSGrad(Xoshiro(seed), 0, 0, 0, stepsz, 0.9, 0.999, 1e-8, zero(x), zero(x), zero(x), copy(x))
AMSGrad(dimension::Int64, stepsz, seed) = AMSGrad(zeros(dimension), stepsz, seed)

function step!(ams::AMSGrad, obj)
	g = gradient_estimate(obj, ams.x, 1, ams.rng)
	ams.ng += 1
	ams.iter += 1
	ams.m = ams.β1*ams.m + (1-ams.β1)*g
	ams.v = ams.β2*ams.v + (1-ams.β2)*(g.^2)
	ams.v̂ = max.(ams.v̂, ams.v)
	ams.x -= ams.γ * ams.m ./(sqrt.(ams.v̂) .+ ams.ϵ)
end
