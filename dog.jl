# Ivgi et al
# DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule
# ICML 2023

mutable struct DoG
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	dmax::Float64
	v::Float64
	x::Vector{Float64}
	x0::Vector{Float64}
end

# DoG and DoWG treat the initial max a little differently, at least per the papers
# DoG: uses epsilon in the first iteration, then removes it entirely
# DoWG: sets r_{-1} = epsilon, and then runs the usual iteration
# For simplicity I'll keep these the same with the DoWG style init
DoG(x::Vector{Float64}, stepsz, seed) = DoG(Xoshiro(seed), 0, 0, 0, stepsz, 1e-6, 0.0, copy(x), copy(x))
DoG(dimension::Int64, stepsz, seed) = DoG(zeros(dimension), stepsz, seed)

function step!(dog::DoG, obj)
	g = gradient_estimate(obj, dog.x, 1, dog.rng)
	dog.ng += 1
	dog.iter += 1
	dog.dmax = max(sqrt(sum((dog.x-dog.x0).^2)), dog.dmax)
	dog.v += sum(g.^2)
	η = dog.γ*dog.dmax/sqrt(dog.v)
	dog.x -= η*g
end
