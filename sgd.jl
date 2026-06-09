# Standard SGD with a 1/sqrt t step size

mutable struct SGD
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	x::Vector{Float64}
end

SGD(x::Vector{Float64}, stepsz, seed) = SGD(Xoshiro(seed), 0, 0, 0, stepsz, copy(x))
SGD(dimension::Int64, stepsz, seed) = SGD(zeros(dimension), stepsz, seed)

function step!(sgd::SGD, obj)
	g = gradient_estimate(obj, sgd.x, 1, sgd.rng)
	sgd.ng += 1
	sgd.iter += 1
	sgd.x -= sgd.γ/sqrt(sgd.iter)*g
end


