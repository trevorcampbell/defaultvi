# Big Batch SGD: Automated Inference using Adaptive Batch Sizes
# De et al 
# AISTATS 2017

mutable struct BigBatch
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	c::Int64
	η::Float64
	n0::Float64
	γ::Float64
	x::Vector{Float64}
end

BigBatch(x::Vector{Float64}, stepsz, seed) = BigBatch(Xoshiro(seed), 0, 0, 0, 2.0, 0.5, 2, stepsz, copy(x))
BigBatch(dimension::Int64, stepsz, seed) = BigBatch(zeros(dimension), stepsz, seed)

function step!(bb::BigBatch, obj)
	bb.iter += 1
	rng = copy(bb.rng)
	m = gradient_estimate(obj, bb.x, 1, bb.rng)
	v = zero(m)
	n = 1
	while true
		n += 1
		ĝ = gradient_estimate(obj, bb.x, 1, bb.rng)
		d = ĝ - m
		m += d/n
		v += d .* (ĝ-m)
		# aggregate at least n0 draws
		if n <= bb.n0
			continue
		end
		# compute the required number of draws
		nreq = (sum(v/(n-1)) / sum(m.^2))
		# if current > required, break out
		if n > nreq
			bb.n0 = n
			break
		end
		#end
	end
	bb.ng += n


	# n stores the number of rng calls
	# compute the function estimate on that same batch
	f = function_estimate(obj, bb.x, n, copy(rng))
	fp = function_estimate(obj, bb.x - bb.γ * m, n, copy(rng))
	bb.nf += 2n 

	ms = sum(m.^2)
	while fp > f - bb.η*bb.γ*ms
		bb.γ /= bb.c
		fp = function_estimate(obj, bb.x - bb.γ * m, n, copy(rng))
		bb.nf += n
	end
	bb.x = bb.x - bb.γ*m
	bb.γ *= bb.c
end


