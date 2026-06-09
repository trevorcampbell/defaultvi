mutable struct SAA
	seed::Int64
	nf::Int64
	ng::Int64
	iter::Int64
	c::Float64
	η::Float64
	n::Int64
	γ::Float64 
	x::Vector{Float64}
	f::Float64
	g::Vector{Float64}
	gs::Float64
end

SAA(x::Vector{Float64}, stepsz, seed) = SAA(seed, 0, 0, 0, 2.0, 0.5, 1, stepsz, copy(x), NaN, copy(x), NaN)
SAA(dimension::Int64, stepsz, seed) = SAA(zeros(dimension), stepsz, seed)

function step!(agd::SAA, obj)
	rng = Xoshiro(agd.seed)
	agd.iter += 1
	
	# compute the gradient estimate for 2n and increase n until reasonably agree with n
	crng = copy(rng)
	agd.g = gradient_estimate(obj, agd.x, agd.n, crng)
	g2 = (agd.g + gradient_estimate(obj, agd.x, agd.n, crng))/2
	agd.ng += 2*agd.n
	while sqrt(sum(agd.g.^2)) < 0.5sqrt(sum(g2.^2)) || sum(agd.g .* g2) < 0
		agd.n *= 2
		agd.g = g2
		g2 = (g2 + gradient_estimate(obj, agd.x, agd.n, crng))/2
		agd.ng += agd.n
		agd.f = NaN # n increased, invalidate f cache. this could be made more efficient (below we recompute the whole f again, could just add terms)
	end
	agd.gs = sum(agd.g.^2)

	# if there is no cached f
	# compute it, otherwise use the cache
	if isnan(agd.f)
		agd.f = function_estimate(obj, agd.x, agd.n, copy(rng))
		agd.nf += agd.n
	end

	xp = agd.x - agd.γ*agd.g
	fp = function_estimate(obj, xp, agd.n, copy(rng))
	agd.nf += agd.n
	while fp > agd.f - agd.η*agd.γ*agd.gs
		agd.γ /= agd.c
		xp = agd.x - agd.γ*agd.g
		fp = function_estimate(obj, xp, agd.n, copy(rng))
		agd.nf += agd.n
	end
	agd.f = fp
	agd.x = xp
	agd.γ *= agd.c
end


