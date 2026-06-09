# SAA with nesterov acceleration

mutable struct SAANA
	seed::Int64
	nf::Int64
	ng::Int64
	iter::Int64
	λ::Float64
	c::Float64
	η::Float64
	n::Int64
	γ::Float64 
	x::Vector{Float64}
	y::Vector{Float64}
	f::Float64
	g::Vector{Float64}
	gs::Float64
end

SAANA(x::Vector{Float64}, stepsz, seed) = SAANA(seed, 0, 0, 0, 1.0, 2.0, 0.5, 1, stepsz, copy(x), copy(x), NaN, copy(x), NaN)
SAANA(dimension::Int64, stepsz, seed) = SAANA(zeros(dimension), stepsz, seed)

function step!(agd::SAANA, obj)
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

	yp = agd.x - agd.γ*agd.g
	fp = function_estimate(obj, yp, agd.n, copy(rng))
	agd.nf += agd.n
	while fp > agd.f - agd.η*agd.γ*agd.gs
		agd.γ /= agd.c
		yp = agd.x - agd.γ*agd.g
		fp = function_estimate(obj, yp, agd.n, copy(rng))
		agd.nf += agd.n
	end

	# apply accelerated update, invalidate the f cache
	λp = (1+sqrt(1+4agd.λ^2))/2
	agd.x = yp + (agd.λ-1)/λp*(yp - agd.y)
	agd.y = yp
	agd.λ = λp
	agd.f = NaN
	agd.γ *= agd.c
end


