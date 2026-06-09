mutable struct SAACG
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
	p::Vector{Float64}
	pg::Float64
	gprev::Vector{Float64}
end

SAACG(x::Vector{Float64}, stepsz, seed) = SAACG(seed, 0, 0, 0, 2.0, 0.5, 1, stepsz, copy(x), NaN, zero(x), 0.0, ones(length(x)))
SAACG(dimension::Int64, stepsz, seed) = SAACG(zeros(dimension), stepsz, seed)

function step!(agd::SAACG, obj)
	rng = Xoshiro(agd.seed)
	agd.iter += 1

	# compute the conjugate gradient estimate for 2n and increase n until reasonably agree with n
	crng = copy(rng)
	g = gradient_estimate(obj, agd.x, agd.n, crng)
	β = max(0, sum(g.*(g - agd.gprev))/sum(agd.gprev.^2)) # Polak-Ribiere with resetting
	p = g + β*agd.p
	g2 = (g + gradient_estimate(obj, agd.x, agd.n, crng))/2
	β2 = max(0, sum(g2.*(g2 - agd.gprev))/sum(agd.gprev.^2))
	p2 = g2 + β2*agd.p
	agd.ng += 2*agd.n
	while sqrt(sum(p.^2)) < 0.5sqrt(sum(p2.^2)) || sum(p .* p2) < 0
		agd.n *= 2
		g = g2
		p = p2
		β = β2
		g2 = (g2 + gradient_estimate(obj, agd.x, agd.n, crng))/2
		β2 = max(0, sum(g2.*(g2 - agd.gprev))/sum(agd.gprev.^2))
		p2 = g2 + β2*agd.p
		agd.ng += agd.n
		agd.f = NaN # n increased, invalidate f cache. this could be made more efficient (below we recompute the whole f again, could just add terms)
	end
	agd.p .= p
	agd.pg = sum(p .* g)
	agd.gprev .= g

	# agd.pg should typically be > 0 (descent direction). If not, reset to just the gradient.
	if agd.pg <= 0
		agd.p .= g
		agd.pg = sum(g.^2)
	end

	# if there is no cached f
	# compute it, otherwise use the cache
	if isnan(agd.f)
		agd.f = function_estimate(obj, agd.x, agd.n, copy(rng))
		agd.nf += agd.n
	end

	xp = agd.x - agd.γ*agd.p
	fp = function_estimate(obj, xp, agd.n, copy(rng))
	agd.ng += agd.n
	agd.nf += agd.n
	while fp > agd.f - agd.η*agd.γ*agd.pg
		agd.γ /= agd.c
		xp = agd.x - agd.γ*agd.p
		fp = function_estimate(obj, xp, agd.n, copy(rng))
		agd.nf += agd.n
		agd.ng += agd.n
	end

	# update state
	agd.f = fp
	agd.x = xp
	agd.γ *= agd.c
end


