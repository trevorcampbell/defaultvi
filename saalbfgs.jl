mutable struct SAALBFGS
	seed::Int64
	nf::Int64
	ng::Int64
	iter::Int64
	m::Int64
	c::Float64
	η::Float64
	n::Int64
	γ::Float64 
	x::Vector{Float64}
	f::Float64
	p::Vector{Float64}
	pg::Float64
	ρ::Vector{Float64}
	s::Vector{Vector{Float64}}
	y::Vector{Vector{Float64}}
end

SAALBFGS(x::Vector{Float64}, stepsz, seed) = SAALBFGS(seed, 0, 0, 0, 10, 2.0, 0.5, 1, stepsz, copy(x), NaN, copy(x), NaN, Vector{Float64}(), Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
SAALBFGS(dimension::Int64, stepsz, seed) = SAALBFGS(zeros(dimension), stepsz, seed)

function preconditioned_gradient(g, ρ, s, y)
	m = length(ρ)
	if m == 0
		return g
	end
	q = copy(g)
	α = zeros(m)
	for i=1:m
		α[m-i+1] = ρ[m-i+1]*sum(s[m-i+1] .* q)
		q = q - α[m-i+1]*y[m-i+1]
	end
	z = sum(s[1].*y[1])/sum(y[1].*y[1]) * q
	for i=1:m
		β = ρ[i]*sum(y[i] .* z)
		z = z + s[i]*(α[i] - β)
	end
	return z
end

function step!(agd::SAALBFGS, obj)
	rng = Xoshiro(agd.seed)
	agd.iter += 1

	# compute the gradient estimate for 2n and increase n until reasonably agree with n
	crng = copy(rng)
	g = gradient_estimate(obj, agd.x, agd.n, crng)
	agd.p = preconditioned_gradient(g, agd.ρ, agd.s, agd.y)
	g2 = (g + gradient_estimate(obj, agd.x, agd.n, crng))/2
	p2 = preconditioned_gradient(g2, agd.ρ, agd.s, agd.y)
	agd.ng += 2*agd.n
	while sqrt(sum(agd.p.^2)) < 0.5sqrt(sum(p2.^2)) || sum(agd.p .* p2) < 0
		agd.n *= 2
		g .= g2
		agd.p .= p2
		g2 = (g2 + gradient_estimate(obj, agd.x, agd.n, crng))/2
		p2 = preconditioned_gradient(g2, agd.ρ, agd.s, agd.y)
		agd.ng += agd.n
		agd.f = NaN # n increased, invalidate f cache. this could be made more efficient (below we recompute the whole f again, could just add terms)
	end
	agd.pg = sum(agd.p .* g)

	# agd.pg should always be > 0. If it is ever negative, it's likely due to numerical issues; reset the memory
	if agd.pg <= 0
		agd.ρ = Vector{Float64}()
		agd.s = Vector{Vector{Float64}}()
		agd.y = Vector{Vector{Float64}}()
		agd.p .= g
		agd.pg = sum(agd.p .* g)
	end

	# if there is no cached f
	# compute it, otherwise use the cache
	if isnan(agd.f)
		agd.f = function_estimate(obj, agd.x, agd.n, copy(rng))
		agd.nf += agd.n
	end

	# line search
	xp = agd.x - agd.γ*agd.p
	fp = function_estimate(obj, xp, agd.n, copy(rng))
	agd.nf += agd.n
	while fp > agd.f - agd.η*agd.γ*agd.pg
		agd.γ /= agd.c
		xp = agd.x - agd.γ*agd.p
		fp = function_estimate(obj, xp, agd.n, copy(rng))
		agd.nf += agd.n
	end

	# update lbfgs memory
	gp = gradient_estimate(obj, xp, agd.n, copy(rng))
	agd.ng += agd.n
	s = -agd.γ*agd.p
	y = gp - g
	if sum(y.*s) > 0
		push!(agd.s, -agd.γ*agd.p)
		push!(agd.y, gp-g)
		push!(agd.ρ, 1.0/sum(y.*s))
	end

	# keep memory limited
	if length(agd.ρ) > agd.m
		popfirst!(agd.s)
		popfirst!(agd.y)
		popfirst!(agd.ρ)
	end

	# update state
	agd.f = fp
	agd.x = xp
	agd.γ *= agd.c
end


