struct MAPObjective{T}
	target::T
end

function initialize(obj::MAPObjective)
	return zeros(dimension(obj.target))
end

function function_estimate(obj::MAPObjective, x, n, rng)
	negative_logp = 0.0
	for i=1:n
		try
			negative_logp += -unbiased_logdensity(rng, obj.target, x)
		catch err
			negative_logp += Inf
		end
	end
	negative_logp /= n
	return negative_logp 
end

function gradient_estimate(obj::MAPObjective, x, n, rng)
	g = zero(x)
	for i=1:n
		_, gi = unbiased_logdensity_and_gradient(rng, obj.target, x)
		g -= gi
	end
	g /= n
	if any(isnan.(g)) || any(isinf.(g))
		throw("Gradient Nan/Inf")
	end
	return g
end



