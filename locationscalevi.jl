abstract type GeneralLocationScaleVariationalObjective end

struct DiagGaussianVariationalObjective{T} <: GeneralLocationScaleVariationalObjective
	q0::Distribution
	target::T # must satisfy the unbiased_logdensityproblems interface
end

DiagGaussianVariationalObjective(target) = DiagGaussianVariationalObjective(MvNormal(zeros(dimension(target)), I), target)

function initialize(obj::DiagGaussianVariationalObjective)
	d = dimension(obj.target)
	return vcat(zeros(d), ones(d)/10.0)
end

function decode_and_logjacobian(obj::DiagGaussianVariationalObjective, λ, z, ϵ)
	d = length(ϵ)
	return λ[1:d] + λ[d+1:end] .* ϵ, sum(log.(abs.(λ[d+1:end])))
end

function encoder_logprob(obj::DiagGaussianVariationalObjective, λ, x, z)
	return 0.0
end

struct FullGaussianVariationalObjective{T} <: GeneralLocationScaleVariationalObjective
	q0::Distribution 
	target::T # must satisfy the unbiased_logdensityproblems interface
end

FullGaussianVariationalObjective(target) = FullGaussianVariationalObjective(MvNormal(zeros(dimension(target)), I), target)

function initialize(obj::FullGaussianVariationalObjective)
	d = dimension(obj.target)
	return vcat(zeros(d), lower_to_vec(Matrix(I,d,d))/10.0)
end

function decode_and_logjacobian(obj::FullGaussianVariationalObjective, λ, z, ϵ)
	d = length(ϵ)
	return λ[1:d] + vec_to_lower(λ[d+1:end])*ϵ, sum(log.(abs.(vec_to_diag(λ[d+1:end]))))
end

function encoder_logprob(obj::FullGaussianVariationalObjective, λ, x, z)
	return 0.0
end


struct VAEVariationalObjective{T} <: GeneralLocationScaleVariationalObjective
	q0::Distribution 
	target::T # must satisfy the unbiased_logdensityproblems interface
	internal_feature_dimension::Int64
	n_layers::Int64
end

VAEVariationalObjective(target) = VAEVariationalObjective(MvNormal(zeros(dimension(target)), I), target, 10, 10)

function initialize(obj::VAEVariationalObjective)
	d = dimension(obj.target)
	dint = obj.internal_feature_dimension
	λdim = 2*(obj.n_layers*(4d*dint + 2d) + 4d*d+2d)
	return randn(Xoshiro(1),λdim)/λdim
end

function relu(x)
	return x >= 0.0 ? x : 0.0
end

function decode_and_logjacobian(obj::VAEVariationalObjective, λ, z, ϵ)
	d = length(z)
	dint = obj.internal_feature_dimension
	# pad z with zeros to start to get the right dimension
	y = zeros(2d)
	y[1:d] .= z
	idx = 0 #decoder starts at idx 0
	for n=1:obj.n_layers
		B = reshape(λ[idx+1:idx+2d*dint], (dint, 2d))
		idx += d*dint
		A = reshape(λ[idx+1:idx+2d*dint], (2d,dint))
		idx += 2d*dint
		b = λ[idx+1:idx+2d]
		idx += 2d
		y = relu.(A*(B*y) + b) + y
		# normalize
		y .-= sum(y)/(2d)
		y /= sqrt(sum(y.^2))
	end
	# last layer full transform
	A = reshape(λ[idx+1:idx+4d*d], (2d,2d))
	idx += 4d*d
	b = λ[idx+1:idx+2d]
	y = A*y+b

	return y[1:d] + y[d+1:end].*ϵ, sum(log.(abs.(y[d+1:end])))
end

function encoder_logprob(obj::VAEVariationalObjective, λ, x, z)
	d = length(x)
	dint = obj.internal_feature_dimension
	# pad x with zeros to start to get the right dimension
	y = vcat(x, zeros(d))
	idx = obj.n_layers*(4d*dint + 2d) + 4d*d+2d #encoder starts half way through the parameters
	for n=1:obj.n_layers
		B = reshape(λ[idx+1:idx+2d*dint], (dint, 2d))
		idx += d*dint
		A = reshape(λ[idx+1:idx+2d*dint], (2d,dint))
		idx += 2d*dint
		b = λ[idx+1:idx+2d]
		idx += 2d
		y = relu.(A*(B*y) + b) + y
		# normalize
		y .-= sum(y)/(2d)
		y /= sqrt(sum(y.^2))
	end
	# last layer full transform
	A = reshape(λ[idx+1:idx+4d*d], (2d,2d))
	idx += 4d*d
	b = λ[idx+1:idx+2d]
	y = A*y+b

	return -sum(log.(abs.(y[d+1:end]))) - 0.5*sum((z-y[1:d]).^2 ./ y[d+1:end].^2)
end


function function_estimate(obj::GeneralLocationScaleVariationalObjective, λ, n, rng)
	negative_elbo = 0.0
	for i=1:n
		ϵ = rand(rng, obj.q0)
		z = rand(rng, obj.q0)
		x, ljac = decode_and_logjacobian(obj, λ, z, ϵ)	
		try
			negative_elbo += -unbiased_logdensity(rng, obj.target, x) - ljac - encoder_logprob(obj, λ, x, z)
		catch err
			return Inf
		end
	end
	return negative_elbo/n
end

# this function is autodiffable with the gradient logp from Stan input as g
# avoids issues with enzyme with a compiled function in the mix
# it has the same derivative as the original negative elbo
# but note that it doesn't have the same value as the negative elbo
function _autodiffable_negative_elbo_proxy(λ, ϵ, z, g, obj::GeneralLocationScaleVariationalObjective)
	x, ljac = decode_and_logjacobian(obj, λ, z, ϵ)
	lpenc = encoder_logprob(obj, λ, x, z)
	return -sum(g .* x) - ljac - lpenc
end

function gradient_estimate(obj::GeneralLocationScaleVariationalObjective, λ, n, rng)
	g = zero(λ)
	for i=1:n
		ϵ = rand(rng, obj.q0)
		z = rand(rng, obj.q0)
		x, ljac = decode_and_logjacobian(obj, λ, z, ϵ)
		_, gn = unbiased_logdensity_and_gradient(rng, obj.target, x)
		dλ = zero(λ)
		dz = zero(z) # needed to avoid issues with unknown activity, but this dz gradient is never used
		Enzyme.autodiff(Enzyme.Reverse, _autodiffable_negative_elbo_proxy, Enzyme.Duplicated(λ,dλ), Enzyme.Const(ϵ), Enzyme.Duplicated(z,dz), Enzyme.Const(gn), Enzyme.Const(obj))
		g += dλ
	end
	if any(isnan.(g)) || any(isinf.(g))
		throw("Gradient Nan/Inf")
	end
	return g/n
end

