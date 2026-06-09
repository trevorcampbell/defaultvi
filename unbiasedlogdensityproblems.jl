# utility function for subsampled stan
function get_subsample_size(pdb, subsampled_posterior)
	modelnamesub = PosteriorDB.info(subsampled_posterior)["model_name"]
	datanamesub = PosteriorDB.info(subsampled_posterior)["data_name"]
	modelsub = PosteriorDB.model(pdb, modelnamesub)
	datasub = PosteriorDB.load(PosteriorDB.dataset(pdb, datanamesub))
	subsampleszstr = PosteriorDB.info(modelsub)["subsample_size"]
	subsample_sz = -1
	if occursin("*", subsampleszstr)
		subsamplesznms = split(subsampleszstr, "*")
		subsample_sz = 1
		for var in subsamplesznms
			subsample_sz *= datasub[var]
		end
	elseif occursin("+", subsampleszstr)
		subsamplesznms = split(subsampleszstr, "+")
		subsample_sz = 0
		for var in subsamplesznms
			subsample_sz += datasub[var]
		end
	elseif occursin("-", subsampleszstr)
		subsamplesznms = split(subsampleszstr, "-")
		subsample_sz = datasub[subsamplesznms[1]] - datasub[subsamplesznms[2]]
	else
		subsample_sz = datasub[subsampleszstr]
	end
	return subsample_sz
end

# custom logdensityproblems interface with unbiased logprob/density
unbiased_logdensity(rng::AbstractRNG, prb::StanProblem, x) = LogDensityProblems.logdensity(prb, x)
unbiased_logdensity_and_gradient(rng::AbstractRNG, prb::StanProblem, x) = LogDensityProblems.logdensity_and_gradient(prb, x)
dimension(prb::StanProblem) = LogDensityProblems.dimension(prb)

struct SubsampledStanProblem
	prb::StanProblem
	N::Int64
end
unbiased_logdensity(rng::AbstractRNG, prb::SubsampledStanProblem, x) = LogDensityProblems.logdensity(prb.prb, vcat(x, rand(rng, 1:prb.N)))
unbiased_logdensity_and_gradient(rng::AbstractRNG, prb::SubsampledStanProblem, x) = begin
 	l, g = LogDensityProblems.logdensity_and_gradient(prb.prb, vcat(x, rand(rng, 1:prb.N)))
 	return l, g[1:end-1]
end
dimension(prb::SubsampledStanProblem) = LogDensityProblems.dimension(prb.prb)-1

struct GaussianLocationProblem
	data::Vector{Vector{Float64}}
	μ0::Vector{Float64}
	Σ0inv::Matrix{Float64}
	Σinv::Matrix{Float64}
end

function unbiased_logdensity(rng::AbstractRNG, prb::GaussianLocationProblem, x)
	N = length(prb.data)
	n = Int(ceil(rand(rng)*N))
	return -0.5N*(prb.data[n]-x)'*prb.Σinv*(prb.data[n]-x) -0.5(x-prb.μ0)'*prb.Σ0inv*(x-prb.μ0)
end

function unbiased_logdensity_and_gradient(rng::AbstractRNG, prb::GaussianLocationProblem, x)
	N = length(prb.data)
	n = Int(ceil(rand(rng)*N))
	return -0.5N*(prb.data[n]-x)'*prb.Σinv*(prb.data[n]-x) -0.5(x-prb.μ0)'*prb.Σ0inv*(x-prb.μ0), N*prb.Σinv*(prb.data[n]-x) +prb.Σ0inv*(prb.μ0-x)
end
dimension(prb::GaussianLocationProblem) = length(prb.μ0)


