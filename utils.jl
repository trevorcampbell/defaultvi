using LogDensityProblems, PosteriorDB, StanLogDensityProblems
using LinearAlgebra, Random, Distributions, MCMCDiagnosticTools
using ProgressBars
using JLD2
using Enzyme
#using Plots
#plotlyjs()

function remove_nonpdb!(targets)
	nonpdb_targets = ["normnmf_liver", "normnmf_lung", "normnmf_ovary", "normnmf_breast", 
			   "normnmf_skin", "normnmf_stomach", "sparsepoiss_crime", "sparselog_prostate", "sparselog_ovarian", "sparselog_leukemia",
			   "sirnb_sirnb", "twocptpop_twocptpop", "neutropenia_neutropenia"]
	append!(nonpdb_targets, [x*"_subsampled" for x in nonpdb_targets])
	filter!(x -> !(x in nonpdb_targets), targets)
end

function lower_to_vec(L)
	return [L[i,j] for j in 1:size(L,2) for i in j:size(L,1)]
end

function vec_to_lower(v)
    n = Int((sqrt(8length(v) + 1) - 1) / 2)
    L = zeros(eltype(v), n, n)
    k = 1
    for j in 1:n
        for i in j:n
            L[i,j] = v[k]
            k += 1
        end
    end
    return L
end

function vec_to_diag(v)
    n = Int((sqrt(8length(v) + 1) - 1) / 2)
    Ld = zeros(eltype(v), n)
    k = 1
    for j in 1:n
	Ld[j] = v[k]
	k += n-j+1
    end
    return Ld
end

# return integral mean, variance
# using independently estimated functions/timepoints
function integrate(ts, objs, objvars, tmin, tmax, objmin=0.0, objminvar=0.0)
	@assert tmin <= tmax
	@assert tmin >= 0
	@assert issorted(ts)

	# create boundaries that are guaranteed to surround tmin, tmax
	xts = copy(ts)
	pushfirst!(xts, -1.0)
	push!(xts, max(tmax, ts[end])+1.0)
	coeffs = zero(xts)
	imin = searchsortedfirst(xts, tmin)
	imax = searchsortedfirst(xts, tmax)

	# handle special case where imin = imax separately
	if imin == imax
		if imin == imax == 1
			println("uh oh -- $(xts)")
		end
		γmin = (tmin - xts[imin-1])/(xts[imin]-xts[imin-1])
		γmax = (tmax - xts[imin-1])/(xts[imin]-xts[imin-1])
		coeffs[imin-1] = (tmax-tmin)/2 * (1-γmax + 1-γmin)
		coeffs[imin] =  (tmax-tmin)/2 *(γmax + γmin)
	else
		for i=2:length(xts)
			if xts[i] <= tmin || tmax < xts[i-1]
				continue
			elseif xts[i-1] <= tmin < xts[i]
				γ = (tmin - xts[i-1])/(xts[i]-xts[i-1])
				coeffs[i-1] += (xts[i]-tmin)/2.0 * (1.0 - γ)
				coeffs[i] += (xts[i]-tmin)/2.0 * (1.0 + γ)
			elseif xts[i-1] <= tmax < xts[i]
				γ = (tmax - xts[i-1])/(xts[i]-xts[i-1])
				coeffs[i-1] += (tmax-xts[i-1])/2.0 * (1.0 + 1.0 - γ)
				coeffs[i] += (tmax-xts[i-1])/2.0 * γ 
				break # can just break out now since tmax is passed
			else
				coeffs[i-1] += (xts[i]-xts[i-1])/2.0
				coeffs[i] += (xts[i]-xts[i-1])/2.0
			end
		end
	end

	# combine coeffs from the two endpoints (constant extrapolation)
	coeffs[2] += coeffs[1]
	coeffs[end-1] += coeffs[end]
	coeffs = coeffs[2:end-1]

	# return the objectives times coefficients
	return sum(coeffs.*(objs.-objmin)), sum((coeffs.^2).*objvars) + sum(coeffs)^2*objminvar
end

function augment_interpolate(ts, vals, newts)
	@assert length(ts) == length(vals)
	i = 1
	j = 1
	augts = Vector{Float64}()
	augvals = Vector{eltype(vals)}()
	while i <= length(ts) || j <= length(newts)
		#println("i $i j $j length(ts) $(length(ts)) length(vals) $(length(vals)) length(newts) $(length(newts))")
		if i <= length(ts) && j <= length(newts)
			if ts[i] <= newts[j]
				push!(augts, ts[i])
				push!(augvals, vals[i])
				i += 1
			else
				if i == 1
					push!(augts, newts[j])
					push!(augvals, vals[1])
				else
					push!(augts, newts[j])
					@assert  0 <= (newts[j] - ts[i-1])/(ts[i]-ts[i-1]) <= 1
					@assert  0 <= (ts[i] - newts[j])/(ts[i]-ts[i-1]) <= 1
					push!(augvals, vals[i-1]*(ts[i] - newts[j])/(ts[i]-ts[i-1]) + vals[i]*(newts[j] - ts[i-1])/(ts[i]-ts[i-1]))
				end
				j += 1
			end
		elseif i <= length(ts)
			push!(augts, ts[i])
			push!(augvals, vals[i])
			i += 1
		else
			push!(augts, newts[j])
			push!(augvals, vals[end])
			j += 1
		end
	end
	# remove duplicate ts
	i = 2
	while i <= length(augts)
		if augts[i] == augts[i-1]
			@assert isnan(augvals[i]) || isnan(augvals[i-1]) || abs(augvals[i] - augvals[i-1])/max(abs(augvals[i])+1, abs(augvals[i-1])+1) < 1e-10
			delidx = i
			if isnan(augvals[i-1])
				delidx = i-1
			end
			deleteat!(augvals, delidx)
			deleteat!(augts, delidx)
		else
			i += 1
		end
	end
	return augts, augvals
end

function get_at_t(t, ts, objs)
	if t <= ts[1]
		return objs[1]
	elseif t >= ts[end]
		return objs[end]
	else
		ind = searchsortedfirst(ts, t)
		return objs[ind-1] + (objs[ind]-objs[ind-1])*(t - ts[ind-1])/(ts[ind]-ts[ind-1])
	end
end

# welford algorithm for streaming mean/var estimation
# careful not to overwrite inputs
function welford_update(x, f, σ2, g, τ2, n, objective, rng)
	fn = function_estimate(objective, x, 1, copy(rng))
	gn = gradient_estimate(objective, x, 1, rng)
	n += 1
	δf = fn-f
	fnew = f+δf/n
	δg = gn-g
	gnew = g+δg/n
	return fnew, σ2+δf*(fn-fnew), gnew, τ2 + δg.*(gn-gnew), n
end

# algorithms
include("adam.jl")
include("adamavg.jl")
include("adasls.jl")
include("adamsls.jl")
include("adagrad.jl")
include("amsgrad.jl")
include("averaging.jl")
include("bigbatch.jl")
include("cocob.jl")
include("dadaptadam.jl")
include("dadaptsgd.jl")
include("dog.jl")
include("dogmom.jl")
include("dowg.jl")
include("dowgmom.jl")
include("ftrl.jl")
include("hypergradient.jl")
include("lion.jl")
include("mechanic.jl")
include("pflug.jl")
include("ponos.jl")
include("rabvi.jl")
include("rabvistream.jl")
include("saa.jl")
include("saana.jl")
include("saalbfgs.jl")
include("saacg.jl")
include("sgd.jl")
include("sls.jl")

# unbiased logdensity problems
include("unbiasedlogdensityproblems.jl")

# MAP objectives 
include("map.jl")

# Gaussian objecties
include("locationscalevi.jl")

