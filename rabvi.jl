# RABVI without the stopping rule
# Welandawe et al 
# A Framework for Improving the Reliability of Black-box Variational Inference
# JMLR 25(219):1−71, 2024.
# https://arxiv.org/abs/2203.15945
# This was originally just written for fixed-step-size SGD but you can "RABVI" any algorithm with a fixed step size
# kcheck=Wmin=200 per viabel repo

mutable struct RABVI{Alg}
	alg::Alg
	iter::Int64
	prevouteriter::Int64
	nf::Int64
	ng::Int64
	opt_time::Float64
	ρ::Float64
	Wmin::Int64
	ESSmin::Float64
	ϵ::Float64
	kconv::Int64
	Wcheck::Int64
	x::Vector{Float64}
	xtrace::Vector{Vector{Float64}}
end

RABVI{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg} = RABVI{Alg}(Alg(copy(x), stepsz, seed), 0, 0, 0, 0, 0.0, 0.5, 200, 50.0, 0.1, -1, -1, copy(x), Vector{Vector{Float64}}())
RABVI{Alg}(dimension::Int64, stepsz, seed) where {Alg} = RABVI{Alg}(zeros(dimension), stepsz, seed)

function coordinate_mcse_ess(v::Vector{Vector{Float64}})
	mean = sum(v)/length(v)
	sd = sqrt.(mapreduce(x -> x.^2, +,  (v .- mean))/(length(v)-1))
	effs = [ess([x[i] for x in v]) for i=1:length(mean)]
    return mean, sd ./ sqrt.(effs), effs
end

function r_hat_windowed(v::Vector{Vector{Float64}}, windows::Vector{Int64})
	rhats = [maximum([rhat([x[i] for x in v[end-w+1:end]]) for i in 1:length(v[1])]) for w in windows]
	bestind = argmin(rhats)
	return rhats[bestind], windows[bestind]
end

function step!(rab::RABVI{Alg}, obj) where {Alg}
	rab.iter += 1

	# compute the iteration number for the current inner FASO iteration
	k = rab.iter - rab.prevouteriter

	# take an optimization step with timing
	t0 = time_ns()/1e9
	step!(rab.alg, obj)
	rab.opt_time += time_ns()/1e9 - t0

	# store iterate history
	push!(rab.xtrace, copy(rab.alg.x))

	# update the running average 
	# in original RABVI, this does not happen, but we need to be able to report a reasonable iterate average at all times
	# it won't change RABVI at its outer iterations since rab.x will be overwritten below during check iterations per FASO code
	rab.x += (rab.alg.x - rab.x)/rab.iter #use the full iteration number here to avoid noisy movement after updating step size

	# if Rhat hasn't converged yet, check for convergence
	Wupper = Int64(floor(k*0.95))
	if rab.kconv < 0 && k % rab.Wmin == 0 && Wupper > rab.Wmin
		Ws = unique(collect(Int64.(floor.(range(rab.Wmin,Wupper,5)))))
		Rhatbest, Wbest = r_hat_windowed(rab.xtrace, Ws)
		# overwrite rab.x with current best window average (per FASO code)
		rab.x = sum(rab.xtrace[end-Wbest+1:end])/Wbest
		if Rhatbest <= 1.1
			rab.Wcheck = Wbest
			rab.kconv = k - Wbest
		end
	end

	# if Rhat has converged, check for MCSE convergence
	if rab.kconv >= 0 && rab.iter - rab.kconv == rab.Wcheck
		t0 = time_ns()/1e9
		converged_mean, converged_se, converged_effs = coordinate_mcse_ess(rab.xtrace[end-rab.Wcheck+1:end])
		mcse_time = time_ns()/1e9 - t0

		# overwrite rab.x with current Wcheck average (per FASO code)
		rab.x = converged_mean

		# update Wcheck based on computation time ratios
		rel_opt_time = rab.opt_time/rab.iter
		rel_mcse_time = mcse_time/rab.Wcheck
		rel_time_ratio = rel_opt_time/rel_mcse_time
		recheck_scale = max(1.05, 1+1/sqrt(1+rel_time_ratio))
		rab.Wcheck = Int64(floor(recheck_scale*rab.Wcheck+1))

		if maximum(converged_se) <= rab.ϵ && minimum(converged_effs) >= rab.ESSmin
			# Rhat, MCSE, ESS all converged. Update step size and ESS threshold and reset memory as needed
			rab.Wcheck = -1
			rab.kconv = -1
			rab.alg.γ *= rab.ρ
			rab.ϵ *= rab.ρ
			rab.prevouteriter = rab.iter
			rab.xtrace = Vector{Vector{Float64}}()
		end
	end

	# grab the current number of gradient/function calls from the inner alg
	rab.nf = rab.alg.nf
	rab.ng = rab.alg.ng
end
