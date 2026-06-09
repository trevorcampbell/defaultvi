include("utils.jl")

function sweeping_rwmh(prb, prbsub, x, iterations, stepsizes, rng)
	accs = zero(stepsizes)
	d = length(x)
	μ = zeros(d)
	Σ = zeros(d,d)
	lμ = 0.0
	lσ2 = 0.0
	lμsub = 0.0
	lσ2sub = 0.0
	n = 0
	lp = LogDensityProblems.logdensity(prb, x)
	for i in ProgressBar(1:iterations)
		for s=1:length(stepsizes)
			xp = x + stepsizes[s]*randn(rng, d)
			try
				lpp = LogDensityProblems.logdensity(prb, xp)
				if log(rand(rng)) <= lpp - lp
					x .= xp
					lp = lpp
					accs[s] += 1
				end
			catch err
			end
		end
		# after burn-in, aggregate statistics
		if i > iterations/2
			n += 1
			# covariance matrix estimation
			δ = x - μ
			μ += δ/n
			δ2 = x - μ
			Σ += δ.*δ2'
			# sampling noise estimation
			δ = lp - lμ
			lμ += δ/n
			δ2 = lp - lμ
			lσ2 += δ*δ2
			# subsampling noise estimation
			if prbsub != nothing
				lpsub = unbiased_logdensity(rng, prbsub, x)
				δ = lpsub - lμsub
				lμsub += δ/n
				δ2 = lpsub - lμsub
				lσ2sub += δ*δ2
			end
		end
	end
	return cond(Σ/n), lσ2/n, lσ2sub/n, accs/iterations
end

function main()
	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	# remove the non-posteriorDB targets
	remove_nonpdb!(targets)

	condnums = zeros(length(targets))
	noises = zeros(length(targets))
	noisesubs = zeros(length(targets))
	dimensions = zeros(length(targets))
	for t in 1:length(targets)
		if occursin("_subsampled", targets[t])
			continue
		end
		println("Estimating problem characteristics for $(targets[t])")
		d = 0
		post = nothing
		postsub = nothing
		prb = nothing
		prbsub = nothing
		redirect_stderr(devnull) do
			post = PosteriorDB.posterior(pdb, targets[t])
			prb = StanProblem(post, "stan")
			d = LogDensityProblems.dimension(prb)
			if targets[t]*"_subsampled" in targets 
				postsub = PosteriorDB.posterior(pdb, targets[t]*"_subsampled")
				subsz = get_subsample_size(pdb, postsub)
				prbsub = SubsampledStanProblem(StanProblem(postsub, "stan"), subsz)
			end
		end
		stepsizes = 10 .^(-6:0.5:1)
		iterations = max(10000, 10*d)
		x = zeros(d)
		κ, σ2, σ2sub, accs = sweeping_rwmh(prb, prbsub, x, iterations, stepsizes, Xoshiro(1))
		println("Cond: $κ Noise: $(sqrt(σ2)) NoiseSub: $(sqrt(σ2sub))")
		println("Accs: $(100*accs)")
		dimensions[t] = d
		noises[t] = sqrt(σ2)
		noisesubs[t] = sqrt(σ2sub)
		condnums[t] = κ
		jldsave("problem_characteristics.jld2"; targets, dimensions, condnums, noises, noisesubs)
	end
end


main()


