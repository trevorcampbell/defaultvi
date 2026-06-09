include("utils.jl")

function optimize!(algorithm, objective, evalobjective, Tmax, seed)
	nfail = 0
	nf = [0.0]
	ng = [0.0]
	ni = [0]
	ts = [0.0]
	xs = [copy(algorithm.x)]
	t = 0.0
	i = 0
	while t < Tmax
		t0 = time_ns()
		try
			step!(algorithm, objective)
		catch err
			nfail += 1
		end
		t += (time_ns() - t0)/10^9
		if log2(algorithm.ng) >= i || t >= Tmax
			push!(ts, t)
			push!(xs, copy(algorithm.x))
			push!(nf, algorithm.nf)
			push!(ng, algorithm.ng)
			push!(ni, algorithm.iter)
			i += 1
		end
	end
	
	# evaluate the objective and gradient sqnorm
	rng = Xoshiro(seed)
	snr = 4
	n0 = 128
	nfaileval = zeros(length(xs))
	ns = zeros(length(xs))
	fs = zeros(length(xs))
	σ2s = zeros(length(xs))
	gs = []
	τ2s = []
	for i=1:length(xs)
		push!(gs, zero(algorithm.x))
		push!(τ2s, zero(algorithm.x))
	end
	t0 = time_ns()/1e9
	# take at least n0 draws each
	for i=1:length(xs)
		for n=1:n0
			try
				fs[i], σ2s[i], gs[i], τ2s[i], ns[i] = welford_update(xs[i], fs[i], σ2s[i], gs[i], τ2s[i], ns[i], evalobjective, rng)
			catch err
				nfaileval[i] += 1
			end
		end
	end
	# use the remaining time to take draws where the snr isn't high
	while time_ns()/1e9 - t0 < Tmax
		# reservoir sampling to choose uniformly over the pairs of items whose snr is high
		fsorted = sortperm(fs)
		idx = 0
		k = 0
		for i=1:(length(xs)-1)
			i1 = fsorted[i]
			i2 = fsorted[i+1]
			# σ2s gets divided by n^2 because 1/n to estimate variance of one draw from Welford, 1/n again to estimate variance of f
			if abs(fs[i1] - fs[i2]) <= 4*sqrt(σ2s[i1]/ns[i1]^2 + σ2s[i2]/ns[i2]^2)
				# i has too much noise. choose i1,i2 proportional to their variances
				k += 1
				prbi = (σ2s[i1]/ns[i1]^2)/(σ2s[i1]/ns[i1]^2 + σ2s[i2]/ns[i2]^2)
				if rand() <= 1/k
					if rand() <= prbi
						idx = i1
					else
						idx = i2
					end
				end
			end
		end
		# if all pairs already obey the snr constraint, choose one uniformly
		if k == 0
			idx = 1 + Int(floor(rand()*length(xs)))
		end

		# Welford Alg to update the mean, variance estimate for idx
		try
			fs[idx], σ2s[idx], gs[idx], τ2s[idx], ns[idx] = welford_update(xs[idx], fs[idx], σ2s[idx], gs[idx], τ2s[idx], ns[idx], evalobjective, rng)
		catch err
			#println(err)
			nfaileval[idx] += 1
		end
	end
	# convert g, tau into scalars, divide counts for moment stats
	for i=1:length(xs)
		gs[i] = sum(gs[i].^2)
		# these get divided by n^2 because 1/n to estimate variance of one draw from welford, 1/n again to estimate variance of f, g
		τ2s[i] = sum(τ2s[i])/ns[i]^2
		σ2s[i] /= ns[i]^2
	end
	return nf, ng, ni, ts, xs, fs, σ2s, gs, τ2s, ns, nfail, nfaileval
end

# Arguments
# AlgType
# Target Name
# Objective Name
# Step Size
# Seed
# Tmax
# standir
# resultsdir
function main(t0)

	# parse command line args
	AlgTypes = Dict{String,Type}()
	# base algorithms
	AlgTypes["Adam"] = Adam
	AlgTypes["AdamAvg"] = AdamAvg
	AlgTypes["AdaGrad"] = AdaGrad
	AlgTypes["AMSGrad"] = AMSGrad
	AlgTypes["AdaSLS"] = AdaSLS
	AlgTypes["AdamSLS"] = AdamSLS
	AlgTypes["BigBatch"] = BigBatch
	AlgTypes["COCOB"] = COCOB
	AlgTypes["DAdaptAdam"] = DAdaptAdam
	AlgTypes["DAdaptSGD"] = DAdaptSGD
	AlgTypes["DoG"] = DoG
	AlgTypes["DoGMom"] = DoGMom
	AlgTypes["DoWG"] = DoWG
	AlgTypes["DoWGMom"] = DoWGMom
	AlgTypes["FTRL"] = FTRL
	AlgTypes["Lion"] = Lion
	AlgTypes["Pflug"] = Pflug
	AlgTypes["PoNoS"] = PoNoS
	AlgTypes["SAA"] = SAA
	AlgTypes["SAANA"] = SAANA
	AlgTypes["SAALBFGS"] = SAALBFGS
	AlgTypes["SAACG"] = SAACG
	AlgTypes["SGD"] = SGD
	AlgTypes["SLS"] = SLS
	
	# mechanic, RABVI, HyperGrad versions
	for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]
		AlgTypes["Mecha"*T] = Mechanic{AlgTypes[T]}
		AlgTypes["RABVI"*T] = RABVI{AlgTypes[T]}
		AlgTypes["SRABVI"*T] = StreamingRABVI{AlgTypes[T]}
		AlgTypes["Hyper"*T] = HyperGradient{AlgTypes[T],4}
	end

	AlgType = AlgTypes[ARGS[1]]
	targetnm = ARGS[2]
	objectivenm = ARGS[3]
	seed = parse(Int64,ARGS[4])
	Tmax = parse(Float64,ARGS[5])
	standir = ARGS[6]
	resultsdir = ARGS[7]

	# remove the subsampled suffix (always postprocess on least noisy version of a problem)
	if occursin("_subsampled", targetnm)
		evaltargetnm = targetnm[begin:end-11]
	end

	#load the problem
	target = nothing
	evaltarget = nothing
	pdb = PosteriorDB.database()
	if targetnm == "gaussian-location"
		rng = Xoshiro(1)
		d = 40
		N = 1000
		L0 = Matrix(I,d,d)
		L = Matrix(I,d,d)
		μ0 = zeros(d)
		data = Vector{Vector{Float64}}()
		μ = μ0 + L0*randn(rng,d)
		for i=1:N
			push!(data, μ+L*randn(rng,d))
		end
		target = GaussianLocationProblem(data, μ0, inv(L0*L0'), inv(L*L'))
		evaltarget = target
	elseif targetnm in PosteriorDB.posterior_names(pdb)
		if occursin("_subsampled", targetnm)
			redirect_stderr(devnull) do
				post = PosteriorDB.posterior(pdb, targetnm)
				subsz = get_subsample_size(pdb, post)
				target = SubsampledStanProblem(StanProblem(post, standir), subsz)
				evalpost = PosteriorDB.posterior(pdb, evaltargetnm)
				evaltarget = StanProblem(evalpost, standir)
			end
		else
			redirect_stderr(devnull) do
				post = PosteriorDB.posterior(pdb, targetnm)
				target = StanProblem(post, standir)
				evaltarget = target
			end
		end
	else
		error("No known target with name $targetnm")
		return
	end

	# load the objective
	objective = nothing
	evalobjective = nothing
	if objectivenm == "MAP"
		objective = MAPObjective(target)
		evalobjective = MAPObjective(evaltarget)
	elseif objectivenm == "DiagGaussianVI"
		objective = DiagGaussianVariationalObjective(target)
		evalobjective = DiagGaussianVariationalObjective(evaltarget)
	elseif objectivenm == "FullGaussianVI"
		objective = FullGaussianVariationalObjective(target)
		evalobjective = FullGaussianVariationalObjective(evaltarget)
	elseif objectivenm == "VAEVI"
		objective = VAEVariationalObjective(target)
		evalobjective = VAEVariationalObjective(evaltarget)
	else
		error("No known objective with name $objectivenm")
		return
	end

	# initialize
	x0 = initialize(objective)

	# force compilation of step! with alg to avoid miscalculation of computation time later on
	_alg = AlgType(x0, 1e-12, seed)
	step!(_alg, objective)

	tf = time_ns()/1e9
	println("Setup time for $(ARGS[1]) $(ARGS[2]) $(ARGS[3]): $(tf-t0)")

	# record a calibration exercise for timing across different cluster nodes
	_rng = Xoshiro(1)
	_d = dimension(target)
	_gtmp = zeros(_d)
	try
		_ftmp, _gtmp = unbiased_logdensity_and_gradient(_rng, target, 1e-12*randn(_d))
	catch err
	end
	t0 = time_ns()/1e9
	tf = t0
	calibration_count = 0
	calibration_fails = 0
	while tf - t0 < 20.0
		calibration_count += 1
		try
			_ftmp, _gi = unbiased_logdensity_and_gradient(_rng, target, 1e-12*randn(_d))
			_gtmp += _gi/calibration_count
		catch err
			calibration_fails += 1
		end
		tf = time_ns()/1e9
	end
	println("Calibration for $(ARGS[1]) $(ARGS[2]) $(ARGS[3]): Time $(tf-t0) Call Count $(calibration_count) Fail Count $(calibration_fails)")

	# run opt for each step size
	for i=8:length(ARGS)
		t0 = time_ns()/1e9
		# check if results already exist; if so, just quit
		fnm = resultsdir*"/$(ARGS[1])-$(ARGS[2])-$(ARGS[3])-$(ARGS[i])-$(ARGS[4]).jld2"
		if isfile(fnm)
			println("$fnm exists. Using cached version.")
			continue
		else
			println("$fnm doesn't exist; optimizing.")
		end
		stepsz = parse(Float64, ARGS[i])
		alg = AlgType(x0, stepsz, seed)
		nfs, ngs, niters, ts, xs, objs, objvars, gradsqs, gradvars, nevalsamps, nfail, nfaileval = optimize!(alg, objective, evalobjective, Tmax, seed)
		jldsave(fnm; stepsz, nfail, nfaileval, nfs, ngs, niters, ts, objs, objvars, gradsqs, gradvars, nevalsamps, calibration_count, calibration_fails)

		tf = time_ns()/1e9
		println("Execution time for $(ARGS[1]) $(ARGS[2]) $(ARGS[3]) step size $(ARGS[i]) : $(tf-t0)")
	end
end

t0 = time_ns()/1e9
main(t0)
