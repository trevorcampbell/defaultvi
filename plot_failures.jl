include("utils.jl")
using Plots, StatsPlots, StatsBase, CategoricalArrays, DataFrames, DataFramesMeta, HDF5
gr()

function main()
	tuning = load("tuning.jld2")
	tuned_stepsizes = tuning["beststepsizes"]
	stepsizes = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e+0"]
	tuned_stepsize_idcs = [findfirst(==(s), stepsizes) for s in tuned_stepsizes]
	algs = tuning["algs"]

	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	# remove the nonposteriorDB targets
	remove_nonpdb!(targets)

	## zip up the product of targets and objectives
	#problems = [(target, objective) for target in targets for objective in objectives]
	#rng = Xoshiro(1)
	#shuffle!(rng, problems)

	# integral time range
	tmin = 0.0
	tmax = 360.0

	# results files have fields
	# objs (KL estimates) objvars (variance in estimates)
	# gradsqs (estimate of gradient sq norms) gradvars (expected value of ||grad - Egrad||^2)
	objname = "objs"
	varname = "objvars"

	hard_failures = zeros(length(algs), length(stepsizes), length(objectives), length(targets))
	soft_failures = zeros(length(algs), length(stepsizes), length(objectives), length(targets))
	tuned_hard_failures = zeros(length(algs), length(objectives), length(targets))
	tuned_soft_failures = zeros(length(algs), length(objectives), length(targets))
	everyone_soft_failed = zeros(length(objectives), length(targets))
	everyone_hard_failed = zeros(length(objectives), length(targets))
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			everyone_soft_failed[ob,t] = 1
			everyone_hard_failed[ob,t] = 1
			for a in 1:length(algs)
				# un-tuned results (across step sizes)
				for s in 1:length(stepsizes)
					fn = "results/"*algs[a]*"-"*targets[t]*"-"*objectives[ob]*"-"*stepsizes[s]*"-1.jld2"
					if isfile(fn)
						res = load(fn)
						# hard failures (either failed evaluation samples, or we can prove it increased the objective)
						if res["nfaileval"][end] > 0 || res[objname][end] - sqrt(res[varname][end]) >= res[objname][1] + sqrt(res[varname][1])
							hard_failures[a,s,ob,t] = 1
							soft_failures[a,s,ob,t] = 1
						#soft failures (we can't prove it decreased the objective)
						elseif res[objname][end] + sqrt(res[varname][end]) >= res[objname][1] - sqrt(res[varname][1])
							soft_failures[a,s,ob,t] = 1
							everyone_hard_failed[ob,t] = 0
						else
							everyone_hard_failed[ob,t] = 0
							everyone_soft_failed[ob,t] = 0
						end
					else
						hard_failures[a,s,ob,t] = 1
						soft_failures[a,s,ob,t] = 1
					end
				end
				# tuned results
				tuned_hard_failures[a,ob,t] = hard_failures[a,tuned_stepsize_idcs[a],ob,t]
				tuned_soft_failures[a,ob,t] = soft_failures[a,tuned_stepsize_idcs[a],ob,t]
			end
		end
	end

	hardfailprobs = 100*(sum(hard_failures, dims=(3,4)))/(length(objectives)*length(targets))
	softfailprobs = 100*(sum(soft_failures, dims=(3,4)))/(length(objectives)*length(targets))

        h5open("failure_grid.h5", "w") do h5
            write(h5, "stepsizes",  stepsizes)
            write(h5, "heatmap",    Array(100 .- softfailprobs[end:-1:2,:]'))
            write(h5, "algorithms", algs[end:-1:2])
        end

	# only plot algs end:-1:2 to remove Ensemble (shouldn't be shown in this grid plot)
	p = Plots.heatmap(100.0 .- softfailprobs[end:-1:2,:], yticks=(1:(length(algs)-1), algs[end:-1:2]), xticks=(1:length(stepsizes), stepsizes), ylabel="Algorithm", xrotation=45, xlabel="Step Size", colorbar_title = "(100 - Failure Probability)", c=cgrad(:inferno), tickfontsize=7, size=(600,800), dpi=600)
	Plots.savefig(p, "failure_grid.png")
	display(p)
	println("waiting")
	readline()

	# compute per-objective fail probabilities and convert to DF for easier grouped plotting
	tunedsoftfailprobs = 100*sum(tuned_soft_failures, dims=3)/length(targets)
	tunedhardfailprobs = 100*sum(tuned_hard_failures, dims=3)/length(targets)
	everyhardfailprobs = 100*sum(everyone_hard_failed,dims=2)/length(targets)
	everysoftfailprobs = 100*sum(everyone_soft_failed,dims=2)/length(targets)
	dfalgs = Vector{String}()
	dfobjs = Vector{String}()
	dfhvals = Vector{Float64}()
	dfsvals = Vector{Float64}()
	for a in 1:length(algs)
		for ob in 1:length(objectives)
			push!(dfalgs, algs[a])
			push!(dfobjs, objectives[ob])
			push!(dfhvals, tunedhardfailprobs[a,ob])
			push!(dfsvals, tunedsoftfailprobs[a,ob])
		end
	end
	# to add the "all failed" category 
	for ob in 1:length(objectives)
		push!(dfalgs, "All")
		push!(dfobjs, objectives[ob])
		push!(dfhvals, everyhardfailprobs[ob])
		push!(dfsvals, everysoftfailprobs[ob])
	end

	# compute fail ordering
	orderingfailprobs = vec(sum(tunedsoftfailprobs,dims=2)/length(objectives))
	ordering = sortperm(orderingfailprobs)
	algs = algs[ordering]

	df = DataFrame(algorithm = dfalgs, objective = dfobjs, svalue = dfsvals, hvalue = dfhvals)
	df.algorithm = categorical(df.algorithm, levels=vcat(["All"], algs), ordered=true)
	df.objective = categorical(df.objective, levels=objectives, ordered=true)

        sub      = @subset(df, :objective .== "MAP")
        alg_pos  = sortperm(sub.algorithm)
        algs_all = Array(sub.algorithm)[alg_pos]

        h5open("failure_bars.h5", "w") do h5
            for (pos_offset, obj) in zip([-0.3, -0.1, 0.1, 0.3], objectives)
                sub = @subset(df, :objective .== obj)
                write(h5, "$(obj)_svalue", sub[alg_pos, :svalue])
                write(h5, "$(obj)_hvalue", sub[alg_pos, :hvalue])
                write(h5, "$(obj)_pos",    collect((1:length(alg_pos)) .+ pos_offset))
            end
            write(h5, "position_labels", collect(1:length(alg_pos)))
            write(h5, "labels", algs_all[alg_pos])
        end

	clrs = [palette(:default)[i] for i in levelcode.(df.objective)]

	p = @df df StatsPlots.groupedbar(:algorithm, :svalue, group = :objective, bar_position=:dodge, xlabel="Algorithm", 
									ylabel="Failure Probability", legendtitle = "Objective", xrotation=90, yrotation=90, 
									 markersize=1, legend=false, size=(1200, 700), left_margin=12Plots.mm, bottom_margin=20Plots.mm, dpi=600, alpha=0.5)
	@df df StatsPlots.groupedbar!(p, :algorithm, :hvalue, group = :objective, bar_position=:dodge, xlabel="Algorithm", 
								  	ylabel="Failure Probability", legendtitle = "Objective", xrotation=90, yrotation=90, color=clrs,
									 markersize=1, legend=false, size=(1200, 700), left_margin=12Plots.mm, bottom_margin=20Plots.mm, dpi=600)
	Plots.savefig(p, "failure_bars.png")
	display(p)
	println("waiting for key press + enter")
	readline()
	
end

main()
