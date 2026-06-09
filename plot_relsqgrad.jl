include("utils.jl")
using Plots, StatsPlots, StatsBase, CategoricalArrays, DataFrames, DataFramesMeta, HDF5
gr()

function main()
	tuning = load("tuning.jld2")
	tuned_stepsizes = tuning["ordered_beststepsizes"]
	stepsizes = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e+0"]
	tuned_stepsize_idcs = [findfirst(==(s), stepsizes) for s in tuned_stepsizes]
	algs = tuning["ordered_algs"]

	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	# remove nonpdb targets
	remove_nonpdb!(targets)

	println("Note: this code assumes you've generated the Ensemble with the gradsq objective. If you generate the Ensemble with the obj objective the results will be incorrect.")

	## zip up the product of targets and objectives
	#problems = [(target, objective) for target in targets for objective in objectives]
	#rng = Xoshiro(1)
	#shuffle!(rng, problems)

	# integral time range
	tmin = 0.0
	tmax = 360.0

	## first find optimal objective across all algs/tuning params/iterations
	bestobjs = Inf*ones(length(objectives), length(targets))
	bestobjvars = Inf*ones(length(objectives), length(targets))
	bestts = Inf*ones(length(objectives), length(targets))
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			found_file = false
			for a in 1:length(algs)
				for s in 1:length(stepsizes)
					fn = "results/"*algs[a]*"-"*targets[t]*"-"*objectives[ob]*"-"*stepsizes[s]*"-1.jld2"
					if isfile(fn)
						found_file = true
						res = load(fn)
						if sum(res["nfaileval"] .== 0) > 0
							imin = argmin(res["objs"][res["nfaileval"] .== 0])
							objmin = (res["objs"][res["nfaileval"] .== 0])[imin]
							objvarmin = (res["objvars"][res["nfaileval"] .== 0])[imin]
							timemin = (res["ts"][res["nfaileval"] .== 0])[imin]
							if objmin < bestobjs[ob,t]
								bestobjs[ob,t] = objmin
								bestobjvars[ob,t] = objvarmin
								bestts[ob,t] = timemin
							end
						end
					end
				end
			end
			if isinf(bestobjs[ob,t]) && found_file
				println("Warning: Problem $(targets[t]) $(objectives[ob]) has no non-failed minimum objectives!")
			end
		end
	end

	# 
	failures = zeros(length(algs), length(objectives), length(targets))
	everyone_failed = zeros(length(objectives), length(targets))
	dfalgs = []
	dfobjnms = []
	dfgradsqs = []
	dfobjs = []
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			baseline_grad = Inf
			baseline_obj = Inf
			baselines = [("Ensemble", "1e-4")]
			for ba in baselines
				if isfile("results/"*ba[1]*"-"*targets[t]*"-"*objectives[ob]*"-"*ba[2]*"-1.jld2")
					res = load("results/"*ba[1]*"-"*targets[t]*"-"*objectives[ob]*"-"*ba[2]*"-1.jld2")
					tmpbaseline_grad = res["gradsqs"][end]
					tmpbaseline_obj = res["objs"][end]
					if tmpbaseline_grad < baseline_grad
						baseline_grad = tmpbaseline_grad
					end
					if tmpbaseline_obj < baseline_obj
						baseline_obj = tmpbaseline_obj
					end

				end
			end
			if baseline_grad == Inf || baseline_obj == Inf
				continue
			end
			allfail = 1
			for a in 1:length(algs)
				fn = "results/"*algs[a]*"-"*targets[t]*"-"*objectives[ob]*"-"*tuned_stepsizes[a]*"-1.jld2"
				push!(dfalgs, algs[a])
				push!(dfobjnms, objectives[ob])
				if isfile(fn)
					res = load(fn)
					# hard failures (we can prove it increased the objective)
					if res["nfaileval"][end] > 0 || res["objs"][end] - sqrt(res["objvars"][end]) >= res["objs"][1] + sqrt(res["objvars"][1])
					# soft failures (we can't prove it decreased the objective)
					#if res["nfaileval"][end] > 0 || res["objs"][end] + sqrt(res["objvars"][end]) >= res["objs"][1] - sqrt(res["objvars"][1])
						failures[a,ob,t] += 1
						push!(dfgradsqs, Inf)
						push!(dfobjs, Inf)
					else
						push!(dfgradsqs, res["gradsqs"][end]/baseline_grad) # res[objname][1])
						push!(dfobjs, res["objs"][end] - bestobjs[ob,t]) #baseline_obj) # res[objname][1])
						allfail = 0
					end
				else
					failures[a,ob,t] += 1
					push!(dfobjs, Inf)
					push!(dfgradsqs, Inf)
				end
			end
			everyone_failed[ob, t] += allfail
		end
	end

	df = DataFrame(algorithm = dfalgs, objective = dfobjnms, gradsq = dfgradsqs, obj = dfobjs)
	df.algorithm = categorical(df.algorithm, levels=algs, ordered=true)
	df.objective = categorical(df.objective, levels=objectives, ordered=true)
	df.gradsq[df.gradsq .>= 1e20] .= 1e20
	df.obj[isnan.(df.obj) .|| isinf.(df.obj)] .= 1000
	df.obj[df.obj .>= 1000] .= 1000

	# log10-transform the gradient relative values before plotting. A box plot of "value" plotted on log10 scale != a box plot of "log 10 value" plotted normally, and the usual interpretation should prefer the latter
	df.gradsq = log10.(df.gradsq)

        h5open("gradsq_rel.h5", "w") do h5
            gd = @groupby(df, :algorithm)
            names = [string(gd[i][1, :algorithm]) for i in 1:gd.ngroups]
            alpha_order_idx = sortperm(names)
            names           = names[alpha_order_idx]
            for (pos_offset, obj) in zip([-0.3, -0.1, 0.1, 0.3], objectives)
                sub    = @subset(df, :objective .== obj)
                subgd  = @groupby(sub, :algorithm)
                values = [subgd[i][:, :gradsq] for i in 1:gd.ngroups]
                values = values[alpha_order_idx]
                for (name, value) in zip(names, values)
                    write(h5, "$(name)_$(obj)", value)
                end
                write(h5, "position_$(obj)", alpha_order_idx .+ pos_offset)
            end
            write(h5, "position_labels", alpha_order_idx)
            write(h5, "labels", names)
        end

	p = @df df StatsPlots.groupedboxplot(:algorithm, :gradsq, group = :objective, xlabel="Algorithm", ylabel="Log₁₀ Relative Squared Gradient", legendtitle = "Objective", xrotation=90, yrotation=90, markersize=1, legend=false, size=(1200, 700), left_margin=12Plots.mm, bottom_margin=20Plots.mm, yticks=-20:2:20, ylimits = (-20,20), dpi=600) #10.0 .^(-20:2:10))
	hline!(p, [0.0], linestyle=:dash, color=:gray, label=nothing)
	Plots.savefig(p, "gradsq_rel.png")
	display(p)
	println("waiting")
	readline()

end


main()


