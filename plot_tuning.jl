include("utils.jl")
using Plots, StatsPlots, StatsBase, DataFrames, CategoricalArrays, DataFramesMeta, HDF5, StableRNGs
gr()

function main()
    rng = StableRNG(1234)

	# algorithms
	algs = ["Ensemble", "Adam","AdamAvg","AdaGrad","AMSGrad","AdaSLS","AdamSLS","BigBatch","COCOB","DAdaptAdam","DAdaptSGD","DoG","DoWG","DoGMom","DoWGMom","FTRL","Lion","Pflug","PoNoS","SAA","SAANA","SAALBFGS","SAACG","SGD","SLS"]
	append!(algs, ["Mecha"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["RABVI"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["SRABVI"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["Hyper"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])

	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	# remove targets from outside posteriordb
	remove_nonpdb!(targets)

	# integral time range
	tmin = 0.0
	tmax = 360.0

	stepsizes = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e+0"]

	# results files have fields
	# objs (KL estimates) objvars (variance in estimates)
	# gradsqs (estimate of gradient sq norms) gradvars (expected value of ||grad - Egrad||^2)
	objname = "objs" #"gradsqs"
	varname = "objvars" #"gradvars"
	objtype = "final" #"integral"

	# first find optimal objective and calibration counts across all algs/tuning params/iterations
	bestobjs = []
	bestobjvars = []
	bestts = []
	bestobjs = Inf*ones(length(objectives), length(targets))
	bestobjvars = Inf*ones(length(objectives), length(targets))
	bestts = tmax*ones(length(objectives), length(targets))
	calibration_count_medians = Inf*ones(length(objectives), length(targets))
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			calib_count_problem = []
			found_file = false
			for a in 1:length(algs)
				for s in 1:length(stepsizes)
					fn = "results/"*algs[a]*"-"*targets[t]*"-"*objectives[ob]*"-"*stepsizes[s]*"-1.jld2"
					if isfile(fn)
						found_file = true
						res = load(fn)
						if sum(res["nfaileval"] .== 0) > 0
							push!(calib_count_problem, res["calibration_count"])
							imin = argmin(res[objname][res["nfaileval"] .== 0])
							objmin = (res[objname][res["nfaileval"] .== 0])[imin]
							objvarmin = (res[varname][res["nfaileval"] .== 0])[imin]
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
			calibration_count_medians[ob,t] = length(calib_count_problem) > 0 ? median(calib_count_problem) : -1
			if isinf(bestobjs[ob,t]) && found_file
				println("Warning: Problem $(targets[t]) $(objectives[ob]) has no non-failed minimum objectives!")
			end
		end
	end

	results = Inf*ones(length(algs), length(stepsizes), length(objectives), length(targets))
	ranks = Inf*ones(length(algs), length(stepsizes), length(objectives), length(targets))
	for a in 1:length(algs)
		for ob in 1:length(objectives)
			for t in 1:length(targets)
				for s in 1:length(stepsizes)
					fn = "results/"*algs[a]*"-"*targets[t]*"-"*objectives[ob]*"-"*stepsizes[s]*"-1.jld2"
					if isfile(fn)
						res = load(fn)
						
						ts_calibrated = res["ts"]*res["calibration_count"]/calibration_count_medians[ob,t]

						# # at tmax objective
						if objtype == "final"
							if any(res["nfaileval"] .> 0)
								results[a,s,ob,t] = Inf
							else
								results[a,s,ob,t] = get_at_t(tmax, ts_calibrated, res[objname])
							end
						elseif objtype == "integral"
							 # integral objective
							if any(res["nfaileval"] .> 0)
								results[a,s,ob,t] = Inf
							else
								results[a,s,ob,t], _ = integrate(ts_calibrated, res[objname], res[varname], tmin, tmax, bestobjs[ob,t], bestobjvars[ob,t])
							end
						else
							throw(Error("Unknown objective type"))
						end
					end
				end
				# to avoid systematically ordered ties due to Infs, shuffle the ones with Inf results
				ranks[a, :, ob, t] = sortperm(sortperm(results[a, :, ob, t]))
				infs = isinf.(results[a, :, ob, t])
				ranks[a, infs, ob, t] .= shuffle(rng, ranks[a, infs, ob, t])
			end
		end
	end

	avgranks = sum(ranks, dims=(3,4))/(length(objectives)*length(targets))
	# get best step size index
	beststepsize_idcs = Vector{Int}()
	for a=1:length(algs)
		push!(beststepsize_idcs, last(findall(==(minimum(avgranks[a,:])), avgranks[a,:])))
	end
	beststepsizes = [stepsizes[s] for s in beststepsize_idcs]
	
	for a in 1:length(algs)
		println("$(algs[a]): Best Rank Stepsize: $(beststepsizes[a])")
	end

	# plot ranks from end:-1:2 to skip index 1 which is Ensemble (should not be included in the tuning grid)
        h5open("avgranks.h5", "w") do h5
            write(h5, "stepsizes",  stepsizes)
            write(h5, "heatmap",    Array(avgranks[end:-1:2,:]'))
            write(h5, "algorithms", algs[end:-1:2])
        end

	p = Plots.heatmap(avgranks[end:-1:2,:], yticks=(1:(length(algs)-1), algs[end:-1:2]), xticks=(1:length(stepsizes), stepsizes), ylabel="Algorithm", xrotation=45, xlabel="Step Size", colorbar_title = "Average Rank", c=cgrad(:inferno, rev=true), tickfontsize=7, size=(600, 800), dpi=600)
	Plots.savefig(p, "tuning_grid.png")
	display(p)
	println("waiting for key press + enter")
	readline()

	results_tuned = Inf*ones(length(objectives), length(targets), length(algs))
	ranks_tuned = Inf*ones(length(objectives), length(targets), length(algs))
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			for a in 1:length(algs)
				results_tuned[ob,t,a] = results[a, beststepsize_idcs[a], ob, t]
			end
			ranks_tuned[ob,t,:] = sortperm(sortperm(results_tuned[ob,t,:]))
			infs = isinf.(results_tuned[ob,t, :])
			ranks_tuned[ob,t, infs] .= shuffle(rng, ranks_tuned[ob,t, infs])
		end
	end
	ranks_tuned = Int.(ranks_tuned)
	
	probs = zeros(length(algs), length(algs))
	wins = zeros(length(algs), length(algs))
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			for a in 1:length(algs)
				probs[a, ranks_tuned[ob,t,a]] += 1
			end
			wins += (results_tuned[ob,t, :]) .< (results_tuned[ob,t, :]')
		end
	end
	probs *= 100/(length(objectives)*length(targets))
	wins  *= 100/(length(objectives)*length(targets))

	# reorder the rows of probs and algsts in order of average rank
	avgrank = vec(sum(ranks_tuned, dims=(1,2))/(length(objectives)*length(targets)))
	ordering = sortperm(avgrank)
	ordered_probs = probs[ordering, :]
	ordered_algs = algs[ordering]
	ordered_beststepsizes = beststepsizes[ordering]

	jldsave("tuning.jld2"; ordered_algs, ordered_beststepsizes, algs, beststepsizes)

    println(ordered_algs)

	# compute average ranks for each objective group and plot bars
	dfalgs = Vector{String}()
	dfobjs = Vector{String}()
	dfvals = Vector{Float64}()
	for a in 1:length(algs)
		algidx = ordering[a]
		for ob in 1:length(objectives)
			for t in 1:length(targets)
				push!(dfalgs, algs[algidx])
				push!(dfobjs, objectives[ob])
				push!(dfvals, ranks_tuned[ob,t,algidx])
			end
		end
	end
	df = DataFrame(algorithm = dfalgs, objective = dfobjs, value = dfvals)
	df.algorithm = categorical(df.algorithm, levels=ordered_algs, ordered=true)
	df.objective = categorical(df.objective, levels=objectives, ordered=true)
	p = @df df StatsPlots.groupedboxplot(:algorithm, :value, group = :objective, xlabel="Algorithm", 
									ylabel="Rank", legendtitle = "Objective", xrotation=90, yrotation=90, whisker_width=0, outliers=false, whiskercolor=:transparent, whiskerlinewidth=0,
									markersize=1, legend=false, legend_column=-1, size=(1200, 700), left_margin=12Plots.mm, bottom_margin=20Plots.mm, dpi=600)
	Plots.savefig(p, "tuned_rankbox.png")
	display(p)

        
        h5open("tuned_rankbox.h5", "w") do h5
            gd = @groupby(df, :algorithm)
            names = [string(gd[i][1, :algorithm]) for i in 1:gd.ngroups]
            alpha_order_idx = sortperm(names)
            names           = names[alpha_order_idx]
            for (pos_offset, obj) in zip([-0.3, -0.1, 0.1, 0.3], objectives)
                sub    = @subset(df, :objective .== obj)
                subgd  = @groupby(sub, :algorithm)
                values = [subgd[i][:, :value] for i in 1:gd.ngroups]
                values = values[alpha_order_idx]
                for (name, value) in zip(names, values)
                    write(h5, "$(name)_$(obj)", value)
                end
                write(h5, "position_$(obj)", alpha_order_idx .+ pos_offset)
            end
            write(h5, "position_labels", alpha_order_idx)
            write(h5, "labels", names)
        end

	println("waiting for key press + enter")
	readline()

	# plot CDFs
        p = plot()
        h5open("smallcdfs.h5", "w") do h5
	    clrs = palette(:glasbey_category10_n256, length(algs))
	    clrs = palette(:default)
            write(h5, "x", collect(1:length(algs)))
	    for i=8:length(algs)
		plot!(1:length(algs), cumsum(ordered_probs[i,:]), seriestype=:steppost, label=false, color=:silver, xlabel="Rank", ylabel="Cumulative Probability", legend=:outerright, legendfontsize=5, dpi=600)
                write(h5, ordered_algs[i], cumsum(ordered_probs[i,:]))
	    end
	    for i=1:7
		plot!(1:length(algs), cumsum(ordered_probs[i,:]), seriestype=:steppost, label=ordered_algs[i], color=clrs[i], xlabel="Rank", ylabel="Cumulative Probability", legend=:outerright, legendfontsize=5, dpi=600)
                write(h5, ordered_algs[i], cumsum(ordered_probs[i,:]))
	    end
        end
	Plots.savefig(p, "smallcdfs.png")
	display(p)
	println("waiting for key press + enter")
	readline()

	# reorder for wins instead
	winsum = vec(sum(wins, dims=2))
	ordering = sortperm(winsum, rev=true)
	ordered_wins = wins[ordering, ordering]
	ordered_algs = algs[ordering]

        h5open("smallmatrix_wins.h5", "w") do h5
            write(h5, "heatmap",          Array(ordered_wins[end:-1:1,:]'))
            write(h5, "ordered_algs_rev", ordered_algs[end:-1:1])
            write(h5, "ordered_algs",     ordered_algs)
        end

	p = Plots.heatmap(ordered_wins[end:-1:1,:], yticks=(1:length(algs), ordered_algs[end:-1:1]), xticks=(1:length(algs), ordered_algs), xrotation=45, ylabel="Algorithm", xlabel="Algorithm", colorbar_title = "Probability Row i Bests Col j", dpi=1200, size=(1000,800), left_margin=20Plots.mm, bottom_margin=20Plots.mm, tickfontsize=7)
	Plots.savefig(p, "smallmatrix_wins.png")
	display(p)
	println("waiting for key press + enter")
	readline()
end


main()


