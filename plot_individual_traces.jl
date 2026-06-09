include("utils.jl")
using Plots, StatsPlots, StatsBase, DataFrames, CategoricalArrays, HDF5
gr()
#plotlyjs()

function main()
	tuning = load("tuning.jld2")
	tuned_stepsizes = tuning["ordered_beststepsizes"]
	algs = tuning["ordered_algs"]

        println(algs)

	# objective functions / gradient estimates
	# a few selected from the original full set
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI", 
			"MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]
	targets = ["bones_data-bones_model", "sat-hier_2pl_subsampled", "diamonds-diamonds", "dogs-dogs_subsampled",
			"science_irt-grsm_latent_reg_irt_subsampled", "pilots-pilots", "mesquite-logmesquite_subsampled", "M0_data-M0_model", "M0_data-M0_model_subsampled"]
	
	# # all of the challenge problems
	# base_objectives =  ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"] 
	# base_targets = ["normnmf_liver", "normnmf_lung", "normnmf_ovary", "normnmf_breast", "normnmf_skin", "normnmf_stomach", "sparsepoiss_crime", "sparselog_prostate", "sparselog_ovarian", "sparselog_leukemia", "sirnb_sirnb", "twocptpop_twocptpop"]
	# append!(base_targets, [t*"_subsampled" for t in base_targets])
	# targets = []
	# objectives = []
	# for ob in base_objectives
	# 	for t in base_targets
	# 		push!(targets, t)
	# 		push!(objectives, ob)
	# 	end
	# end

        h5open("problem_trace.h5", "w") do h5
	    for (objective, target) in zip(objectives, targets)
		to_plot = ["Ensemble", "SAA", "AdaGrad", "SLS", "HyperAdam", "DAdaptAdam"]

		results_algs = []
		results = []
		calibration_counts = Vector{Float64}()
		init_vals = Vector{Float64}()
		final_vals = Vector{Float64}()
		mint = Inf
		maxt = -Inf
		minint = Inf
		minidx = Inf
		for a in 1:length(algs)
			fn = "results/"*algs[a]*"-"*target*"-"*objective*"-"*tuned_stepsizes[a]*"-1.jld2"
                    @info("", target, algs[a], isfile(fn))

			if isfile(fn)
				push!(results, load(fn))
				push!(results_algs, algs[a])
				push!(calibration_counts, results[end]["calibration_count"])
				@assert results[end]["ts"][1] == 0.0
				if results[end]["ts"][2] < mint
					mint = results[end]["ts"][2]
				end
				if results[end]["ts"][2] < mint
					mint = results[end]["ts"][2]
				end
				if results[end]["nfaileval"][1] == 0
					push!(init_vals, results[end]["objs"][1])
				end
				if results[end]["nfaileval"][end] == 0
					push!(final_vals, results[end]["objs"][end])
				end
				if sum(results[end]["nfaileval"]) == 0
					val, _ = integrate(results[end]["ts"], results[end]["objs"], results[end]["objvars"], 0.0, 360.0)
					if val < minint
						minint = val
						minidx = length(results)
					end
				end
			end
		end
		# this target did not have any output
		if length(calibration_counts) == 0 || all(isnan.(init_vals)) || all(isnan.(final_vals))
			continue
		end
		calibration_median = median(calibration_counts)
		maxy = quantile(init_vals[.!isnan.(init_vals)], 0.9)
		miny = quantile(final_vals[.!isnan.(final_vals)], 0.1)


		to_plot_idcs = [findfirst(==(x),results_algs) for x in to_plot]

		if objective == "MAP"
			ylabel = "Negative Log Target"
			yscale = :identity
			ylimits = (miny-0.05*(maxy-miny), maxy+0.05*(maxy-miny))
			yticks = :nothing
		elseif objective == "VAEVI"
			ylabel = "Squared Gradient Norm"
			yscale = :log10
			ylimits = :nothing
			yticks = 10.0 .^ (-100:2:100)
		else
			ylabel = "Negative ELBO"
			yscale = :identity
			ylimits = (miny-0.05*(maxy-miny), maxy+0.05*(maxy-miny))
			yticks = :nothing
		end
		
		#clrs = palette(:glasbey_category10_n256, length(algs))

		# plot all first in grey with thin lines
		clrs = palette(:default)
		p = plot()
		for a in 1:length(results_algs)
			res = results[a]
			res["ts"][1] = mint/10 # remove t=0 point and just use a lower bound on the first recorded time
			ts = res["ts"]*res["calibration_count"]/calibration_median
			ys = objective == "VAEVI" ? res["gradsqs"] : res["objs"] # plot gradsqs for VAEs, ELBO for diag/full gauss, objective for MAP
			ys = Float64.(ys[res["nfaileval"] .== 0])
			ts = ts[res["nfaileval"] .== 0]
			plot!(p, ts, ys,
							color=:silver, label=false, xlabel="Time (s)", ylabel=ylabel, xscale=:log10, yscale=yscale,
							ylimits=ylimits,
							xticks=10.0 .^(-7:7), yticks = yticks,
							linewidth=0.5, legend=false, dpi=600)
                    write(h5, "$(target)_$(objective)_$(results_algs[a])_time", ts)
                    write(h5, "$(target)_$(objective)_$(results_algs[a])_value", ys)
		end
		# plot specific choices
		for idx in 1:length(to_plot_idcs)
			a = to_plot_idcs[idx]
			if a != nothing
				res = results[a]
				res["ts"][1] = mint/10 # remove t=0 point and just use a lower bound on the first recorded time
				ts = res["ts"]*res["calibration_count"]/calibration_median
				ys = objective == "VAEVI" ? res["gradsqs"] : res["objs"] # plot gradsqs for VAEs, ELBO for diag/full gauss, objective for MAP
				ys = ys[res["nfaileval"] .== 0]
				ts = ts[res["nfaileval"] .== 0]
				plot!(p, ts, ys,
								label=results_algs[a], color=clrs[idx], xlabel="Time (s)", ylabel=ylabel, xscale=:log10, yscale=yscale,
								ylimits=ylimits,
								xticks=10.0 .^(-7:7), yticks = yticks,
								legend=:topright, dpi=600)
                            #write(h5, "$(target)_$(objective)_$(results_algs[a])_time", ts)
                            #write(h5, "$(target)_$(objective)_$(results_algs[a])_value", ys)
			end
		end

		Plots.savefig(p, "elbo_"*target*"_"*objective*".png")
	end
    end
end

main()
