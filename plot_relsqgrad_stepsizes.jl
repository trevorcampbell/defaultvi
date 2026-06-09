include("utils.jl")
using Plots, StatsPlots, StatsBase, CategoricalArrays, DataFrames, DataFramesMeta, HDF5
gr()

function main()
	algs = ["Adam","AdamAvg","AdaGrad","AMSGrad","AdaSLS","AdamSLS","BigBatch","COCOB","DAdaptAdam","DAdaptSGD","DoG","DoWG","DoGMom","DoWGMom","FTRL","Lion","Pflug","PoNoS","SAA","SAANA","SAALBFGS","SAACG","SGD","SLS"]
	append!(algs, ["Mecha"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["RABVI"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["SRABVI"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["Hyper"*T for T in ["Adam", "AdamAvg", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])

	alg = "SAALBFGS"
	stepsizes = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e+0"]

	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

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

	dfobjnms = []
	dfgradsqs = []
	dfsteps = []
	dfobjs = []
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			gradsqs = Vector{Float64}()
			objs = Vector{Float64}()
			for s in 1:length(stepsizes)
				fn = "results/"*alg*"-"*targets[t]*"-"*objectives[ob]*"-"*stepsizes[s]*"-1.jld2"
				push!(dfsteps, stepsizes[s])
				push!(dfobjnms, objectives[ob])
				if isfile(fn)
					res = load(fn)
					# hard failures (we can prove it increased the objective)
					if res["nfaileval"][end] > 0 || res["objs"][end] - sqrt(res["objvars"][end]) >= res["objs"][1] + sqrt(res["objvars"][1])
					# soft failures (we can't prove it decreased the objective)
					#if res["nfaileval"][end] > 0 || res["objs"][end] + sqrt(res["objvars"][end]) >= res["objs"][1] - sqrt(res["objvars"][1])
						push!(gradsqs, Inf)
						push!(objs, Inf)
					else
						push!(gradsqs, res["gradsqs"][end]) # res[objname][1])
						push!(objs, res["objs"][end] - bestobjs[ob,t]) #baseline_obj) # res[objname][1])
					end
				else
					push!(objs, Inf)
					push!(gradsqs, Inf)
				end
			end
			gradsqs /= minimum(gradsqs)
			gradsqs[isnan.(gradsqs) .|| (gradsqs .< 1)] .= 1.0 #no improvement
			gradsqs[gradsqs .> 1e20] .= 1e20
			objs /= minimum(objs)
			objs[isnan.(objs) .|| (objs .< 1)] .= 1.0 # no improvement
			objs[objs .> 1e20] .= 1e20
			append!(dfgradsqs, gradsqs)
			append!(dfobjs, objs)
		end
	end

	df = DataFrame(stepsize = dfsteps, objective = dfobjnms, gradsq = dfgradsqs, obj = dfobjs)
	df.objective = categorical(df.objective, levels=objectives, ordered=true)

	# log10-transform the gradient relative values before plotting. A box plot of "value" plotted on log10 scale != a box plot of "log 10 value" plotted normally, and the usual interpretation should prefer the latter
	df.gradsq = log10.(df.gradsq)

        h5open("gradsq_rel_$(alg).h5", "w") do h5
            gd = @groupby(df, :stepsize)
            stepsizes = [gd[i][1, :stepsize] for i in 1:gd.ngroups]
            labels    = string.(stepsizes)
            for (pos_offset, obj) in zip([-0.3, -0.1, 0.1, 0.3], objectives)
                sub    = @subset(df, :objective .== obj)
                subgd  = @groupby(sub, :stepsize)
                values = [subgd[i][:, :gradsq] for i in 1:gd.ngroups]
                for (stepsize, value) in zip(dfsteps, values)
                    write(h5, "$(obj)_$(stepsize)", value)
                end
                write(h5, "position_$(obj)", collect((1:length(stepsizes)) .+ pos_offset))
            end
            write(h5, "position_labels", collect(1:length(stepsizes)))
            write(h5, "labels", labels)
        end

	p = @df df StatsPlots.groupedboxplot(:stepsize, :gradsq, group = :objective, xlabel="Step Size", ylabel="Log₁₀ Relative Squared Gradient", legendtitle = "Objective", xrotation=45, markersize=1, 
										 legend=false, size=(1600, 400), left_margin=12Plots.mm, bottom_margin=20Plots.mm, yticks=-20:2:20) #10.0 .^(-20:2:10))
	hline!(p, [0.0], linestyle=:dash, color=:gray, label=nothing)
	Plots.savefig(p, "gradsq_rel_"*alg*".png")
	display(p)
	println("waiting")
	readline()

end


main()


