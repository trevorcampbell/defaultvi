include("utils.jl")
using Plots, StatsPlots, StatsBase #, DataFrames, CategoricalArrays
using HDF5
gr()

function main()
	# algorithms that use one gradient eval per iteration and do not much else
	algs = ["SGD", "Adam", "AdamAvg", "DoG", "DoWG", "AMSGrad", "AdaGrad"]
	stepsizes = ["1e-4", "1e-4", "1e-4", "1e+0", "1e+0", "1e-3", "1e-2"]

	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)
	# remove the non-posteriorDB targets
	remove_nonpdb!(targets)
	
	objective_dimensions = zeros(length(objectives), length(targets))
	dimensions = zeros(length(targets))
	for t in 1:length(targets)
		d = 0
		redirect_stderr(devnull) do
			post = PosteriorDB.posterior(pdb, targets[t])
			prb = StanProblem(post, "stan")
			d = LogDensityProblems.dimension(prb)
			if occursin("_subsampled", targets[t])
				d -= 1
			end
		end
		dimensions[t] = d
		objective_dimensions[1, t] = d # MAP
		objective_dimensions[2, t] = 2d # Diag VI
		objective_dimensions[3, t] = d^2 + 3d/2 # full VI
		nlayers = 10
		dint = 10
		objective_dimensions[4, t] = 2*(nlayers*(4d*dint + 2d) + 4d*d+2d)
	end

	# first find optimal objective across all algs/tuning params/iterations
	itercts = zeros(length(objectives), length(targets))
	p = Plots.plot()
	q = Plots.plot()
	q2 = Plots.plot()

        h5open("iteration_scatter.h5", "w") do h5
	for ob in 1:length(objectives)
		objective = objectives[ob]
		for t in 1:length(targets)
			target = targets[t]
			found_file = false
			cts = zeros(Int64, length(algs))
			for a in 1:length(algs)
				fn = "results/"*algs[a]*"-"*target*"-"*objective*"-"*stepsizes[a]*"-1.jld2"
				if isfile(fn)
					found_file = true
					res = load(fn)
					cts[a] = res["niters"][end]
				end
			end
			itercts[ob, t] = !found_file ? -1 : median(cts)
		end
		row_itercts = itercts[ob, itercts[ob, :] .> 0]
		row_dims = dimensions[itercts[ob, :] .> 0]
		row_objdims = objective_dimensions[ob, itercts[ob, :] .> 0]

                write(h5, "$(objective)_x",           row_itercts)
                write(h5, "$(objective)_targetdim_y", row_dims)
                write(h5, "$(objective)_objdim_y",    row_objdims)

		Plots.scatter!(q, row_itercts, row_dims, xlabel="Iterations", ylabel="Target Dimension", xticks = 10 .^(3:8), xscale=:log10, yscale=:log10, label=objective, alpha=0.6)
		Plots.scatter!(q2, row_itercts, row_objdims, xlabel="Iterations", ylabel="Objective Dimension", xticks = 10 .^(3:8), yticks=10 .^(0:7), xscale=:log10, yscale=:log10, label=objective, alpha=0.6)
	end
        end

	Plots.savefig(q, "iteration_scatter_grouped.png")
	display(q)
	println("waiting for key press+enter to continue...")
	readline()

	Plots.savefig(q2, "iteration_scatter_obj_grouped.png")
	display(q2)
	println("waiting for key press+enter to continue...")
	println("waiting")
	readline()

end


main()


