include("utils.jl")
using Plots, StatsPlots, DataFrames, CategoricalArrays
using StableRNGs, HDF5, DataFramesMeta, StatsBase

function main()
        rng = StableRNG(123)
	d = JLD2.load("problem_characteristics.jld2")
	targets = Vector{String}(d["targets"])
	condnums = Vector{Float64}(d["condnums"])
	dimensions = Vector{Float64}(d["dimensions"])
	noises = Vector{Float64}(d["noises"])
	noisesubs = Vector{Float64}(d["noisesubs"])
	noisesubs[noisesubs .== 0.0] = noises[noisesubs .== 0.0]

        x = dimensions .*(1 .+ 0.3*(rand(rng, length(dimensions)).-0.5))
        y = condnums.*(1 .+ 0.3*(rand(rng, length(dimensions)).-0.5))
        z = log10.(noisesubs./noises)

        h5open("problem_scatter.h5", "w") do h5
            write(h5, "dimensions", x)
            write(h5, "condnumber", y)
            write(h5, "noise",      z)
        end

	p = scatter(x, y, zcolor=z, xscale=:log10, yscale=:log10, yticks = 10 .^(0:8), xticks=10 .^(0:4), legend=false, xlabel="Dimension (15% Jitter)", ylabel="Condition Number (15% Jitter)", colorbar_title = "Log₁₀ Noise Ratio", colorbar=true, markersize=4, dpi=600)#, markerstrokewidth=0.1)
	Plots.savefig(p, "problem_scatter.png")
	display(p)
	println("Waiting for keypress before continuing...")
	readline()

	# duplicate dimensions for those entries that have noisesub > 0 (those will have two versions in the big experiment, one noisy, one not)
	dimensions = vcat(dimensions, dimensions[noisesubs .== noises])
	# construct a dataframe to enable plotting of stacked histogram
	nlayers = 10
	dint = 10
	dims = vcat(log10.(dimensions), log10.(2*dimensions), log10.(dimensions.*(1 .+ dimensions)), log10.(2*(nlayers*(4*dint*dimensions + 2*dimensions) + 4*dimensions.^2 + 2*dimensions)))
	objs = vcat(repeat(["MAP"], length(dimensions)), repeat(["DiagGaussianVI"], length(dimensions)), repeat(["FullGaussianVI"], length(dimensions)), repeat(["VAEVI"], length(dimensions)))
	objs = categorical(objs, levels=["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"], ordered=true)
	df = DataFrame(dimension = dims, objective = objs)

       
        h5open("problem_hist.h5", "w") do h5
            bins = 0:.5:10
            hists = Histogram[]
            for obj in ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]
                sub      = @subset(df, :objective .== obj)
                dims     = sub.dimension
                hist     = fit(Histogram, dims, bins)
                push!(hists, hist)
            end

            write(h5, "hist_x", collect(bins[1:end-1]))

            y = zeros(Int, length(first(hists).weights))
            for (hist, obj) in zip(hists, ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"])
                for (idx, weight) in enumerate(hist.weights)
                    y[idx] += weight
                end
                write(h5, "$(obj)_hist_y", y)
            end
        end

	p = @df df StatsPlots.groupedhist(:dimension, group=:objective, bar_position=:stack, bins=18, xlabel="Log₁₀ Dimension", ylabel="Count", dpi=600)
	Plots.savefig(p, "problem_histogram.png")
	display(p)
	println("Waiting for keypress before continuing...")
	readline()

end


main()


