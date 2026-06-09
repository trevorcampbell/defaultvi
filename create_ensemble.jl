include("utils.jl")
using StatsBase

function main()
	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# which objective function are we using to define the ensemble?
	# either "obj" for ELBO or "grad" for grad sq norm
	ensemble_type = "obj" # "grad"

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	ensemble = [("Adam", "1e-3"), ("Adam", "1e-4"), ("DoWG", "1e+0"), ("Lion", "1e-5"), ("SAALBFGS", "1e-8")]
	for ob in 1:length(objectives)
		for t in 1:length(targets)
			nevalsamps = Vector{Float64}()
			nfaileval = Vector{Float64}()
			niters = Vector{Float64}()
			ts = Vector{Float64}()
			objs = Vector{Float64}()
			objvars = Vector{Float64}()
			gradsqs = Vector{Float64}()
			gradvars = Vector{Float64}()
			ngs = Vector{Float64}()
			nfs = Vector{Float64}()
			calibration_count = Inf
			calibration_fails = Inf
			nfail = Inf
			stepsz = 0.0
			# first get median calibration count
			calib_counts = Vector{Int64}()
			calib_fails = Vector{Int64}()
			onefound = false
			for en in 1:length(ensemble)
				alg = ensemble[en][1]
				stp = ensemble[en][2]
				fn = "results/"*alg*"-"*targets[t]*"-"*objectives[ob]*"-"*stp*"-1.jld2"
				if isfile(fn)
					onefound = true
					res = load(fn)
					push!(calib_counts, res["calibration_count"])
					push!(calib_fails, res["calibration_fails"])
				end
			end
			# if none of the algorithms produced output for this problem, skip
			if !onefound
                @warn("failed!", targets[t], objectives[ob])
				continue
			end
			calibration_count = median(calib_counts)
			@assert all(calib_fails .== 0)
			calibration_fails = 0

			# now take the ensemble minimum 
			for en in 1:length(ensemble)
				alg = ensemble[en][1]
				stp = ensemble[en][2]
				fn = "results/"*alg*"-"*targets[t]*"-"*objectives[ob]*"-"*stp*"-1.jld2"
				#println("Processing $fn")
				if isfile(fn)
					res = load(fn)
					# take the minimum number of failed steps over all methods
					nfail = res["nfail"] < nfail ? res["nfail"] : nfail
					# take the pointwise minimum of the current best and next ensemble method
					if length(ts) == 0
						# for the first method, just copy
						nevalsamps = res["nevalsamps"]
						nfaileval = res["nfaileval"]
						ts = res["ts"]*res["calibration_count"]/calibration_count
						objs = res["objs"]
						objvars = res["objvars"]
						gradsqs = res["gradsqs"]
						gradvars = res["gradvars"]
						niters = res["niters"]*res["calibration_count"]/calibration_count
						ngs = res["ngs"]*res["calibration_count"]/calibration_count
						nfs = res["nfs"]*res["calibration_count"]/calibration_count
					else
						# for subsequent methods, take the pointwise minimum by expanding both to have the same time points via linear interpolation and then minimizing

						# load the new ensemble method
						_nevalsamps = res["nevalsamps"]
						_nfaileval = res["nfaileval"]
						_ts = res["ts"]*res["calibration_count"]/calibration_count
						_objs = res["objs"]
						_objvars = res["objvars"]
						_gradsqs = res["gradsqs"]
						_gradvars = res["gradvars"]
						_niters = res["niters"]*res["calibration_count"]/calibration_count
						_ngs = res["ngs"]*res["calibration_count"]/calibration_count
						_nfs = res["nfs"]*res["calibration_count"]/calibration_count

						# expand the times of both the current ensemble and new method
						ts_new, objs = augment_interpolate(ts, objs, _ts)
						_, objvars = augment_interpolate(ts, objvars, _ts)
						_, gradsqs = augment_interpolate(ts, gradsqs, _ts)
						_, gradvars = augment_interpolate(ts, gradvars, _ts)
						_, niters = augment_interpolate(ts, niters, _ts)
						_, ngs = augment_interpolate(ts, ngs, _ts)
						_, nfs = augment_interpolate(ts, nfs, _ts)
						_, nevalsamps = augment_interpolate(ts, nevalsamps, _ts)
						_, nfaileval = augment_interpolate(ts, nfaileval, _ts)

						_ts_new, _objs = augment_interpolate(_ts, _objs, ts)
						_, _objvars = augment_interpolate(_ts, _objvars, ts)
						_, _gradsqs = augment_interpolate(_ts, _gradsqs, ts)
						_, _gradvars = augment_interpolate(_ts, _gradvars, ts)
						_, _niters = augment_interpolate(_ts, _niters, ts)
						_, _ngs = augment_interpolate(_ts, _ngs, ts)
						_, _nfs = augment_interpolate(_ts, _nfs, ts)
						_, _nevalsamps = augment_interpolate(_ts, _nevalsamps, ts)
						_, _nfaileval = augment_interpolate(_ts, _nfaileval, ts)

						ts = ts_new
						_ts = _ts_new

						# take the pointwise minimum
						@assert length(ts) == length(_ts)
						@assert sum(abs.(ts-_ts)) < 1e-12
						@assert ensemble_type in ["obj", "grad"]
						for i=1:length(ts)
							if (ensemble_type == "obj" && (_objs[i] <= objs[i] || isnan(objs[i]))) ||
							   (ensemble_type == "grad" && (_gradsqs[i] <= gradsqs[i] || isnan(gradsqs[i])))
								objs[i] = _objs[i]
								objvars[i] = _objvars[i]
								gradsqs[i] = _gradsqs[i]
								gradvars[i] = _gradvars[i]
								niters[i] = _niters[i]
								ngs[i] = _ngs[i]
								nfs[i] = _nfs[i]
								nevalsamps[i] = _nevalsamps[i]
								nfaileval[i] = _nfaileval[i]
							end
						end
					end
				end
			end
            @info("success", targets[t], objectives[ob])
			stepsizes = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e+0"]
			for stepsz in stepsizes
				efn = "results/Ensemble-"*targets[t]*"-"*objectives[ob]*"-"*stepsz*"-1.jld2"
				jldsave(efn; stepsz, nfail, nfaileval, nfs, ngs, niters, ts, objs, objvars, gradsqs, gradvars, nevalsamps, calibration_count, calibration_fails)
			end
		end
	end
end


main()


