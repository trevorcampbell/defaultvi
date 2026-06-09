# Streaming RABVI without the stopping rule
# Based on: Welandawe et al, "A Framework for Improving the Reliability of Black-box Variational Inference"
# JMLR 25(219):1−71, 2024. https://arxiv.org/abs/2203.15945
#
# This version replaces the full iterate trace with geometric batch statistics,
# achieving O(log(N/Wmin) * D) memory instead of O(N * D).
# Uses batch-means ESS (Liu, Vats, Flegal 2022) and basic split-chain R-hat.
# Chan/Pebay parallel variance algorithm for combining batch statistics.
#
# Step-size reduction: γ₁ = γ² (quadratic), motivated by bias=O(γ), noise=O(√γ).
# Base batch size grows as max(Wmin, 10*floor(n^{1/3})) for BM-ESS consistency.

struct BatchStats
	mean::Vector{Float64}   # D-dim batch mean
	var::Vector{Float64}    # D-dim batch variance (n-1 denominator)
	count::Int64            # number of samples in this batch
	level::Int64            # merge level: 0 = raw, 1 = merged pair, etc.
end

# Chan/Pebay parallel merge of two batches
function merge_batches(a::BatchStats, b::BatchStats)
	n = a.count + b.count
	δ = b.mean .- a.mean
	μ = (a.count .* a.mean .+ b.count .* b.mean) ./ n
	M2 = (a.count - 1) .* a.var .+ (b.count - 1) .* b.var .+ (a.count * b.count / n) .* δ .^ 2
	return BatchStats(μ, M2 ./ (n - 1), n, max(a.level, b.level) + 1)
end

# Combine multiple batches into a single aggregate
function combine_batch_stats(batches::AbstractVector{BatchStats})
	result = batches[1]
	for i in 2:length(batches)
		result = merge_batches(result, batches[i])
	end
	return result
end

mutable struct StreamingRABVI{Alg}
	alg::Alg
	iter::Int64
	prevouteriter::Int64
	nf::Int64
	ng::Int64
	opt_time::Float64
	Wmin::Int64
	ESSmin::Float64
	ϵ::Float64
	kconv::Int64
	Wcheck::Int64
	x::Vector{Float64}
	# Streaming batch storage (replaces xtrace)
	batches::Vector{BatchStats}
	cur_mean::Vector{Float64}    # Welford running mean for current batch
	cur_M2::Vector{Float64}      # Welford running M2 for current batch
	cur_count::Int64             # count within current batch
	# Adaptive batch sizing
	cur_batch_size::Int64        # current level-0 batch size (grows within a phase)
	n_phase::Int64               # total samples added in current phase
	batch_just_finalized::Bool   # flag for triggering convergence checks
	# Iterate averaging: warm-start Polyak with weight from previous phase's window
	x_prev::Vector{Float64}     # converged estimate from previous phase
	n_prev::Int64                # window size used to compute x_prev
	γ_prev::Float64             # step size of the phase that produced x_prev
	x_phase_sum::Vector{Float64} # running sum of iterates in current phase
	x_phase_count::Int64         # count of iterates in current phase
end

function StreamingRABVI{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg}
	D = length(x)
	StreamingRABVI{Alg}(
		Alg(copy(x), stepsz, seed),
		0, 0, 0, 0, 0.0,       # iter, prevouteriter, num_fn_evals, num_grad_evals, opt_time
		200,                     # Wmin
		50.0,                    # ESSmin
		0.1,                     # ϵ
		-1, -1,                  # kconv, Wcheck
		copy(x),                 # x
		BatchStats[],            # batches
		zeros(D),                # cur_mean
		zeros(D),                # cur_M2
		0,                       # cur_count
		200,                     # cur_batch_size (= Wmin initially)
		0,                       # n_phase
		false,                   # batch_just_finalized
		copy(x),                 # x_prev (init to starting point)
		0,                       # n_prev (no prior weight initially)
		1.0,                     # γ_prev (init to stepsz; irrelevant when n_prev=0)
		zeros(D),                # x_phase_sum
		0                        # x_phase_count
	)
end

StreamingRABVI{Alg}(dimension::Int64, stepsz, seed) where {Alg} = StreamingRABVI{Alg}(zeros(dimension), stepsz, seed)

# Enforce the geometric merge invariant: at most 2 batches per level.
function enforce_merge_invariant!(rab::StreamingRABVI)
	level = 0
	while true
		indices = findall(b -> b.level == level, rab.batches)
		if length(indices) <= 2
			break
		end
		i, j = indices[1], indices[2]
		merged = merge_batches(rab.batches[i], rab.batches[j])
		rab.batches[i] = merged
		deleteat!(rab.batches, j)
		level += 1
	end
end

# Add a single sample to the streaming accumulator (Welford online update)
function add_sample!(rab::StreamingRABVI, x::Vector{Float64})
	rab.cur_count += 1
	rab.n_phase += 1
	δ = x .- rab.cur_mean
	rab.cur_mean .+= δ ./ rab.cur_count
	δ2 = x .- rab.cur_mean
	rab.cur_M2 .+= δ .* δ2

	if rab.cur_count == rab.cur_batch_size
		finalize_batch!(rab)
	end
end

# Complete the current batch and add it to storage
function finalize_batch!(rab::StreamingRABVI)
	var = rab.cur_M2 ./ (rab.cur_count - 1)
	push!(rab.batches, BatchStats(copy(rab.cur_mean), var, rab.cur_count, 0))
	rab.cur_mean .= 0.0
	rab.cur_M2 .= 0.0
	rab.cur_count = 0
	enforce_merge_invariant!(rab)

	# Update batch size: max(Wmin, 10*floor(n_phase^{1/3}))
	rab.cur_batch_size = max(rab.Wmin, 10 * floor(Int64, rab.n_phase^(1/3)))

	rab.batch_just_finalized = true
end

# Weighted mean over selected batches
function window_mean(batches::AbstractVector{BatchStats})
	N = sum(b.count for b in batches)
	return sum(b.count .* b.mean for b in batches) ./ N
end

# Split-chain R-hat from batch statistics.
function streaming_rhat(batches::AbstractVector{BatchStats})
	total = sum(b.count for b in batches)
	half_target = total ÷ 2

	cumsum = 0
	split_idx = 0
	for i in 1:length(batches)
		cumsum += batches[i].count
		if cumsum >= half_target
			split_idx = i
			break
		end
	end
	split_idx = clamp(split_idx, 1, length(batches) - 1)

	first_half = @view batches[1:split_idx]
	second_half = @view batches[split_idx+1:end]

	stats1 = combine_batch_stats(first_half)
	stats2 = combine_batch_stats(second_half)

	n_avg = (stats1.count + stats2.count) / 2

	W = (stats1.var .+ stats2.var) ./ 2

	# Between-chain variance, corrected for unequal chain lengths
	n1 = stats1.count
	n2 = stats2.count
	Δ = stats1.mean .- stats2.mean
	B_over_n = (2.0 * n1 * n2 / (n1 + n2)^2) .* Δ .^ 2

	var_hat = ((n_avg - 1) / n_avg) .* W .+ B_over_n
	rhat_vec = similar(W)
	for i in eachindex(W)
		if W[i] <= 0.0
			rhat_vec[i] = var_hat[i] <= 0.0 ? 1.0 : Inf
		else
			rhat_vec[i] = sqrt(var_hat[i] / W[i])
		end
	end
	return maximum(rhat_vec)
end

# Batch-means ESS with unequal batch sizes.
function streaming_ess(batches::AbstractVector{BatchStats})
	K = length(batches)
	N = sum(b.count for b in batches)

	grand_mean = sum(b.count .* b.mean for b in batches) ./ N
	sigma2_inf = sum(b.count .* (b.mean .- grand_mean) .^ 2 for b in batches) ./ (K - 1)

	overall = combine_batch_stats(batches)
	sigma2 = overall.var

	ess = similar(sigma2)
	for i in eachindex(ess)
		if sigma2_inf[i] <= 0.0 || sigma2[i] <= 0.0
			ess[i] = Float64(N)
		else
			ess[i] = clamp(N * sigma2[i] / sigma2_inf[i], 1.0, Float64(N))
		end
	end
	return ess
end

# Compute mean, MCSE, and ESS from batch statistics
function streaming_mcse_ess(batches::AbstractVector{BatchStats})
	overall = combine_batch_stats(batches)
	sd = sqrt.(max.(overall.var, 0.0))
	effs = streaming_ess(batches)
	mcse = sd ./ sqrt.(effs)
	return overall.mean, mcse, effs
end

# Find the best window (subset of trailing batches) for R-hat
function r_hat_windowed_streaming(batches::Vector{BatchStats})
	num_batches = length(batches)
	min_b = min(3, num_batches)
	max_b = num_batches

	candidate_sizes = unique(Int64.(floor.(range(min_b, max_b, length=5))))

	best_rhat = Inf
	best_w = min_b
	for w in candidate_sizes
		selected = @view batches[num_batches-w+1:num_batches]
		rh = streaming_rhat(selected)
		if rh < best_rhat
			best_rhat = rh
			best_w = w
		end
	end
	return best_rhat, best_w
end

# Select trailing batches covering at least target_count samples
function select_batches_by_count(batches::Vector{BatchStats}, target_count::Int64)
	cumsum = 0
	for i in length(batches):-1:1
		cumsum += batches[i].count
		if cumsum >= target_count
			return i:length(batches)
		end
	end
	return 1:length(batches)
end

# Update rab.x as a warm-start Polyak average:
# The previous phase's estimate is weighted by n_prev/γ_prev (the step size
# that produced x_prev), reflecting how many effective samples it represents.
# x = (w_prev * x_prev + x_phase_count * x_phase_mean) / (w_prev + x_phase_count)
function update_iterate_average!(rab::StreamingRABVI)
	w_prev = rab.n_prev / rab.γ_prev
	total_weight = w_prev + rab.x_phase_count
	if total_weight > 0 && rab.x_phase_count > 0
		x_phase_mean = rab.x_phase_sum ./ rab.x_phase_count
		rab.x .= (w_prev .* rab.x_prev .+ rab.x_phase_count .* x_phase_mean) ./ total_weight
	end
end

function step!(rab::StreamingRABVI{Alg}, obj) where {Alg}
	rab.iter += 1

	# inner iteration number for the current FASO phase
	k = rab.iter - rab.prevouteriter

	# take an optimization step with timing
	t0 = time_ns() / 1e9
	step!(rab.alg, obj)
	rab.opt_time += time_ns() / 1e9 - t0

	# streaming accumulation
	rab.batch_just_finalized = false
	add_sample!(rab, rab.alg.x)

	# Iterate averaging: before R-hat convergence, use warm-start Polyak;
	# after R-hat convergence, use only iterates from the converged window onward
	rab.x_phase_sum .+= rab.alg.x
	rab.x_phase_count += 1
	if rab.kconv < 0
		# Pre-convergence: blend previous phase estimate with current phase mean
		update_iterate_average!(rab)
	else
		# Post-convergence: use only the converged-window iterates
		# x_phase_sum/count were reset at R-hat convergence to start fresh
		if rab.x_phase_count > 0
			rab.x .= rab.x_phase_sum ./ rab.x_phase_count
		end
	end

	# R-hat convergence check: fires when a batch was just finalized
	num_batches = length(rab.batches)
	if rab.kconv < 0 && rab.batch_just_finalized && num_batches >= 3
		Rhatbest, Wbest = r_hat_windowed_streaming(rab.batches)

		# Overwrite rab.x with best window average
		selected = @view rab.batches[num_batches-Wbest+1:num_batches]
		rab.x = window_mean(selected)

		if Rhatbest <= 1.1
			total_samples = sum(b.count for b in selected)
			rab.Wcheck = total_samples
			rab.kconv = k - total_samples
			# Reset phase accumulation to start fresh from the converged window
			rab.x_phase_sum .= 0.0
			rab.x_phase_count = 0
		end
	end

	# MCSE convergence check
	if rab.kconv >= 0 && k - rab.kconv >= rab.Wcheck
		selected = select_batches_by_count(rab.batches, Int64(rab.Wcheck))

		t0 = time_ns() / 1e9
		converged_mean, converged_se, converged_effs = streaming_mcse_ess(@view rab.batches[selected])
		mcse_time = time_ns() / 1e9 - t0

		rab.x = converged_mean

		# update Wcheck based on computation time ratios
		rel_opt_time = rab.opt_time / rab.iter
		n_selected = length(selected)
		rel_mcse_time = mcse_time / n_selected
		rel_time_ratio = rel_opt_time / rel_mcse_time
		recheck_scale = max(1.05, 1 + 1 / sqrt(1 + rel_time_ratio))
		rab.Wcheck = Int64(floor(recheck_scale * rab.Wcheck + 1))

		if maximum(converged_se) <= rab.ϵ && minimum(converged_effs) >= rab.ESSmin
			# All diagnostics converged.
			# Step-size reduction: γ₁ = γ² (quadratic, since bias=γ, noise=√γ₁)
			# Tolerance tracks bias: ϵ₁ = ϵ * γ (since bias₁ = γ₁ = γ²)
			γ_old = rab.alg.γ
			rab.alg.γ = γ_old^2
			rab.ϵ *= γ_old

			# Carry forward the converged estimate with weight = final window size
			# γ_prev records the step size that produced x_prev (before reduction)
			window_size = sum(rab.batches[i].count for i in selected)
			rab.x_prev .= converged_mean
			rab.n_prev = window_size
			rab.γ_prev = γ_old

			# Reset phase state
			rab.Wcheck = -1
			rab.kconv = -1
			rab.prevouteriter = rab.iter
			empty!(rab.batches)
			rab.cur_mean .= 0.0
			rab.cur_M2 .= 0.0
			rab.cur_count = 0
			rab.cur_batch_size = rab.Wmin
			rab.n_phase = 0
			rab.x_phase_sum .= 0.0
			rab.x_phase_count = 0
		end
	end

	# grab current gradient/function call counts from inner alg
	rab.nf = rab.alg.nf
	rab.ng = rab.alg.ng
end
