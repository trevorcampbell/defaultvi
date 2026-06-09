# Orabona and Pal
# Parameter-free stochastic optimization of variationally coherent functions
# arXiv:2102.00236, 2021.
# Algorithm 1

mutable struct FTRL
	rng::AbstractRNG
	nf::Int64
	ng::Int64
	iter::Int64
	γ::Float64
	S2::Float64
	Q::Float64
	θ::Vector{Float64}
	x::Vector{Float64}
	x0::Vector{Float64}
end

FTRL(x::Vector{Float64}, stepsz, seed) = FTRL(Xoshiro(seed), 0, 0, 0, stepsz, 4.0, 0.0,  zero(x), copy(x), copy(x))
FTRL(dimension::Int64, stepsz, seed) = FTRL(zeros(dimension), stepsz, seed)

function step!(ft::FTRL, obj)
	ft.iter += 1
	θnorm = sqrt(sum(ft.θ.^2)) 
	ft.x = ft.x0
	if θnorm <= ft.S2
		ft.x += ft.θ/(2*ft.S2)*exp(θnorm^2/(4*ft.S2) - ft.Q)
	else
		ft.x += ft.θ/(2*θnorm)*exp(θnorm/2 - ft.S2/4 - ft.Q)
	end
	ĝ = gradient_estimate(obj, ft.x, 1, ft.rng)
	ft.ng += 1
	# using 1/sqrt(t) learning rates; not much advice given about this in the paper, but 
	# since ||g||^2 is presumably converging -> 0, sum ||g_t||^2/t should be finite
	ℓ = (ft.γ/sqrt(ft.iter))*ĝ 
	ft.S2 += sum(ℓ.^2)
	ft.Q += sum(ℓ.^2)/sqrt(ft.S2)
	ft.θ = ft.θ-ℓ
end
