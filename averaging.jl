# averaging schemes
# 1/t + (1-1/t)
# 1/sqrt(t) + (1-1/sqrt(t))
# latter half log spaced


mutable struct Averaging{Alg}
	alg::Alg
	iter::Int64
	nf::Int64
	ng::Int64
	x::Vector{Float64}
end

Averaging{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg} = Averaging{Alg}(Alg(copy(x), stepsz, seed), 0, 0, 0, copy(x))
Averaging{Alg}(dimension::Int64, stepsz, seed) where {Alg} = Averaging{Alg}(zeros(dimension), stepsz, seed)

function step!(avg::Averaging{Alg}, obj) where {Alg}
	avg.iter += 1
	step!(avg.alg, obj)
	avg.x = (1/avg.iter)*avg.alg.x + (1 - 1/avg.iter)*avg.x
	avg.nf = avg.alg.nf
	avg.ng = avg.alg.ng
end

mutable struct SqrtAveraging{Alg}
	alg::Alg
	iter::Int64
	nf::Int64
	ng::Int64
	x::Vector{Float64}
end

SqrtAveraging{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg} = SqrtAveraging{Alg}(Alg(copy(x), stepsz, seed), 0, 0, 0, copy(x))
SqrtAveraging{Alg}(dimension::Int64, stepsz, seed) where {Alg} = SqrtAveraging{Alg}(zeros(dimension), stepsz, seed)

function step!(avg::SqrtAveraging{Alg}, obj) where {Alg}
	avg.iter += 1
	step!(avg.alg, obj)
	avg.x = (1/sqrt(avg.iter))*avg.alg.x + (1 - 1/sqrt(avg.iter))*avg.x
	avg.nf = avg.alg.nf
	avg.ng = avg.alg.ng
end

mutable struct HalfAveraging{Alg}
	alg::Alg
	iter::Int64
	nf::Int64
	ng::Int64
	x::Vector{Float64}
	xprev::Vector{Float64}
	xcur::Vector{Float64}
	nprev::Int64
	ncur::Int64
end

HalfAveraging{Alg}(x::Vector{Float64}, stepsz, seed) where {Alg} = HalfAveraging{Alg}(Alg(copy(x), stepsz, seed), 0, 0, 0, copy(x), copy(x), copy(x), 0, 0)
HalfAveraging{Alg}(dimension::Int64, stepsz, seed) where {Alg} = HalfAveraging{Alg}(zeros(dimension), stepsz, seed)

function step!(avg::HalfAveraging{Alg}, obj) where {Alg}
	avg.iter += 1
	step!(avg.alg, obj)
	avg.ncur += 1
	avg.xcur += avg.alg.x

	x̃ = avg.nprev == 0 ? avg.xprev : avg.xprev / avg.nprev
	avg.x = (avg.xcur + max(0, avg.nprev - avg.ncur/2)*x̃)/(avg.ncur +  max(0, avg.nprev - avg.ncur/2))

	if avg.nprev <= avg.ncur/2
		avg.xprev = avg.xcur
		avg.nprev = avg.ncur
		avg.xcur = zero(avg.x)
		avg.ncur = 0
	end

	avg.nf = avg.alg.nf
	avg.ng = avg.alg.ng
end

