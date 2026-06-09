using PosteriorDB, Random

function main()
	
	# create a list of jobs to run on cluster compute / locally
	# randomize the problems (objective function and target) so that we can process small subsets 
	# of problems at a time to get partial results early

	# algorithms
	algs = ["Adam","AdaGrad","AMSGrad","AdaSLS","AdamSLS","AutoSGD","BigBatch","COCOB","DAdaptAdam","DAdaptSGD","DoG","DoWG","FTRL","Lion","Pflug","PoNoS","SAA","SAANA","SAALBFGS","SAACG","SGD","SLS"]
	append!(algs, ["Mecha"*T for T in ["Adam", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["RABVI"*T for T in ["Adam", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["SRABVI"*T for T in ["Adam", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	append!(algs, ["Hyper"*T for T in ["Adam", "AdaGrad", "AMSGrad", "DoG", "DoWG", "Lion", "SGD"]])
	
	# objective functions / gradient estimates
	objectives = ["MAP", "DiagGaussianVI", "FullGaussianVI", "VAEVI"]

	# targets
	pdb = PosteriorDB.database()
	targets = PosteriorDB.posterior_names(pdb)
	# remove the large neural net problem, it takes 100x as long as everything else, not suitable for cluster compute
	filter!(x ->x != "mnist-nn_rbm1bJ100", targets)
	filter!(x ->x != "mnist-nn_rbm1bJ100_subsampled", targets)

	# zip up the product of targets and objectives
	problems = [(target, objective) for target in targets for objective in objectives]
	rng = Xoshiro(1)
	shuffle!(rng, problems)
	
	open("jobs.txt", "w") do io
	for problem in problems
			for alg in algs
				println(io, alg*" "*problem[1]*" "*problem[2])
			end
		end
	end
end

main()


