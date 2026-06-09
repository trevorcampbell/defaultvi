# Default Optimizers for VI

This is the repository of code and results for

Campbell, Huggins, Kim, Margossian. "Large-scale empirical tuning and comparison of default optimizers for variational inference." https://arxiv.org/pdf/2606.07841 .

Running this code requires SubsampledPosteriorDB: https://github.com/trevorcampbell/subsampledposteriordb 

Results tarball will be uploaded and linked from here soon. Please check back soon for more detailed instructions. 

Scripts:
- `main.jl`: Runs one optimization algorithm on one objective for one target, with potentially multiple step sizes.
- `problem_characteristics.jl`: Computes the condition number, noise, and dimension for each problem in PosteriorDB.
- `plot_problem_characteristics.jl`: Plots Figure 1, the scatter of dimension vs condition number coloured by noise, and the histogram of optimization parameter dimension.
- `plot_iteration_hist.jl`: Plots Figure 2, scatter of number of completed iterations versus target/objective dimension and coloured by objective type.
- `plot_tuning.jl`: Plots Figure 3, Figure 9, Figure 10, Figure 11, and computes the tuned step size parameters for all algorithms .
- `plot_relsqgrad.jl`: Plots Figure 7, final relative square gradient norm.
- `plot_relsqgrad_stepsizes.jl`: Plots Figure 5, relative squared gradient norm for a few algorithms versus step size.
- `plot_individual_traces.jl`: Plots Figure 6 and 12, individual optimization traces with a few algorithms highlighted.

All other `.jl` files are implementations of various optimization algorithms and utility code.



