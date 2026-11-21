# Shared by gap_julia_cli.jl and gap_python.py: run equation_search on the
# numpy dataset written by gap_python.py, with the Options PySR builds.
using SymbolicRegression

function load_xy(dir)
    # numpy's row-major (1000, 5) bytes are exactly a column-major 5x1000
    # matrix, which is the features-by-rows layout equation_search expects.
    X = Matrix{Float32}(undef, 5, 1000)
    y = Vector{Float32}(undef, 1000)
    read!(joinpath(dir, "X.f32"), X)
    read!(joinpath(dir, "y.f32"), y)
    return X, y
end

function run_direct(X, y; save_to_file::Bool, niterations=40, seed=0)
    options = Options(;
        binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
        populations=31, population_size=27, ncycles_per_iteration=380,
        maxsize=30, deterministic=true, seed, save_to_file,
        output_directory=mktempdir(),
    )
    t = @elapsed ((_, hof) = equation_search(X, y; options, niterations,
        parallelism=:serial, verbosity=0, progress=false, return_state=true))
    # Summarize the hall of fame so runs can be checked for doing the same work.
    members = [m for (m, e) in zip(hof.members, hof.exists) if e]
    best = minimum(m.loss for m in members)
    return (t, length(members), Float64(best))
end
