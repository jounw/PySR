# Shared by gap_julia_cli.jl and gap_python.py: run equation_search on the
# numpy dataset written by gap_python.py, with the Options PySR builds.
using SymbolicRegression

function load_xy(dir)
    X = Matrix{Float32}(undef, 1000, 5)
    y = Vector{Float32}(undef, 1000)
    read!(joinpath(dir, "X.f32"), X)  # row-major from numpy == column-major X'
    read!(joinpath(dir, "y.f32"), y)
    return permutedims(X), y
end

function run_direct(X, y; save_to_file::Bool, niterations=40, seed=0)
    options = Options(;
        binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
        populations=31, population_size=27, ncycles_per_iteration=380,
        maxsize=30, deterministic=true, seed, save_to_file,
        output_directory=mktempdir(),
    )
    t = @elapsed equation_search(X, y; options, niterations, parallelism=:serial,
        verbosity=0, progress=false, return_state=true)
    return t
end
