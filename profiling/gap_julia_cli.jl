include(joinpath(@__DIR__, "direct_run.jl"))
X, y = load_xy(joinpath(@__DIR__, "out", "data"))
seed = parse(Int, read(joinpath(@__DIR__, "out", "data", "seed.txt"), String))
run_direct(X, y; save_to_file=false, niterations=2)
for save in (false, true, false, true)
    println("julia-cli save_to_file=$save: ", round(run_direct(X, y; save_to_file=save, seed); digits=2), " s")
end
