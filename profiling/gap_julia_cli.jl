include(joinpath(@__DIR__, "direct_run.jl"))
X, y = load_xy(joinpath(@__DIR__, "out", "data"))
seed = parse(Int, read(joinpath(@__DIR__, "out", "data", "seed.txt"), String))
run_direct(X, y; save_to_file=false, niterations=2)
for save in (false, true, false, true)
    t, n, best = run_direct(X, y; save_to_file=save, seed)
    println("julia-cli save_to_file=$save: ", round(t; digits=2), " s  hof=$n best_loss=$(round(best; sigdigits=3))")
end
