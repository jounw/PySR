# Measure what the type instabilities found by warntype.jl cost per call:
# eval_loss / simplify+combine with the search's Options (abstract operators
# field, so a runtime dispatch per call) vs. specialized_options (concrete).
# Alternates A/B 7 times and reports the median ns per call.
using SymbolicRegression, Random, Statistics, Printf
const SR = SymbolicRegression
const LF = SR.LossFunctionsModule
Random.seed!(0)
const NROWS = parse(Int, get(ENV, "NROWS", "1000"))
println("rows: ", NROWS)
X = randn(Float32, 5, NROWS)
y = 2.5382f0 .* cos.(X[4, :]) .+ X[1, :] .^ 2 .- 0.5f0
options = Options(; binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
    population_size=27, ncycles_per_iteration=380, maxsize=30)
spec = SR.CoreModule.OptionsStructModule.specialized_options(options)
dataset = Dataset(X, y)
LF.update_baseline_loss!(dataset, options)
ps = SR.init_plugin_states(options, dataset)
pop = Population(dataset; population_size=200, nlength=10, options, nfeatures=5, plugin_states=ps)
trees = [m.tree for m in pop.members]
println("abstract operators field: ", fieldtype(typeof(options), :operators))
println("specialized operators field concrete: ", isconcretetype(fieldtype(typeof(spec), :operators)))

@noinline function loss_loop(trees, dataset, opt, n)
    s = 0.0f0
    for _ in 1:n, t in trees
        s += LF.eval_loss(t, dataset, opt)
    end
    s
end
@noinline function simp_loop(trees, opt, n)
    c = 0
    for _ in 1:n, t in trees
        t2 = combine_operators(simplify_tree!(copy(t), opt.operators), opt.operators)
        c += length(get_tree(t2))
    end
    c
end

function bench(f, args...; reps=7)
    f(args...)
    ts = Float64[]
    for _ in 1:reps
        push!(ts, @elapsed f(args...))
    end
    ts
end

ncalls_loss = 20 * length(trees)
ncalls_simp = 50 * length(trees)
# Check both variants compute the same thing (question 4: did it error?).
@assert loss_loop(trees, dataset, options, 1) == loss_loop(trees, dataset, spec, 1)
res = Dict{String,Vector{Float64}}()
for r in 1:7  # alternate A/B
    for (name, opt) in (("generic", options), ("specialized", spec))
        append!(get!(res, "eval_loss/" * name, Float64[]), bench(loss_loop, trees, dataset, opt, 20; reps=1) ./ ncalls_loss)
        append!(get!(res, "simplify+combine/" * name, Float64[]), bench(simp_loop, trees, opt, 50; reps=1) ./ ncalls_simp)
    end
end
for k in sort(collect(keys(res)))
    v = res[k] .* 1e9
    @printf("%-32s median %8.0f ns/call  range %8.0f - %8.0f  (n=%d)\n", k, median(v), minimum(v), maximum(v), length(v))
end
