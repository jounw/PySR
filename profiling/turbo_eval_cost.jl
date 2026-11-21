# Per-call eval_loss cost with turbo=false vs turbo=true on the same trees,
# to separate a per-evaluation speedup from a change in search trajectory.
# Alternates A/B 7 times; also reports how far the two losses differ.
using SymbolicRegression, LoopVectorization, Random, Statistics, Printf
const SR = SymbolicRegression
const LF = SR.LossFunctionsModule
const NROWS = parse(Int, get(ENV, "NROWS", "1000"))
Random.seed!(0)
X = randn(Float32, 5, NROWS)
y = 2.5382f0 .* cos.(X[4, :]) .+ X[1, :] .^ 2 .- 0.5f0
mk(turbo) = Options(; binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
    population_size=27, ncycles_per_iteration=380, maxsize=30, turbo)
plain, turbo = mk(false), mk(true)
dataset = Dataset(X, y)
LF.update_baseline_loss!(dataset, plain)
ps = SR.init_plugin_states(plain, dataset)
pop = Population(dataset; population_size=200, nlength=10, options=plain, nfeatures=5, plugin_states=ps)
trees = [m.tree for m in pop.members]

@noinline function loss_loop(trees, dataset, opt, n)
    s = 0.0
    for _ in 1:n, t in trees
        s += LF.eval_loss(t, dataset, opt)
    end
    s
end

# Did it compute the same thing? (finite losses only)
lp = [LF.eval_loss(t, dataset, plain) for t in trees]
lt = [LF.eval_loss(t, dataset, turbo) for t in trees]
ok = isfinite.(lp) .& isfinite.(lt)
rel = abs.(lp[ok] .- lt[ok]) ./ max.(abs.(lp[ok]), eps(Float32))
@printf("rows=%d trees=%d finite=%d  max rel loss diff=%.2e  median=%.2e  identical=%d\n",
    NROWS, length(trees), count(ok), maximum(rel), median(rel), count(==(0), rel))

n = max(1, 20_000 ÷ NROWS)
res = Dict("plain" => Float64[], "turbo" => Float64[])
loss_loop(trees, dataset, plain, 1); loss_loop(trees, dataset, turbo, 1)
for _ in 1:7, (name, opt) in (("plain", plain), ("turbo", turbo))
    push!(res[name], (@elapsed loss_loop(trees, dataset, opt, n)) / (n * length(trees)) * 1e9)
end
for k in ("plain", "turbo")
    v = res[k]
    @printf("eval_loss %-6s median %8.0f ns/call  range %8.0f - %8.0f  (n=7)\n", k, median(v), minimum(v), maximum(v))
end
@printf("per-call speedup: %.2fx\n", median(res["plain"]) / median(res["turbo"]))
