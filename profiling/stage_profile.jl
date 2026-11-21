# Sample-profile SymbolicRegression.equation_search and attribute samples to
# search stages (evolution, mutation, evaluation, constant optimization, ...).
#
# Settings mirror profiling/stage_timing.py: Float32, PySR default population
# settings, 40 iterations, serial, deterministic.
#
# Usage: julia --project=<pysr julia_env> profiling/stage_profile.jl

using SymbolicRegression, Profile, Random, Printf

Random.seed!(0)
X = randn(Float32, 5, 1000)
y = 2.5382f0 .* cos.(X[4, :]) .+ X[1, :] .^ 2 .- 0.5f0

function run(niterations)
    options = Options(;
        binary_operators=[+, *, -, /],
        unary_operators=[cos, exp, sin],
        populations=31,
        population_size=27,
        ncycles_per_iteration=380,
        maxsize=30,
        deterministic=true,
        seed=0,
        save_to_file=false,
    )
    equation_search(X, y; options, niterations, parallelism=:serial,
        verbosity=0, progress=false, return_state=true)
end

# Cold call includes compilation of the specialized search for these options.
t_cold = @timed run(2)
@printf("cold run (niterations=2): %.2f s, compile %.1f%%, gc %.1f%%\n",
    t_cold.time, 100 * t_cold.compile_time / t_cold.time, 100 * t_cold.gctime / t_cold.time)

warm = [@timed run(40) for _ in 1:1]
for (i, t) in enumerate(warm)
    @printf("warm run %d (niterations=40): %.2f s, compile %.2f s, gc %.1f%%, alloc %.2f GB\n",
        i, t.time, t.compile_time, 100 * t.gctime / t.time, t.bytes / 1e9)
end

Profile.init(; n=50_000_000, delay=0.001)
Profile.clear()
@profile run(40)
data, lidict = Profile.retrieve()

# Stages, outermost first. A sample counts toward every stage on its stack
# (inclusive time), so nested stages add up to more than 100%.
stages = [
    "run", "equation_search", "_equation_search", "_main_search_loop!", "_dispatch_s_r_cycle",
    "s_r_cycle", "reg_evol_cycle",
    "best_of_sample", "next_generation", "crossover_generation",
    "check_constraints", "compute_complexity",
    "optimize_and_simplify_population", "simplify_tree!", "combine_operators",
    "optimize_constants", "finalize_costs",
    "eval_cost", "eval_loss", "_eval_loss", "eval_tree_array",
    "update_hall_of_fame!", "migrate!", "update_frequencies!",
    "copy", "gc", "runtime dispatch (jl_apply_generic)", "type inference",
]
normalize(n) = replace(string(n), r"^#+" => "", r"#\d+$" => "", r"#.*$" => "")

counts = Dict(s => 0 for s in stages)
# Exclusive: each sample goes to its innermost frame among these buckets.
buckets = ["eval_tree_array", "_eval_loss", "optimize_constants", "simplify_tree!",
    "combine_operators", "next_generation", "crossover_generation", "best_of_sample",
    "reg_evol_cycle", "s_r_cycle", "optimize_and_simplify_population",
    "update_hall_of_fame!", "migrate!", "_main_search_loop!", "equation_search", "run"]
excl = Dict{String,Int}()
unattributed_tops = Dict{String,Int}()
self_counts = Dict{String,Int}()
nsamples = 0
i = 1
while i <= length(data)
    j = i
    while j <= length(data) && data[j] != 0
        j += 1
    end
    if j > i
        global nsamples += 1
        seen = Set{String}()
        first_frame = true
        for ip in data[i:(j - 1)]
            for fr in get(lidict, ip, Base.StackTraces.StackFrame[])
                n = normalize(fr.func)
                if first_frame && !fr.from_c
                    self_counts[n * " @ " * basename(string(fr.file))] =
                        get(self_counts, n * " @ " * basename(string(fr.file)), 0) + 1
                    first_frame = false
                end
                fr.from_c && occursin("gc", lowercase(n)) && push!(seen, "gc")
                fr.from_c && occursin("apply_generic", n) && push!(seen, "runtime dispatch (jl_apply_generic)")
                occursin("typeinf", n) && push!(seen, "type inference")
                n in keys(counts) && push!(seen, n)
            end
        end
        for s in seen
            counts[s] += 1
        end
        # innermost bucket: frames are listed innermost first
        b = "(none)"
        for ip in data[i:(j - 1)], fr in get(lidict, ip, Base.StackTraces.StackFrame[])
            n = normalize(fr.func)
            if n in buckets
                b = n
                break
            end
        end
        excl[b] = get(excl, b, 0) + 1
        if !("run" in seen)
            outer = "?"
            for ip in reverse(data[i:(j - 1)]), fr in get(lidict, ip, Base.StackTraces.StackFrame[])
                fr.from_c && continue
                outer = normalize(fr.func) * " @ " * basename(string(fr.file))
                break
            end
            unattributed_tops[outer] = get(unattributed_tops, outer, 0) + 1
        end
    end
    global i = j + 1
end

println("\nsamples: ", nsamples, " (1 ms period)")
println("inclusive share of profiled run(40):")
for s in stages
    @printf("  %-36s %6.1f%%\n", s, 100 * counts[s] / nsamples)
end
println("\ntop 25 self-time frames:")
for (k, v) in first(sort(collect(self_counts); by=last, rev=true), 25)
    @printf("  %6.1f%%  %s\n", 100 * v / nsamples, k)
end

println("\nexclusive (innermost bucket) share:")
for (k, v) in sort(collect(excl); by=last, rev=true)
    @printf("  %6.1f%%  %s\n", 100 * v / nsamples, k)
end
println("\nsamples without `run` on the stack, by outermost Julia frame:")
for (k, v) in first(sort(collect(unattributed_tops); by=last, rev=true), 10)
    @printf("  %6.1f%%  %s\n", 100 * v / nsamples, k)
end
