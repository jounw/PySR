# Run @code_warntype on SymbolicRegression.jl hot-path functions with the same
# setup PySR uses by default (Float32, PySR's default population settings).
#
# Writes the full @code_warntype text per function to profiling/out/warntype/
# and prints a summary of non-concrete inferred types found by code_typed.
#
# Usage: julia --project=<pysr julia_env> profiling/warntype.jl

using SymbolicRegression, InteractiveUtils, Random
using SymbolicRegression: eval_tree_array, simplify_tree!, combine_operators
const SR = SymbolicRegression
const SI = SR.SingleIterationModule
const RE = SR.RegularizedEvolutionModule
const MU = SR.MutateModule
const CX = SR.CrossoverModule
const CO = SR.ConstantOptimizationModule
const LF = SR.LossFunctionsModule
const PO = SR.PopulationModule
const HF = SR.HallOfFameModule
const CP = SR.ComplexityModule

Random.seed!(0)
const X = randn(Float32, 5, 1000)
const y = 2.5382f0 .* cos.(X[4, :]) .+ X[1, :] .^ 2 .- 0.5f0

const options = Options(;
    binary_operators=[+, *, -, /],
    unary_operators=[cos, exp, sin],
    population_size=27,
    ncycles_per_iteration=380,
    maxsize=30,
)
const dataset = Dataset(X, y)
LF.update_baseline_loss!(dataset, options)
const plugin_states = SR.init_plugin_states(options, dataset)
const pop = Population(dataset; population_size=27, nlength=3, options, nfeatures=5, plugin_states)
const member = pop.members[1]
const member2 = pop.members[2]
const ex = member.tree
const node = ex.tree
const trace = SR.TracingModule.new_trace(options)
const hof = HF.HallOfFame(options, dataset)

# (label, target function, thunk). Keyword calls go through a thunk; the
# optimized IR of the thunk then holds an :invoke of the real body method
# (keyword wrappers inline away), which is what @code_warntype should inspect.
cases = [
    ("eval_tree_array(ex, X, operators)", eval_tree_array, () -> eval_tree_array(ex, X, options.operators)),
    ("eval_tree_array(node, X, operators)", eval_tree_array, () -> eval_tree_array(node, X, options.operators)),
    ("eval_tree_array(ex, X, options) [SR wrapper]", eval_tree_array, () -> eval_tree_array(ex, X, options)),
    ("_update_hall_of_fame_unchecked!", HF._update_hall_of_fame_unchecked!, () -> HF._update_hall_of_fame_unchecked!(hof, member, 5)),
    ("eval_loss", LF.eval_loss, () -> LF.eval_loss(ex, dataset, options)),
    ("_eval_loss", LF._eval_loss, () -> LF._eval_loss(ex, dataset, options, true, nothing)),
    ("eval_cost", LF.eval_cost, () -> LF.eval_cost(dataset, member, options)),
    ("compute_complexity", CP.compute_complexity, () -> CP.compute_complexity(ex, options)),
    ("best_of_sample", PO.best_of_sample, () -> PO.best_of_sample(pop, options; plugin_states)),
    ("next_generation", MU.next_generation, () -> MU.next_generation(dataset, member, 30, options; tmp_trace=trace, plugin_states)),
    ("crossover_generation", CX.crossover_generation, () -> CX.crossover_generation(member, member2, dataset, 30, options; plugin_states)),
    ("optimize_constants", CO.optimize_constants, () -> CO.optimize_constants(dataset, member, options)),
    ("simplify_tree!", simplify_tree!, () -> simplify_tree!(copy(ex), options.operators)),
    ("combine_operators", combine_operators, () -> combine_operators(copy(ex), options.operators)),
    ("reg_evol_cycle", RE.reg_evol_cycle, () -> RE.reg_evol_cycle(dataset, pop, 30, options, trace; plugin_states, best_seen=hof)),
    ("s_r_cycle", SI.s_r_cycle, () -> SI.s_r_cycle(dataset, pop, 1, 30; options, trace, plugin_states)),
    ("optimize_and_simplify_population", SI.optimize_and_simplify_population, () -> SI.optimize_and_simplify_population(dataset, pop, options, 30, trace)),
    ("update_hall_of_fame!", HF.update_hall_of_fame!, () -> HF.update_hall_of_fame!(hof, pop.members, options)),
]

# A type is "bad" if it is not concrete. Small Unions of concrete types are
# reported separately because Julia handles them with union splitting.
function classify(t)
    t isa Core.Const && return :ok
    t isa Core.PartialStruct && return :ok
    t = Core.Compiler.widenconst(t)
    isconcretetype(t) && return :ok
    t === Union{} && return :ok
    t isa Type{<:Type} && return :ok
    if t isa Union
        parts = Base.uniontypes(t)
        all(isconcretetype, parts) && length(parts) <= 4 && return :small_union
    end
    return :abstract
end

outdir = joinpath(@__DIR__, "out", "warntype")
mkpath(outdir)

function find_target(thunk, f)
    ci, _ = only(code_typed(thunk, (); optimize=true))
    fname = string(nameof(f))
    for stmt in ci.code
        Meta.isexpr(stmt, :invoke) || continue
        mi = stmt.args[1]
        mi isa Core.CodeInstance && (mi = mi.def)
        mi isa Core.MethodInstance || continue
        n = string(mi.def.name)
        if n == fname || startswith(n, "#" * fname * "#")
            T = mi.specTypes
            return T.parameters[1].instance, Tuple{T.parameters[2:end]...}
        end
    end
    # Fallback: walk the unoptimized IR. A positional call to `f` is the
    # target; a keyword call goes through Core.kwcall, so descend into it and
    # take the `#f#N` body function it calls.
    function walk(g, argT, depth)
        depth > 3 && return nothing
        ci0, _ = only(code_typed(g, argT; optimize=false))
        at(a) = a isa Core.SSAValue ? Core.Compiler.widenconst(ci0.ssavaluetypes[a.id]) :
            a isa Core.Argument ? Core.Compiler.widenconst(ci0.slottypes[a.n]) :
            a isa Core.SlotNumber ? Core.Compiler.widenconst(ci0.slottypes[a.id]) :
            a isa GlobalRef ? Core.Typeof(getglobal(a.mod, a.name)) : Core.Typeof(a)
        for stmt in ci0.code
            Meta.isexpr(stmt, :call) || continue
            ft = at(stmt.args[1])
            args = map(at, stmt.args[2:end])
            if ft === typeof(f) || (isdefined(ft, :instance) && startswith(string(nameof(ft.instance)), "#" * fname * "#"))
                return ft.instance, Tuple{args...}
            elseif ft === typeof(Core.kwcall) && length(args) >= 2 && args[2] === typeof(f)
                r = walk(Core.kwcall, Tuple{args...}, depth + 1)
                r === nothing || return r
            end
        end
        return nothing
    end
    r = walk(thunk, Tuple{}, 0)
    r === nothing || return r
    error("no call to $fname found in thunk IR")
end

summary = []
for (name, target, thunk) in cases
    println(stderr, "checking ", name)
    thunk()  # run once so it compiles and is known to work
    io = IOBuffer()
    f, rest = find_target(thunk, target)
    println(stderr, "  -> ", f, " ", rest)
    code_warntype(IOContext(io, :color => false), f, rest)
    text = String(take!(io))
    write(joinpath(outdir, replace(name, r"[^A-Za-z0-9_!]" => "_") * ".txt"), text)

    ci2, rt2 = only(code_typed(f, rest; optimize=true))
    bad = String[]
    unions = String[]
    dyn = String[]
    for (i, t) in enumerate(ci2.ssavaluetypes)
        stmt = ci2.code[i]
        stmt isa Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode,Nothing} && continue
        # A :call that survives optimization to a non-builtin function is a
        # runtime dispatch (statically resolved calls become :invoke).
        if Meta.isexpr(stmt, :call)
            fa = stmt.args[1]
            fv = fa isa GlobalRef ? (isdefined(fa.mod, fa.name) ? getglobal(fa.mod, fa.name) : nothing) : fa
            if !(fv isa Core.Builtin || fv isa Core.IntrinsicFunction)
                push!(dyn, first("%$i::$(Core.Compiler.widenconst(t))  <=  $(sprint(show, stmt))", 300))
            end
        end
        c = classify(t)
        c === :ok && continue
        s = "%$i::$(Core.Compiler.widenconst(t))  <=  $(sprint(show, ci2.code[i]))"
        push!(c === :abstract ? bad : unions, first(s, 300))
    end
    slotbad = String[]
    for (sn, st) in zip(ci2.slotnames, ci2.slottypes)
        classify(st) === :abstract && push!(slotbad, "$(sn)::$(Core.Compiler.widenconst(st))")
    end
    push!(summary, (; name, f=string(f), argtypes=string(rest), rt=rt2,
        rt_class=classify(rt2), nbad=length(bad), nunion=length(unions),
        bad, unions, slotbad, dyn))
end

for s in summary
    println("="^100)
    println(s.name, "   [", s.f, "]")
    println("  return type: ", s.rt, "  (", s.rt_class, ")")
    println("  runtime dispatches: ", length(s.dyn))
    for l in first(s.dyn, 12)
        println("    dyn   ", l)
    end
    println("  abstract SSA values: ", s.nbad, "   small unions: ", s.nunion,
        "   abstract slots: ", length(s.slotbad))
    for l in s.slotbad
        println("    slot  ", l)
    end
    for l in first(s.bad, 15)
        println("    bad   ", l)
    end
    for l in first(s.unions, 5)
        println("    union ", l)
    end
end
