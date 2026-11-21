# Check SymbolicRegression's eval_tree_array(tree, X, options) wrapper with
# `options` known only by type (as inside the search loop), so field access is
# not constant-folded the way it is for a const global.
using SymbolicRegression, InteractiveUtils, Random
Random.seed!(0)
X = randn(Float32, 5, 1000)
y = 2.5382f0 .* cos.(X[4, :]) .+ X[1, :] .^ 2 .- 0.5f0
options = Options(; binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
    population_size=27, ncycles_per_iteration=380, maxsize=30)
dataset = Dataset(X, y)
ps = SymbolicRegression.init_plugin_states(options, dataset)
pop = Population(dataset; population_size=27, nlength=3, options, nfeatures=5, plugin_states=ps)
ex = pop.members[1].tree
@noinline sr_eval(ex, X, options) = eval_tree_array(ex, X, options)
sr_eval(ex, X, options)
println("--- optimized IR, sr_eval(ex, X, options) ---")
display(code_typed(sr_eval, (typeof(ex), typeof(X), typeof(options)); optimize=true))
println("\n--- @code_warntype ---")
@code_warntype sr_eval(ex, X, options)
println("\noptions.operators field type: ", fieldtype(typeof(options), :operators))
println("elementwise_loss field type: ", fieldtype(typeof(options), :elementwise_loss))
for f in fieldnames(typeof(options))
    T = fieldtype(typeof(options), f)
    isconcretetype(T) || T isa Union && all(isconcretetype, Base.uniontypes(T)) || println("abstract Options field: ", f, "::", T)
end

# One level down: the SR wrapper body itself.
ci, _ = only(code_typed(sr_eval, (typeof(ex), typeof(X), typeof(options)); optimize=true))
inv = ci.code[1].args[1]
inv isa Core.CodeInstance && (inv = inv.def)
println("\n--- SR wrapper body: ", inv.def.name, " ---")
body, rt = only(Base.code_typed_by_type(inv.specTypes; optimize=true))
println("return type: ", rt)
for (i, st) in enumerate(body.code)
    Meta.isexpr(st, :call) || continue
    fa = st.args[1]
    fv = fa isa GlobalRef ? getglobal(fa.mod, fa.name) : fa
    (fv isa Core.Builtin || fv isa Core.IntrinsicFunction) && continue
    println("runtime dispatch %$i::", body.ssavaluetypes[i], "  <=  ", first(sprint(show, st), 250))
end
for (i, t) in enumerate(body.ssavaluetypes)
    T = Core.Compiler.widenconst(t)
    st = body.code[i]
    st isa Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode,Nothing} && continue
    isconcretetype(T) || T === Union{} || println("non-concrete %$i::", T, "  <=  ", first(sprint(show, st), 200))
end
