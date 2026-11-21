"""Diff the Options PySR passes to equation_search against the Options used
by profiling/stage_profile.jl, to explain the PySR vs direct-Julia time gap."""

import numpy as np

import pysr
from pysr.julia_helpers import jl_deserialize
from pysr.julia_import import jl

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
model = pysr.PySRRegressor(
    niterations=1,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["cos", "exp", "sin"],
    parallelism="serial",
    random_state=0,
    deterministic=True,
    progress=False,
    verbosity=0,
    temp_equation_file=True,
)
model.fit(X, y)
py_opts = jl_deserialize(model.julia_options_stream_)
jl.py_opts = py_opts
print(
    jl.seval(
        """
        let a = py_opts,
            b = SymbolicRegression.Options(;
                binary_operators=[+, *, -, /], unary_operators=[cos, exp, sin],
                populations=31, population_size=27, ncycles_per_iteration=380,
                maxsize=30, deterministic=true, seed=0, save_to_file=false)
            io = IOBuffer()
            for f in fieldnames(typeof(a))
                f in (:operators, :input_stream, :output_directory, :seed) && continue
                va, vb = getfield(a, f), getfield(b, f)
                isequal(va, vb) || string(va) == string(vb) ||
                    println(io, rpad(string(f), 32), " pysr=", first(repr(va), 120), "   julia=", first(repr(vb), 120))
            end
            String(take!(io))
        end
        """
    )
)
