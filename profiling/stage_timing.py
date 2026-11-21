"""Time each stage of PySRRegressor.fit without modifying PySR source.

Wraps private PySRRegressor methods at runtime and runs fit() several times
so the first (JIT-compiling) call can be compared with warm calls.

Usage: python profiling/stage_timing.py [n_fits]
"""

import functools
import json
import sys
import time
from collections import defaultdict

T0 = time.perf_counter()
import numpy as np  # noqa: E402

t = time.perf_counter()
import pysr  # noqa: E402  (starts Julia and loads SymbolicRegression.jl)
from pysr import sr as sr_mod  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - t

timings: dict[str, float] = defaultdict(float)


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timings[name] += time.perf_counter() - start

    return wrapper


cls = sr_mod.PySRRegressor
for name in [
    "_setup_equation_file",
    "_validate_and_modify_params",
    "_validate_and_set_fit_params",
    "_pre_transform_training_data",
    "_load_type_spec_runtime",
    "_checkpoint",
    "_run",
    "get_hof",
]:
    setattr(cls, name, timed(name, getattr(cls, name)))
sr_mod.jl_serialize = timed("jl_serialize", sr_mod.jl_serialize)

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

n_fits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
results = []
for i in range(n_fits):
    timings.clear()
    model = pysr.PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin"],
        parallelism="serial",
        random_state=0,
        deterministic=True,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    start = time.perf_counter()
    model.fit(X, y)
    fit_total = time.perf_counter() - start

    start = time.perf_counter()
    model.sympy()
    model.predict(X)
    post = time.perf_counter() - start

    row = dict(timings)
    inner = row["_run"] - row.get("get_hof", 0) - row.get("jl_serialize", 0)
    row["_run: options build + equation_search"] = inner
    row["fit_total"] = fit_total
    row["sympy+predict"] = post
    row["best_loss"] = float(model.get_best()["loss"])
    row["n_equations"] = len(model.equations_)
    results.append(row)

print(json.dumps({"import_pysr_s": IMPORT_SECONDS, "fits": results}, indent=2))
