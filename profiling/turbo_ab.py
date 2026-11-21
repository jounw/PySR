"""A/B PySRRegressor.fit with turbo=False vs turbo=True on the same data and seed.

Warms up both sides first (turbo=True installs and loads LoopVectorization.jl
on first use), then alternates A, B, A, B, ... and prints time, hall-of-fame
size, and best loss per run so both sides can be checked for doing the same work.

Usage: python profiling/turbo_ab.py [n_pairs] [n_rows]
"""

import sys
import time

import numpy as np

import pysr

n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

rng = np.random.default_rng(0)
X = rng.normal(size=(n_rows, 5)).astype(np.float32)
y = (2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5).astype(np.float32)


def fit(turbo: bool, niterations: int = 40):
    m = pysr.PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin"],
        parallelism="serial",
        random_state=0,
        deterministic=True,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        turbo=turbo,
    )
    t = time.perf_counter()
    m.fit(X, y)
    t = time.perf_counter() - t
    eq = m.equations_
    return t, len(eq), float(eq["loss"].min()), list(eq["equation"])


for turbo in (False, True):
    t, *_ = fit(turbo, niterations=2)
    print(f"warmup turbo={turbo}: {t:.2f} s", flush=True)

results = {False: [], True: []}
equations = {}
for _ in range(n_pairs):
    for turbo in (False, True):
        t, n, best, eqs = fit(turbo)
        results[turbo].append(t)
        equations.setdefault(turbo, eqs)
        print(f"rows={n_rows} turbo={turbo}: {t:.2f} s  hof={n} best_loss={best:.3g}", flush=True)

for turbo, ts in results.items():
    print(
        f"turbo={turbo}: median {np.median(ts):.2f} s  "
        f"range {min(ts):.2f}-{max(ts):.2f}  n={len(ts)}"
    )
print("same final equations:", equations[False] == equations[True])
