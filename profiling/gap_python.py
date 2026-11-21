"""Same data and Options three ways inside Python: direct equation_search via
juliacall (save_to_file off/on) and PySRRegressor.fit."""

import os
import time

import numpy as np

import pysr
from pysr.julia_import import jl

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "out", "data")
os.makedirs(data, exist_ok=True)
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5)).astype(np.float32)
y = (2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5).astype(np.float32)
X.tofile(os.path.join(data, "X.f32"))
y.tofile(os.path.join(data, "y.f32"))

jl.seval(f'include(raw"{os.path.join(here, "direct_run.jl")}")')
jl.seval(f'const GX, GY = load_xy(raw"{data}")')
# PySR derives the Julia seed from random_state; use the same one.
from sklearn.utils import check_random_state

SEED = int(check_random_state(0).randint(0, 2**31 - 1))
with open(os.path.join(data, "seed.txt"), "w") as f:
    f.write(str(SEED))
jl.seval("run_direct(GX, GY; save_to_file=false, niterations=2)")


def pysr_fit():
    m = pysr.PySRRegressor(
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
    t = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t


pysr_fit()  # warm up the PySR-specific path
for _ in range(2):
    for save in ("false", "true"):
        t = jl.seval(f"run_direct(GX, GY; save_to_file={save}, seed={SEED})")
        print(f"python-juliacall direct save_to_file={save}: {t:.2f} s", flush=True)
    print(f"python PySRRegressor.fit: {pysr_fit():.2f} s", flush=True)
