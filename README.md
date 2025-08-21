# TinyMPC Python Interface

Python wrapper for [TinyMPC](https://tinympc.org/). Provides MPC setup/solve and code generation with the C++ core. Tested on Linux/macOS.

## Installation

```bash
pip install tinympc
```

For development:
```bash
git clone --recurse-submodules https://github.com/TinyMPC/tinympc-python.git
cd tinympc-python
pip install -e .
```
If you cloned without submodules:
```bash
git submodule update --init --recursive
```

## Examples

See `examples/` for end-to-end scripts:

- cartpole_example_one_solve.py – single solve
- cartpole_example_mpc.py – MPC loop
- cartpole_example_mpc_reference_constrained.py – reference tracking + constraints
- cartpole_example_code_generation.py – codegen for cartpole
- quadrotor_hover_code_generation.py – codegen for quadrotor (requires `pip install autograd`)
- rocket_landing_constraints.py – SOC + bounds example

## Usage

### Basic MPC workflow

```python
import numpy as np
import tinympc

A = ...  # nx x nx
B = ...  # nx x nu
Q = ...  # nx x nx (diagonal)
R = ...  # nu x nu (diagonal)
N = 20

solver = tinympc.TinyMPC()
solver.setup(A, B, Q, R, N, rho=1.0, verbose=False)

x0 = np.array([...])
x_ref = np.zeros((A.shape[0], N))
u_ref = np.zeros((B.shape[1], N-1))

solver.set_x0(x0)
solver.set_x_ref(x_ref)
solver.set_u_ref(u_ref)

solution = solver.solve()  # dict
u0 = solution["controls"]         # first control (nu,)
X = solution["states_all"].T       # N x nx
U = solution["controls_all"].T     # (N-1) x nu
```

### Code generation

```python
solver = tinympc.TinyMPC()
solver.setup(A, B, Q, R, N, rho=1.0)

# Optional: bounds
u_min, u_max = -0.5, 0.5
solver.set_bound_constraints([], [], u_min, u_max)

solver.codegen("out")  # generates C++ sources into ./out
```

### Adaptive Rho (sensitivity) workflow

```python
from autograd import numpy as anp

solver = tinympc.TinyMPC()
solver.setup(A, B, Q, R, N, rho=1.0)

# Option 1: numerical sensitivity (if implemented in your workflow)
# dK, dP, dC1, dC2 = solver.compute_sensitivity_autograd()

# Option 2: compute cache terms, then generate with precomputed sensitivity
Kinf, Pinf, Quu_inv, AmBKt = solver.compute_cache_terms()

# If you already have dK, dP, dC1, dC2:
# solver.codegen_with_sensitivity("out", dK, dP, dC1, dC2)
```

## Constraints API

Constraints are set explicitly (setup does not set any). Each call auto-enables the corresponding flags.

```python
# Bounds (box)
solver.set_bound_constraints(x_min, x_max, u_min, u_max)

# Linear inequalities
solver.set_linear_constraints(Alin_x, blin_x, Alin_u, blin_u)

# Second‑order cones (inputs first, then states)
solver.set_cone_constraints(Acu, qcu, cu, Acx, qcx, cx)

# Equality (as two inequalities)
solver.set_equality_constraints(Aeq_x, beq_x, Aeq_u, beq_u)
```

Notes:
- Shapes across horizon:
  - x_min/x_max: nx × N, u_min/u_max: nu × (N−1). Scalars/vectors are expanded automatically.
- Linear vectors `blin_x`, `blin_u` can be 1×K or K×1; they are normalized internally.
- SOC `cu`, `cx` must be vectors (1×K or K×1 both accepted).

## API Reference

Core methods:
```python
solver.setup(A, B, Q, R, N, rho=1.0, fdyn=None, verbose=False, **settings)
solver.set_x0(x0)
solver.set_x_ref(x_ref)
solver.set_u_ref(u_ref)
solver.set_bound_constraints(x_min, x_max, u_min, u_max)
solver.set_linear_constraints(Alin_x, blin_x, Alin_u, blin_u)
solver.set_cone_constraints(Acu, qcu, cu, Acx, qcx, cx)
solver.set_equality_constraints(Aeq_x, beq_x, Aeq_u=None, beq_u=None)
solver.update_settings(abs_pri_tol=1e-3, abs_dua_tol=1e-3, max_iter=100, ...)
solution = solver.solve()
solver.codegen(output_dir)
solver.codegen_with_sensitivity(output_dir, dK, dP, dC1, dC2)
```

## Docs

See the website: `https://tinympc.org`.
