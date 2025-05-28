Python wrapper for [TinyMPC](https://tinympc.org/).

## Installation

```bash
pip install tinympc
```

For development installation (optional):
```bash
git clone https://github.com/TinyMPC/tinympc-python.git
cd tinympc-python
pip install -e .
```

## Examples

The `examples/` directory contains several demonstration files:

### Basic Examples
- `cartpole_example_one_solve.py` - Single solve for cartpole problem
- `cartpole_example_mpc.py` - MPC implementation for cartpole
- `cartpole_example_mpc_constrained.py` - MPC with constraints

### Code Generation Examples 
*Note: Quadrotor Code generation examples require autograd: `pip install autograd`*

- `cartpole_example_code_generation.py` - Code generation for cartpole
- `quadrotor_hover_code_generation.py` - Code generation for quadrotor hover
  - For online hyperparameter tuning, set `ENABLE_ADAPTIVE_RHO = True` in the file

## Documentation

Documentation and examples can be found [here](https://tinympc.org/get-started/installation/).