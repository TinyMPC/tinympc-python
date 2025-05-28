Python wrapper for [TinyMPC](https://tinympc.org/).

## Installation

```bash
    pip install tinympc
```


For development installation:
```bash
git clone --recursive https://github.com/TinyMPC/tinympc-python.git
cd tinympc-python
pip install -e .
```

## Examples

The `examples/` directory contains:
- `cartpole_example_one_solve.py`
- `cartpole_example_mpc_constrained.py`
- `cartpole_example_mpc.py`
- `cartpole_example_code_generation.py`
- `quadrotor_hover_code_generation.py` - For online hyperparameter tuning of rho set ```ENABLE_ADAPTIVE_RHO``` to ```True```



## Documentation

Documentation and examples can be found [here](https://tinympc.org/get-started/installation/).