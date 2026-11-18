"""Models are imported lazily so that, for example, the neural network never loads LightGBM in the same
process (on macOS, LightGBM and PyTorch each ship their own OpenMP runtime and can crash together)."""
import importlib

_LAZY = {
    "Climatology": ".climatology",
    "LightGBMQuantile": ".lgbm_qr",
    "LinearQuantileRegression": ".linear_qr",
    "NNQuantile": ".nn_qr",
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")