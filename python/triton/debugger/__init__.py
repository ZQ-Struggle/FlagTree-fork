from __future__ import annotations

import sys

from triton._components import public_module


_impl = public_module("debugger")
__all__ = tuple(getattr(_impl, "__all__", (name for name in vars(_impl) if not name.startswith("_"))))

if hasattr(_impl, "__path__"):
    __path__ = _impl.__path__
    _private_prefix = f"{_impl.__name__}."
    for _module_name, _module in tuple(sys.modules.items()):
        if _module_name.startswith(_private_prefix):
            _suffix = _module_name[len(_impl.__name__):]
            sys.modules.setdefault(f"{__name__}{_suffix}", _module)


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))


for _name in __all__:
    globals()[_name] = getattr(_impl, _name)
