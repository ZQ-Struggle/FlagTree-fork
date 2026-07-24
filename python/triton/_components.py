from __future__ import annotations

from importlib import import_module
import sys
from threading import RLock
from types import ModuleType
from typing import Any, MutableMapping


COMPONENT_API_VERSION = 1
_SPECS = {
    "debugger": ("flagtree_debugger", "flagtree-debugger"),
    "profiler": ("flagtree_profiler", "flagtree-profiler"),
}


class ComponentNotInstalledError(ModuleNotFoundError):
    pass


class ComponentCompatibilityError(ImportError):
    pass


_lock = RLock()
_components: dict[str, Any] = {}


def _spec(name: str) -> tuple[str, str]:
    try:
        return _SPECS[name]
    except KeyError as error:
        raise ComponentCompatibilityError(
            f"unsupported FlagTree component {name!r}"
        ) from error


def _validate_component(name: str, component: Any) -> Any:
    actual_name = str(getattr(component, "name", ""))
    if actual_name != name:
        raise ComponentCompatibilityError(
            f"FlagTree component {name!r} returned component {actual_name!r}"
        )
    api_version = int(getattr(component, "api_version", -1))
    if api_version != COMPONENT_API_VERSION:
        raise ComponentCompatibilityError(
            f"FlagTree {name} component API mismatch: core={COMPONENT_API_VERSION}, "
            f"component={api_version}. Install matching FlagTree wheels."
        )
    core_series = str(getattr(component, "core_version_series", ""))
    if core_series:
        from triton import __version__

        installed_series = ".".join(str(__version__).split(".")[:2])
        if installed_series != core_series:
            raise ComponentCompatibilityError(
                f"FlagTree {name} targets core {core_series}, but the installed "
                f"FlagTree core is {__version__}. Install matching wheels."
            )
    return component


def register_component(name: str, component: Any) -> Any:
    _spec(name)
    component = _validate_component(name, component)
    with _lock:
        current = _components.get(name)
        if current is not None and current is not component:
            raise ComponentCompatibilityError(
                f"FlagTree component {name!r} is already loaded"
            )
        _components[name] = component
    return component


def load_component(name: str, *, required: bool = True) -> Any | None:
    module_name, install_name = _spec(name)
    with _lock:
        if name in _components:
            return _components[name]
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            if not required:
                return None
            raise ComponentNotInstalledError(
                f"FlagTree {name} is not installed. Install it with "
                f"`python -m pip install {install_name}`."
            ) from None
        return register_component(name, getattr(module, "component", module))


def expose_public_module(namespace: MutableMapping[str, Any], name: str) -> None:
    component = load_component(name)
    implementation = component.module()
    if not isinstance(implementation, ModuleType):
        raise ComponentCompatibilityError(
            f"FlagTree component {name!r} did not provide a public Python module"
        )
    public_names = tuple(
        getattr(
            implementation,
            "__all__",
            (item for item in vars(implementation) if not item.startswith("_")),
        )
    )
    namespace["_impl"] = implementation
    namespace["__all__"] = public_names
    if hasattr(implementation, "__path__"):
        namespace["__path__"] = implementation.__path__
        private_prefix = f"{implementation.__name__}."
        public_prefix = str(namespace["__name__"])
        for module_name, module in tuple(sys.modules.items()):
            if module_name.startswith(private_prefix):
                suffix = module_name[len(implementation.__name__):]
                sys.modules.setdefault(f"{public_prefix}{suffix}", module)
    namespace["__getattr__"] = lambda attribute: getattr(implementation, attribute)
    namespace["__dir__"] = lambda: sorted(set(namespace) | set(dir(implementation)))
    for public_name in public_names:
        namespace[public_name] = getattr(implementation, public_name)


def load_dialects(context: Any) -> None:
    with _lock:
        components = tuple(_components.values())
    for component in components:
        callback = getattr(component, "load_dialects", None)
        if callable(callback):
            callback(context)


def _call_debugger(method: str, *args: Any):
    with _lock:
        component = _components.get("debugger")
    callback = getattr(component, method, None)
    if callable(callback):
        return callback(*args)
    return None


def apply_compile_options(options: dict[str, Any]) -> None:
    _call_debugger("apply_compile_options", options)


def run_compiler_hook(stage: str, module: Any, metadata: dict[str, Any]) -> None:
    _call_debugger("run_compiler_hook", stage, module, metadata)


def annotate_statement(kind: str, generator: Any, node: Any, target: Any, value: Any) -> None:
    _call_debugger("annotate_statement", kind, generator, node, target, value)


def debug_collect_start(semantic: Any, level: Any, addr_level: Any):
    return load_component("debugger").debug_collect_start(semantic, level, addr_level)


def debug_collect_end(semantic: Any):
    return load_component("debugger").debug_collect_end(semantic)
