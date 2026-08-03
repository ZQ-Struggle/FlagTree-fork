"""Integration boundary between FlagTree core and bundled FlagPrism.

Core compiler and runtime code must use this module instead of importing the
Debugger or Profiler implementations directly. Optional callbacks are no-ops
until their package is imported and registers a compatible component.
"""

from __future__ import annotations

from importlib import import_module
from threading import RLock
from typing import Any


COMPONENT_API_VERSION = 1
_COMPONENT_MODULES = {
    "debugger": "flagtree.debugger",
    "profiler": "flagtree.profiler",
}
_COMPONENT_BUILD_OPTION = "TRITON_BUILD_FLAGPRISM"


class ComponentNotInstalledError(ModuleNotFoundError):
    pass


class ComponentCompatibilityError(ImportError):
    pass


_lock = RLock()
_components: dict[str, Any] = {}


def _module_name(name: str) -> str:
    try:
        return _COMPONENT_MODULES[name]
    except KeyError as error:
        raise ComponentCompatibilityError(
            f"unsupported FlagPrism component {name!r}"
        ) from error


def _validate_component(name: str, component: Any) -> Any:
    actual_name = str(getattr(component, "name", ""))
    if actual_name != name:
        raise ComponentCompatibilityError(
            f"FlagPrism component {name!r} returned {actual_name!r}"
        )
    api_version = int(getattr(component, "api_version", -1))
    if api_version != COMPONENT_API_VERSION:
        raise ComponentCompatibilityError(
            f"FlagTree {name} API mismatch: core={COMPONENT_API_VERSION}, "
            f"component={api_version}. Use a matching FlagPrism submodule revision."
        )
    core_series = str(getattr(component, "core_version_series", ""))
    if core_series:
        from triton import __version__

        installed_series = ".".join(str(__version__).split(".")[:2])
        if installed_series != core_series:
            raise ComponentCompatibilityError(
                f"FlagTree {name} targets core {core_series}, but the installed "
                f"FlagTree core is {__version__}."
            )
    return component


def register_component(name: str, component: Any) -> Any:
    _module_name(name)
    component = _validate_component(name, component)
    with _lock:
        current = _components.get(name)
        if current is not None and current is not component:
            raise ComponentCompatibilityError(
                f"FlagPrism component {name!r} is already loaded"
            )
        _components[name] = component
    return component


def load_component(name: str, *, required: bool = True) -> Any | None:
    module_name = _module_name(name)
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
                f"FlagTree {name} is not included in this build. Rebuild FlagTree "
                "with its FlagPrism submodule initialized and "
                f"`{_COMPONENT_BUILD_OPTION}=ON`."
            ) from None
        return register_component(name, getattr(module, "component", module))


def _registered_components() -> tuple[Any, ...]:
    with _lock:
        return tuple(_components.values())


def _call_registered(method: str, *args: Any) -> None:
    for component in _registered_components():
        callback = getattr(component, method, None)
        if callable(callback):
            callback(*args)


def _call_required(name: str, method: str, *args: Any):
    component = load_component(name)
    callback = getattr(component, method, None)
    if not callable(callback):
        raise ComponentCompatibilityError(
            f"FlagTree {name} does not implement required callback {method!r}"
        )
    return callback(*args)


def load_dialects(context: Any) -> None:
    _call_registered("load_dialects", context)


def apply_compile_options(options: dict[str, Any]) -> None:
    _call_registered("apply_compile_options", options)


def run_compiler_hook(stage: str, module: Any, metadata: dict[str, Any]) -> None:
    _call_registered("run_compiler_hook", stage, module, metadata)


def annotate_statement(
    kind: str, generator: Any, node: Any, target: Any, value: Any
) -> None:
    _call_registered("annotate_statement", kind, generator, node, target, value)


def debug_collect_start(semantic: Any, level: Any, addr_level: Any):
    return _call_required(
        "debugger", "debug_collect_start", semantic, level, addr_level
    )


def debug_collect_end(semantic: Any):
    return _call_required("debugger", "debug_collect_end", semantic)


def ascend_debugger_launch_context(
    metadata: Any,
    grid: Any,
    stream: Any,
    launch_metadata: Any,
    kernel_args: Any,
):
    return _call_required(
        "debugger",
        "ascend_launch_context",
        metadata,
        grid,
        stream,
        launch_metadata,
        kernel_args,
    )
