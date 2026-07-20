from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from threading import RLock
from types import ModuleType
from typing import Any, Mapping, Sequence


COMPONENT_API_VERSION = 1
ENTRY_POINT_GROUP = "flagtree.components"


@dataclass(frozen=True)
class ComponentSpec:
    distribution: str
    install_name: str


_SPECS = {
    "debugger": ComponentSpec("flagtree-debugger", "flagtree-debugger"),
    "profiler": ComponentSpec("flagtree-profiler", "flagtree-profiler"),
}


class ComponentNotInstalledError(ModuleNotFoundError):
    pass


class ComponentCompatibilityError(ImportError):
    pass


@dataclass(frozen=True)
class PreparedComponentLaunch:
    kernel_args: tuple[int, ...] = ()
    prepared: tuple[tuple[Any, Any], ...] = ()


_lock = RLock()
_components: dict[str, Any] = {}


def _entry_points(name: str):
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=ENTRY_POINT_GROUP, name=name))
    return [entry for entry in discovered.get(ENTRY_POINT_GROUP, ()) if entry.name == name]


def _missing_message(name: str) -> str:
    spec = _SPECS.get(name, ComponentSpec(f"flagtree-{name}", f"flagtree-{name}"))
    return (
        f"FlagTree {name} is not installed. Install the matching optional component with "
        f"`python -m pip install {spec.install_name}`."
    )


def _validate_component(name: str, component: Any) -> Any:
    actual_name = str(getattr(component, "name", ""))
    if actual_name != name:
        raise ComponentCompatibilityError(
            f"FlagTree component entry point {name!r} returned component {actual_name!r}"
        )
    api_version = int(getattr(component, "api_version", -1))
    if api_version != COMPONENT_API_VERSION:
        raise ComponentCompatibilityError(
            f"FlagTree {name} component API mismatch: core={COMPONENT_API_VERSION}, "
            f"component={api_version}. Install a component wheel matching this FlagTree build."
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
    """Register an already imported component.

    This is public for editable builds and tests. Installed wheels normally use
    the ``flagtree.components`` entry-point group instead.
    """
    component = _validate_component(name, component)
    with _lock:
        current = _components.get(name)
        if current is not None:
            if current is not component:
                raise ComponentCompatibilityError(f"FlagTree component {name!r} is already registered")
            return current
        _components[name] = component
    register = getattr(component, "register", None)
    try:
        if callable(register):
            register()
    except BaseException:
        with _lock:
            if _components.get(name) is component:
                del _components[name]
        raise
    return component


def load_component(name: str, *, required: bool = True) -> Any | None:
    with _lock:
        current = _components.get(name)
        if current is not None:
            return current

        entries = _entry_points(name)
        if not entries:
            if required:
                raise ComponentNotInstalledError(_missing_message(name))
            return None
        if len(entries) != 1:
            providers = ", ".join(sorted(entry.value for entry in entries))
            raise ComponentCompatibilityError(
                f"multiple FlagTree {name} components are installed: {providers}"
            )

        loaded = entries[0].load()
        component = getattr(loaded, "component", loaded)
        return register_component(name, component)


def is_component_available(name: str) -> bool:
    try:
        return load_component(name, required=False) is not None
    except (ImportError, OSError):
        return False


def public_module(name: str) -> ModuleType:
    component = load_component(name)
    module = getattr(component, "module", None)
    if callable(module):
        module = module()
    if not isinstance(module, ModuleType):
        raise ComponentCompatibilityError(
            f"FlagTree component {name!r} did not provide a public Python module"
        )
    return module


def loaded_components() -> Mapping[str, Any]:
    with _lock:
        return dict(_components)


def _ordered_components() -> tuple[Any, ...]:
    components = loaded_components()
    return tuple(
        component
        for _, component in sorted(
            components.items(),
            key=lambda item: (int(getattr(item[1], "priority", 0)), item[0]),
        )
    )


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def required_component_names(metadata: Any) -> tuple[str, ...]:
    names = _metadata_value(metadata, "required_components", ()) or ()
    if isinstance(names, str):
        names = (names,)
    result = {str(name) for name in names}
    # Read caches produced before required_components was introduced.
    if bool(_metadata_value(metadata, "debug_enabled", False)):
        result.add("debugger")
    return tuple(sorted(result))


def ensure_required_components(metadata: Any) -> None:
    versions = _metadata_value(metadata, "component_api_versions", {}) or {}
    for name in required_component_names(metadata):
        component = load_component(name)
        expected = versions.get(name) if isinstance(versions, Mapping) else None
        if expected is not None and int(expected) != int(component.api_version):
            raise ComponentCompatibilityError(
                f"compiled kernel requires {name} component API {expected}, but "
                f"API {component.api_version} is installed"
            )


def load_dialects(context: Any) -> None:
    for component in _ordered_components():
        callback = getattr(component, "load_dialects", None)
        if callable(callback):
            callback(context)


def run_compiler_hook(stage: str, module: Any, metadata: dict[str, Any]) -> None:
    for component in _ordered_components():
        callback = getattr(component, "run_compiler_hook", None)
        if callable(callback):
            callback(stage, module, metadata)


def update_compile_metadata(metadata: dict[str, Any]) -> None:
    for component in _ordered_components():
        callback = getattr(component, "update_compile_metadata", None)
        if callable(callback):
            callback(metadata)


def set_instrumentation_mode(mode: str) -> None:
    for component in _ordered_components():
        callback = getattr(component, "set_instrumentation_mode", None)
        if callable(callback):
            callback(mode)


def needs_launch_metadata(metadata: Any) -> bool:
    ensure_required_components(metadata)
    for component in _ordered_components():
        callback = getattr(component, "needs_launch_metadata", None)
        if callable(callback) and callback(metadata):
            return True
    return False


def prepare_kernel_launch(
    metadata: Any,
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Sequence[Any] | None = None,
) -> PreparedComponentLaunch | None:
    ensure_required_components(metadata)
    prepared_components: list[tuple[Any, Any]] = []
    hidden_args: list[int] = []
    try:
        for component in _ordered_components():
            callback = getattr(component, "prepare_kernel_launch", None)
            if not callable(callback):
                continue
            prepared = callback(
                metadata,
                int(stream),
                launch_metadata,
                tuple(kernel_args or ()),
            )
            if prepared is None:
                continue
            prepared_components.append((component, prepared))
            hidden_args.extend(int(arg) for arg in getattr(prepared, "kernel_args", ()))
    except BaseException as error:
        finalize_prepared_launch(
            PreparedComponentLaunch(tuple(hidden_args), tuple(prepared_components)),
            error,
        )
        raise
    if not prepared_components:
        return None
    return PreparedComponentLaunch(tuple(hidden_args), tuple(prepared_components))


def finalize_prepared_launch(
    prepared: PreparedComponentLaunch | None,
    error: BaseException | None = None,
) -> None:
    if prepared is None:
        return
    first_error: BaseException | None = None
    for component, component_launch in reversed(prepared.prepared):
        callback = getattr(component, "finalize_prepared_launch", None)
        if not callable(callback):
            continue
        try:
            callback(component_launch, error)
        except BaseException as finalize_error:
            if first_error is None:
                first_error = finalize_error
    if first_error is not None and error is None:
        raise first_error


def component_cache_key() -> str:
    keys = []
    for name, component in sorted(loaded_components().items()):
        callback = getattr(component, "cache_key", None)
        value = callback() if callable(callback) else getattr(component, "version", "")
        keys.append(f"{name}:{value}")
    return "|".join(keys)
