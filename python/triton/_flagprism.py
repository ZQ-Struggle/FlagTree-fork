"""Integration boundary between FlagTree core and FlagPrism.

Compiler and runtime code call this module instead of importing profiler or
debugger implementations. Callbacks are no-ops until a compatible FlagPrism
component registers itself.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any


HOST_API_VERSION = (2, 0)
HOST_CAPABILITIES = frozenset(
    {
        "compiler.dialects.v1",
        "compiler.events.v1",
        "compiler.options.v1",
        "frontend.statement_events.v1",
        "language.debug_collect.v1",
        "runtime.launch_context.v1",
    }
)
_COMPONENT_MODULES = {
    "debugger": "flagtree.debugger",
    "profiler": "flagtree.profiler",
}
_lock = RLock()
_components: dict[str, Any] = {}


class ComponentNotInstalledError(ModuleNotFoundError):
    pass


class ComponentCompatibilityError(ImportError):
    pass


@dataclass(frozen=True)
class HostInfo:
    api_version: tuple[int, int]
    capabilities: frozenset[str]
    core_version: str


@dataclass(frozen=True)
class CompilerEvent:
    phase: str
    ir_kind: str
    backend: str
    module: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StatementResult:
    name: str | None
    value: Any


@dataclass(frozen=True)
class StatementEvent:
    kind: str
    source: str
    statement_id: int
    builder: Any
    results: tuple[StatementResult, ...]


@dataclass(frozen=True)
class LaunchEvent:
    backend: str
    metadata: Any
    grid: Any
    stream: Any
    launch_metadata: Any
    kernel_args: Any


def get_host_info() -> HostInfo:
    from triton import __version__

    return HostInfo(HOST_API_VERSION, HOST_CAPABILITIES, str(__version__))


def _normalize_api_version(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, 0
    parts = value.split(".", 1) if isinstance(value, str) else tuple(value)
    if len(parts) not in {1, 2}:
        raise ComponentCompatibilityError(
            f"invalid FlagPrism API version {value!r}"
        )
    try:
        return int(parts[0]), int(parts[1]) if len(parts) == 2 else 0
    except (TypeError, ValueError) as error:
        raise ComponentCompatibilityError(
            f"invalid FlagPrism API version {value!r}"
        ) from error


def _validate_component(name: str, component: Any) -> Any:
    if name not in _COMPONENT_MODULES:
        raise ComponentCompatibilityError(
            f"unsupported FlagPrism component {name!r}"
        )
    actual_name = str(getattr(component, "name", ""))
    if actual_name != name:
        raise ComponentCompatibilityError(
            f"FlagPrism component {name!r} returned {actual_name!r}"
        )
    api_version = _normalize_api_version(
        getattr(component, "api_version", (-1, 0))
    )
    if api_version[0] != HOST_API_VERSION[0] or api_version > HOST_API_VERSION:
        raise ComponentCompatibilityError(
            f"FlagPrism API mismatch: host={HOST_API_VERSION}, "
            f"component={api_version}"
        )
    required = frozenset(
        str(capability)
        for capability in getattr(component, "required_capabilities", ())
    )
    if missing := required - HOST_CAPABILITIES:
        raise ComponentCompatibilityError(
            "FlagPrism component requires unsupported capabilities: "
            + ", ".join(sorted(missing))
        )
    return component


def register_component(name: str, component: Any) -> Any:
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
    try:
        module_name = _COMPONENT_MODULES[name]
    except KeyError as error:
        raise ComponentCompatibilityError(
            f"unsupported FlagPrism component {name!r}"
        ) from error
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
                f"FlagPrism {name} is not included in this FlagTree build"
            ) from None
        return register_component(name, getattr(module, "component", module))


def _callbacks(method: str) -> tuple[Any, ...]:
    with _lock:
        components = tuple(_components.values())
    return tuple(
        callback
        for component in components
        if callable(callback := getattr(component, method, None))
    )


def _call_required(name: str, method: str, *args: Any):
    component = load_component(name)
    callback = getattr(component, method, None)
    if not callable(callback):
        raise ComponentCompatibilityError(
            f"FlagPrism {name} does not implement {method!r}"
        )
    return callback(*args)


def load_dialects(context: Any) -> None:
    for callback in _callbacks("load_dialects"):
        callback(context)


def apply_compile_options(options: dict[str, Any]) -> None:
    for callback in _callbacks("apply_compile_options"):
        callback(options)


def _metadata_backend(metadata: dict[str, Any]) -> str:
    target = metadata.get("target")
    backend = target.get("backend") if isinstance(target, dict) else getattr(
        target, "backend", None
    )
    return str(backend or "").lower()


def emit_compiler_event(
    *, phase: str, ir_kind: str, module: Any, metadata: dict[str, Any]
) -> None:
    callbacks = _callbacks("on_compiler_event")
    if not callbacks:
        return
    event = CompilerEvent(
        phase=str(phase),
        ir_kind=str(ir_kind),
        backend=_metadata_backend(metadata),
        module=module,
        metadata=metadata,
    )
    for callback in callbacks:
        callback(event)


def _statement_source(generator: Any, node: ast.AST) -> str:
    source = None
    if hasattr(node, "lineno"):
        try:
            source = ast.get_source_segment(generator.jit_fn.src, node)
        except Exception:
            pass
    if source is None and hasattr(ast, "unparse"):
        try:
            source = ast.unparse(node)
        except Exception:
            pass
    return "" if source is None else " ".join(source.strip().split())


def _statement_id(generator: Any, node: ast.AST) -> int:
    line = int(generator.begin_line + getattr(node, "lineno", 0))
    column = int(getattr(node, "col_offset", 0))
    return max(0, min(line * 1000 + min(column, 999), (1 << 31) - 1))


def _assignment_results(
    target: ast.AST, value: Any
) -> tuple[StatementResult, ...]:
    if isinstance(target, ast.Name):
        return (StatementResult(target.id, value),)
    if isinstance(target, ast.Tuple):
        values = getattr(value, "values", ())
        results = []
        for child, child_value in zip(target.elts, values):
            results.extend(_assignment_results(child, child_value))
        return tuple(results)
    return ()


def emit_statement_event(
    kind: str, generator: Any, node: Any, target: Any, value: Any
) -> None:
    callbacks = _callbacks("on_statement_event")
    if not callbacks:
        return
    if kind == "assignment":
        results = () if target is None else _assignment_results(target, value)
    elif kind == "expression":
        results = (StatementResult(None, value),)
    else:
        raise ValueError(f"unsupported statement event {kind!r}")
    event = StatementEvent(
        kind=kind,
        source=_statement_source(generator, node),
        statement_id=_statement_id(generator, node),
        builder=generator.builder,
        results=results,
    )
    for callback in callbacks:
        callback(event)


def debug_collect_start(semantic: Any, level: Any, addr_level: Any):
    return _call_required(
        "debugger", "debug_collect_start", semantic, level, addr_level
    )


def debug_collect_end(semantic: Any):
    return _call_required("debugger", "debug_collect_end", semantic)


def debugger_launch_context(
    backend: str,
    metadata: Any,
    grid: Any,
    stream: Any,
    launch_metadata: Any,
    kernel_args: Any,
):
    return _call_required(
        "debugger",
        "launch_context",
        LaunchEvent(
            backend=str(backend).lower(),
            metadata=metadata,
            grid=grid,
            stream=stream,
            launch_metadata=launch_metadata,
            kernel_args=kernel_args,
        ),
    )
