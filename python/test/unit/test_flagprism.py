import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace

import pytest

from triton import _flagprism


def _load_build_helper():
    path = (
        Path(__file__).resolve().parents[3]
        / "third_party"
        / "FlagPrism"
        / "python"
        / "flagprism_build.py"
    )
    spec = importlib.util.spec_from_file_location("_test_flagprism_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_helper = _load_build_helper()


@pytest.fixture(autouse=True)
def isolated_components():
    previous = dict(_flagprism._components)
    _flagprism._components.clear()
    try:
        yield
    finally:
        _flagprism._components.clear()
        _flagprism._components.update(previous)


def _component(name, **attributes):
    values = {
        "name": name,
        "api_version": _flagprism.HOST_API_VERSION,
    }
    values.update(attributes)
    return SimpleNamespace(**values)


def test_public_component_modules_use_flagtree_namespace():
    assert _flagprism._COMPONENT_MODULES == {
        "debugger": "flagtree.debugger",
        "profiler": "flagtree.profiler",
    }


@pytest.mark.parametrize(
    "component",
    ("debugger", "profiler"),
)
def test_missing_component_has_build_instruction(
    monkeypatch, component
):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_flagprism, "import_module", missing)
    with pytest.raises(_flagprism.ComponentNotInstalledError) as error:
        _flagprism.load_component(component)
    assert "TRITON_BUILD_FLAGPRISM=ON" in str(error.value)


@pytest.mark.parametrize(
    ("value", "enabled"),
    ((None, True), ("ON", True), ("OFF", False)),
)
def test_build_helper_uses_unified_switch(monkeypatch, tmp_path, value, enabled):
    if value is None:
        monkeypatch.delenv("TRITON_BUILD_FLAGPRISM", raising=False)
    else:
        monkeypatch.setenv("TRITON_BUILD_FLAGPRISM", value)

    config = _build_helper.FlagPrismBuildConfig.from_environment(tmp_path)
    assert config.enabled is enabled


def test_known_component_is_loaded_once(monkeypatch):
    component = _component("debugger")
    module = SimpleNamespace(component=component)
    loads = []

    def import_component(name):
        loads.append(name)
        return module

    monkeypatch.setattr(_flagprism, "import_module", import_component)
    assert _flagprism.load_component("debugger") is component
    assert _flagprism.load_component("debugger") is component
    assert loads == ["flagtree.debugger"]


def test_component_api_mismatch_is_rejected():
    component = _component("debugger", api_version=999)
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_legacy_callback_api_is_rejected():
    component = _component("debugger", api_version=1)
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_component_api_minor_newer_than_host_is_rejected():
    component = _component("debugger", api_version=(2, 1))
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_component_compatibility_uses_capabilities_not_core_series():
    component = _component(
        "debugger",
        api_version=(2, 0),
        core_version_series="0.0",
        required_capabilities={"compiler.events.v1"},
    )
    assert _flagprism.register_component("debugger", component) is component


def test_missing_host_capability_is_rejected():
    component = _component(
        "debugger", required_capabilities={"runtime.future_adapter.v1"}
    )
    with pytest.raises(
        _flagprism.ComponentCompatibilityError,
        match="runtime.future_adapter.v1",
    ):
        _flagprism.register_component("debugger", component)


def test_optional_hooks_are_noops_until_a_component_registers():
    events = []
    component = _component(
        "debugger",
        apply_compile_options=lambda options: options.update(
            instrumentation_mode="debug"
        ),
        on_compiler_event=events.append,
        on_statement_event=events.append,
    )
    options = {}
    _flagprism.apply_compile_options(options)
    assert options == {}

    _flagprism.register_component("debugger", component)
    _flagprism.apply_compile_options(options)
    metadata = {
        "key": "value",
        "target": SimpleNamespace(backend="cuda"),
    }
    _flagprism.emit_compiler_event(
        phase="post_override",
        ir_kind="ttir",
        module="module",
        metadata=metadata,
    )
    node = ast.parse("result = value").body[0]
    generator = SimpleNamespace(
        begin_line=20,
        builder="builder",
        jit_fn=SimpleNamespace(src="result = value"),
    )
    _flagprism.emit_statement_event(
        "assignment", generator, node, node.targets[0], "value"
    )

    assert options == {"instrumentation_mode": "debug"}
    compiler_event, statement_event = events
    assert compiler_event == _flagprism.CompilerEvent(
        phase="post_override",
        ir_kind="ttir",
        backend="cuda",
        module="module",
        metadata=metadata,
    )
    assert statement_event.kind == "assignment"
    assert statement_event.source == "result = value"
    assert statement_event.statement_id == 21000
    assert statement_event.builder == "builder"
    assert statement_event.results == (
        _flagprism.StatementResult(name="result", value="value"),
    )


def test_statement_normalization_is_skipped_without_a_consumer():
    _flagprism.register_component("profiler", _component("profiler"))
    _flagprism.emit_statement_event(
        "assignment", "not-a-generator", "not-an-ast-node", None, "value"
    )


def test_required_backend_neutral_launch_context_is_forwarded():
    context = nullcontext((123,))
    events = []

    def launch_context(event):
        events.append(event)
        return context

    component = _component(
        "debugger",
        launch_context=launch_context,
    )
    _flagprism.register_component("debugger", component)

    result = _flagprism.debugger_launch_context(
        "CANN", "metadata", (1, 2, 3), "stream", "launch_metadata", ("arg",)
    )
    assert result is context
    assert events == [
        _flagprism.LaunchEvent(
            backend="cann",
            metadata="metadata",
            grid=(1, 2, 3),
            stream="stream",
            launch_metadata="launch_metadata",
            kernel_args=("arg",),
        )
    ]


def test_unknown_components_are_rejected():
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="unsupported"):
        _flagprism.load_component("custom")


@pytest.mark.parametrize("enabled", (True, False))
def test_build_tree_cleanup_prevents_split_wheel_artifacts(tmp_path, enabled):
    build_lib = tmp_path / "build-lib"
    triton_root = build_lib / "triton"
    flagtree_root = build_lib / "flagtree"
    native_root = triton_root / "_C"
    cache_root = triton_root / "__pycache__"
    config = _build_helper.FlagPrismBuildConfig(
        enabled=enabled,
        relative_root=Path("third_party/FlagPrism"),
        root=tmp_path / "FlagPrism",
    )

    for path in (
        triton_root / "debugger" / "old.py",
        build_lib / "flagtree_debugger" / "old.py",
        flagtree_root / "debugger" / "old.py",
        native_root / "libproton.so",
        cache_root / "_components.cpython-311.pyc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    config.prepare_build_tree(str(build_lib))
    assert not (triton_root / "debugger").exists()
    assert not (flagtree_root / "debugger").exists()
    assert not list(native_root.glob("libproton*"))

    expected_native = flagtree_root / "profiler" / (
        "_native" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    )
    if enabled:
        for path in (
            flagtree_root / "debugger" / "__init__.py",
            flagtree_root / "profiler" / "__init__.py",
            expected_native,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"current")

    # build_py can copy these stale source-tree files after CMake completes.
    for path in (
        native_root / "libproton.so",
        cache_root / "_components.cpython-311.pyc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    config.finalize_build_tree(str(build_lib))
    assert not (cache_root / "_components.cpython-311.pyc").exists()
    if enabled:
        assert expected_native.is_file()
        assert not list(native_root.glob("libproton*"))
        assert (flagtree_root / "debugger").is_dir()
        assert (flagtree_root / "profiler").is_dir()
    else:
        assert not list(native_root.glob("libproton*"))
        assert not (flagtree_root / "debugger").exists()
        assert not (flagtree_root / "profiler").exists()
