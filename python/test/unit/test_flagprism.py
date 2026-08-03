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
        "api_version": _flagprism.COMPONENT_API_VERSION,
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


def test_optional_hooks_are_noops_until_a_component_registers():
    events = []
    component = _component(
        "debugger",
        apply_compile_options=lambda options: options.update(
            instrumentation_mode="debug"
        ),
        run_compiler_hook=lambda stage, module, metadata: events.append(
            (stage, module, metadata)
        ),
        annotate_statement=lambda *args: events.append(args),
    )
    options = {}
    _flagprism.apply_compile_options(options)
    assert options == {}

    _flagprism.register_component("debugger", component)
    _flagprism.apply_compile_options(options)
    _flagprism.run_compiler_hook("ttir", "module", {"key": "value"})
    _flagprism.annotate_statement("expression", "generator", "node", None, "value")

    assert options == {"instrumentation_mode": "debug"}
    assert events == [
        ("ttir", "module", {"key": "value"}),
        ("expression", "generator", "node", None, "value"),
    ]


def test_required_ascend_launch_context_is_forwarded():
    context = nullcontext((123,))
    component = _component(
        "debugger",
        ascend_launch_context=lambda *args: context,
    )
    _flagprism.register_component("debugger", component)

    result = _flagprism.ascend_debugger_launch_context(
        "metadata", (1, 2, 3), "stream", "launch_metadata", ("arg",)
    )
    assert result is context


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

    expected_native = native_root / (
        "libproton" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
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
        assert sorted(path.name for path in native_root.glob("libproton*")) == [
            expected_native.name
        ]
        assert (flagtree_root / "debugger").is_dir()
        assert (flagtree_root / "profiler").is_dir()
    else:
        assert not list(native_root.glob("libproton*"))
        assert not (flagtree_root / "debugger").exists()
        assert not (flagtree_root / "profiler").exists()
