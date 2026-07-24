from types import ModuleType, SimpleNamespace

import pytest

from triton import _components


@pytest.fixture(autouse=True)
def isolated_components():
    previous = dict(_components._components)
    _components._components.clear()
    try:
        yield
    finally:
        _components._components.clear()
        _components._components.update(previous)


def _component(name, **attributes):
    values = {
        "name": name,
        "api_version": _components.COMPONENT_API_VERSION,
        "module": lambda: ModuleType(f"test_{name}"),
    }
    values.update(attributes)
    return SimpleNamespace(**values)


def test_missing_component_has_install_command(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_components, "import_module", missing)
    with pytest.raises(_components.ComponentNotInstalledError) as error:
        _components.load_component("debugger")
    assert "python -m pip install flagtree-debugger" in str(error.value)


def test_known_component_module_is_loaded_once(monkeypatch):
    component = _component("debugger")
    module = SimpleNamespace(component=component)
    loads = []

    def import_component(name):
        loads.append(name)
        return module

    monkeypatch.setattr(_components, "import_module", import_component)
    assert _components.load_component("debugger") is component
    assert _components.load_component("debugger") is component
    assert loads == ["flagtree_debugger"]


def test_component_api_mismatch_is_rejected():
    component = _component("debugger", api_version=999)
    with pytest.raises(_components.ComponentCompatibilityError, match="API mismatch"):
        _components.register_component("debugger", component)


def test_debugger_hooks_are_optional_and_direct():
    events = []
    component = _component(
        "debugger",
        apply_compile_options=lambda options: options.update(instrumentation_mode="debug"),
        run_compiler_hook=lambda stage, module, metadata: events.append(
            (stage, module, metadata)
        ),
        annotate_statement=lambda *args: events.append(args),
    )
    options = {}
    _components.apply_compile_options(options)
    assert options == {}

    _components.register_component("debugger", component)
    _components.apply_compile_options(options)
    _components.run_compiler_hook("ttir", "module", {"key": "value"})
    _components.annotate_statement("expression", "generator", "node", None, "value")

    assert options == {"instrumentation_mode": "debug"}
    assert events == [
        ("ttir", "module", {"key": "value"}),
        ("expression", "generator", "node", None, "value"),
    ]


def test_unknown_components_are_rejected():
    with pytest.raises(_components.ComponentCompatibilityError, match="unsupported"):
        _components.load_component("custom")
