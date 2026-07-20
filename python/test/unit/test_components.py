from types import ModuleType, SimpleNamespace

import pytest

from triton import _components


@pytest.fixture(autouse=True)
def isolated_components(monkeypatch):
    previous = dict(_components._components)
    _components._components.clear()
    monkeypatch.setattr(_components, "_entry_points", lambda name: [])
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


def test_missing_component_has_install_command():
    with pytest.raises(_components.ComponentNotInstalledError) as error:
        _components.load_component("debugger")
    assert "python -m pip install flagtree-debugger" in str(error.value)


def test_entry_point_component_is_loaded_once(monkeypatch):
    component = _component("debugger")
    loads = []

    class EntryPoint:
        value = "test:debugger"

        def load(self):
            loads.append(True)
            return component

    monkeypatch.setattr(
        _components, "_entry_points", lambda name: [EntryPoint()]
    )
    assert _components.load_component("debugger") is component
    assert _components.load_component("debugger") is component
    assert loads == [True]


def test_component_api_mismatch_is_rejected():
    component = _component("debugger", api_version=999)
    with pytest.raises(_components.ComponentCompatibilityError, match="API mismatch"):
        _components.register_component("debugger", component)


def test_compile_and_launch_hooks_follow_priority_and_finalize_in_reverse():
    events = []

    def make(name, priority, hidden_arg):
        def compiler_hook(stage, module, metadata):
            events.append(("compile", name, stage, module, metadata))

        def prepare(metadata, stream, launch_metadata, kernel_args):
            events.append(("prepare", name, stream, launch_metadata, kernel_args))
            return SimpleNamespace(kernel_args=(hidden_arg,))

        def finalize(prepared, error):
            events.append(("finalize", name, prepared.kernel_args, error))

        return _component(
            name,
            priority=priority,
            run_compiler_hook=compiler_hook,
            prepare_kernel_launch=prepare,
            finalize_prepared_launch=finalize,
        )

    _components.register_component("second", make("second", 20, 2))
    _components.register_component("first", make("first", 10, 1))

    metadata = {}
    _components.run_compiler_hook("ttir", "module", metadata)
    prepared = _components.prepare_kernel_launch(
        metadata, 7, {"grid": (1, 1, 1)}, ("x",)
    )
    assert prepared is not None
    assert prepared.kernel_args == (1, 2)
    _components.finalize_prepared_launch(prepared)

    assert [event[:2] for event in events] == [
        ("compile", "first"),
        ("compile", "second"),
        ("prepare", "first"),
        ("prepare", "second"),
        ("finalize", "second"),
        ("finalize", "first"),
    ]


def test_compiled_metadata_component_api_is_checked():
    _components.register_component("debugger", _component("debugger"))
    metadata = {
        "required_components": ["debugger"],
        "component_api_versions": {"debugger": 99},
    }
    with pytest.raises(
        _components.ComponentCompatibilityError,
        match="requires debugger component API 99",
    ):
        _components.ensure_required_components(metadata)
