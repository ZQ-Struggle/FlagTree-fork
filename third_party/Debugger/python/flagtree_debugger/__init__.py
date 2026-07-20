from __future__ import annotations

import sys
from importlib import metadata

from . import compiler as _compiler_module
from . import native as _native_module
from . import runtime as _runtime_module
from .api import *  # noqa: F403
from .api import __all__


try:
    __version__ = metadata.version("flagtree-debugger")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"


class _DebuggerComponent:
    name = "debugger"
    api_version = 1
    version = __version__
    core_version_series = "3.5"
    priority = 100

    @staticmethod
    def module():
        return sys.modules[__name__]

    @staticmethod
    def register() -> None:
        return None

    @staticmethod
    def is_active() -> bool:
        return is_active()  # noqa: F405

    @staticmethod
    def load_dialects(context) -> None:
        from .compiler import load_dialects

        load_dialects(context)

    @staticmethod
    def run_compiler_hook(stage: str, module, metadata: dict) -> None:
        from .compiler import run_compiler_hook

        run_compiler_hook(stage, module, metadata)

    @staticmethod
    def update_compile_metadata(metadata: dict) -> None:
        from .compiler import update_compile_metadata

        update_compile_metadata(metadata)

    @staticmethod
    def set_instrumentation_mode(mode: str) -> None:
        from .compiler import set_instrumentation_mode

        set_instrumentation_mode(mode)

    @staticmethod
    def needs_launch_metadata(metadata) -> bool:
        enabled = (
            bool(metadata.get("debug_enabled", False))
            if isinstance(metadata, dict)
            else bool(getattr(metadata, "debug_enabled", False))
        )
        return enabled or is_active()  # noqa: F405

    @staticmethod
    def prepare_kernel_launch(metadata, stream, launch_metadata, kernel_args):
        if isinstance(metadata, dict):
            hidden_arg = bool(metadata.get("debug_launch_hidden_arg", False))
            records = int(metadata.get("debug_records_per_instance", 0) or 0)
        else:
            hidden_arg = bool(getattr(metadata, "debug_launch_hidden_arg", False))
            records = int(getattr(metadata, "debug_records_per_instance", 0) or 0)
        if hidden_arg and records > 0:
            return prepare_kernel_launch(  # noqa: F405
                metadata, stream, launch_metadata, kernel_args
            )
        return prepare_metadata_only_kernel_launch(  # noqa: F405
            metadata, stream, launch_metadata, kernel_args
        )

    @staticmethod
    def finalize_prepared_launch(prepared, error) -> None:
        finalize_prepared_launch(prepared, error)  # noqa: F405

    @staticmethod
    def cache_key() -> str:
        from .api import current_compile_config, is_active

        state = repr(current_compile_config()) if is_active() else "inactive"
        return f"{__version__}:{state}"


component = _DebuggerComponent()
