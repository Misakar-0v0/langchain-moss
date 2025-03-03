from typing import Optional, Dict, Any, List
from ghostos_container import Container, Provider, provide
from ghostos_moss import (
    DefaultMOSSProvider, DefaultModulesProvider, MossCompiler, MossRuntime,
    PyContext,
)

__all__ = [
    'get_container',
    'set_container',
    'compile_moss_runtime',
    'get_moss_compiler',
    'bootstrap_container',
    'Container', 'Provider', provide,

]

_container: Optional[Container] = None


def get_container() -> Container:
    global _container
    if _container is None:
        container = Container("langchain-moss")
        container.register(DefaultModulesProvider())
        container.register(DefaultMOSSProvider())
        _container = container
        _container.bootstrap()
    return _container


def set_container(container: Container) -> None:
    global _container
    _container = container


def bootstrap_container(container: Container) -> None:
    container.bootstrap()
    set_container(container)


def get_moss_compiler() -> MossCompiler:
    return get_container().force_fetch(MossCompiler)


def compile_moss_runtime(
        modulename: str,
        *,
        providers: List[Provider] = None,
        bindings: Dict[Any, Any] = None,
        local_values: Dict[str, Any] = None,
        injections: Dict[str, Any] = None,
        pycontext: PyContext = None,
) -> MossRuntime:
    compiler = get_moss_compiler()
    if pycontext:
        compiler = compiler.join_context(pycontext)

    if providers:
        for p in providers:
            compiler.register(p)
    if bindings:
        for key, val in bindings.items():
            compiler.bind(key, val)

    if local_values:
        compiler = compiler.with_locals(**local_values)

    if injections:
        compiler = compiler.injects(**injections)

    with compiler:
        return compiler.compile(modulename)
