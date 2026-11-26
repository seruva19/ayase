"""Cached AST helpers for metric and model catalog introspection."""

import ast
import inspect
import textwrap
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Set, Tuple, Union


_METADATA_MODULE_CACHE: Dict[str, Tuple[int, int, str, ast.Module]] = {}
_METADATA_FUNCTION_CACHE: Dict[str, Tuple[int, int, Dict[int, str]]] = {}


@lru_cache(maxsize=2048)
def _metadata_tree(source: str) -> ast.Module:
    """Parse inspected source without treating comments or prose as executable evidence."""
    try:
        return ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return ast.Module(body=[], type_ignores=[])


def _metadata_module_source(module: Any) -> Tuple[str, ast.Module]:
    """Share parsed literal globals across classes defined in the same source file."""
    path = Path(inspect.getfile(module))
    stat = path.stat()
    key = str(path)
    cached = _METADATA_MODULE_CACHE.get(key)
    if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        return cached[2], cached[3]
    source = inspect.getsource(module)
    tree = _metadata_tree(source)
    _METADATA_MODULE_CACHE[key] = (stat.st_mtime_ns, stat.st_size, source, tree)
    return source, tree


def _metadata_function_source(function: Callable[..., Any]) -> str:
    """Slice cached module lines instead of repeatedly tokenizing every method."""
    module = inspect.getmodule(function)
    if module is None:
        return textwrap.dedent(inspect.getsource(function))
    source, tree = _metadata_module_source(module)
    path = Path(inspect.getfile(module))
    stat = path.stat()
    cached = _METADATA_FUNCTION_CACHE.get(str(path))
    if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
        lines = source.splitlines(keepends=True)
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
                functions[start] = textwrap.dedent("".join(lines[start - 1 : node.end_lineno]))
        cached = (stat.st_mtime_ns, stat.st_size, functions)
        _METADATA_FUNCTION_CACHE[str(path)] = cached
    code = getattr(function, "__code__", None)
    if code is not None and code.co_firstlineno in cached[2]:
        return cached[2][code.co_firstlineno]
    return textwrap.dedent(inspect.getsource(function))


def _metadata_class_tree(cls: type) -> ast.Module:
    """Find top-level/nested classes in the already parsed defining module."""
    module = inspect.getmodule(cls)
    if module is not None and "<locals>" not in cls.__qualname__:
        _, tree = _metadata_module_source(module)
        body = tree.body
        declaration = None
        for name in cls.__qualname__.split("."):
            declaration = next(
                (node for node in body if isinstance(node, ast.ClassDef) and node.name == name),
                None,
            )
            if declaration is None:
                break
            body = declaration.body
        if declaration is not None:
            return ast.Module(body=[copy(declaration)], type_ignores=[])
    tree = copy(_metadata_tree(inspect.getsource(cls)))
    tree.body = [copy(node) for node in tree.body]
    return tree


@lru_cache(maxsize=1024)
def _metric_source_fields_cached(
    source: str,
) -> Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
    """Return statically observed metric reads, writes, and dataset writes.

    Literal field declarations remain separate from observed writes. Local aliases
    are tracked within each function, including literal getattr/setattr calls.
    This is source evidence, not proof that a branch runs or produces a value.
    """
    reads: Set[str] = set()
    writes: Set[str] = set()
    dataset_writes: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.aliases: Set[str] = set()

        def is_metrics(self, node: ast.AST) -> bool:
            return (isinstance(node, ast.Attribute) and node.attr == "quality_metrics") or (
                isinstance(node, ast.Name) and node.id in self.aliases
            )

        def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
            previous = self.aliases
            self.aliases = set()
            for statement in node.body:
                self.visit(statement)
            self.aliases = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            is_alias = self.is_metrics(node.value)
            for target in node.targets:
                self.visit(target)
                if isinstance(target, ast.Name):
                    if is_alias:
                        self.aliases.add(target.id)
                    else:
                        self.aliases.discard(target.id)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            initial = set(self.aliases)
            for statement in node.body:
                self.visit(statement)
            body_aliases = set(self.aliases)
            self.aliases = initial
            for statement in node.orelse:
                self.visit(statement)
            self.aliases |= body_aliases

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if self.is_metrics(node.value):
                if isinstance(node.ctx, ast.Load):
                    reads.add(node.attr)
                elif isinstance(node.ctx, ast.Store):
                    writes.add(node.attr)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if isinstance(node.target, ast.Attribute) and self.is_metrics(node.target.value):
                reads.add(node.target.attr)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr"}
                and len(node.args) >= 2
                and self.is_metrics(node.args[0])
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                target = writes if node.func.id == "setattr" else reads
                target.add(node.args[1].value)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_dataset_metric"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                dataset_writes.add(node.args[0].value)
            self.generic_visit(node)

    Visitor().visit(_metadata_tree(source))
    return frozenset(reads), frozenset(writes), frozenset(dataset_writes)


def _metric_source_fields(source: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return isolated copies of cached source evidence for documentation callers."""
    reads, writes, dataset_writes = _metric_source_fields_cached(source)
    return set(reads), set(writes), set(dataset_writes)
