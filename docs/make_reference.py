"""Writes docs/reference.md from the package itself. It holds every name
the package root exports, with its signature and its docstring, and
every key of the two config files, with its type and its default.

The test suite fails when docs/reference.md is not what this writes, so
run it after changing a public signature, a docstring or a config key.

    python docs/make_reference.py
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import types
import typing
from pathlib import Path

import annotated_types
from pydantic import BaseModel

import cil_regionalization as cilreg
from cil_regionalization.config import Config
from cil_regionalization.pipelines.montecarlo import MonteCarloPipelineConfig

OUT = Path(__file__).resolve().parent / "reference.md"

# Every name of cilreg.__all__ is in one group, in the order a reader
# meets it.
GROUPS = {
    "Published weights": [
        "fetch_weights",
        "list_cached",
        "clear_cache",
        "WeightsArtifact",
        "apply_weights",
        "SourceUnitPolicies",
        "read_netcdf_grid",
        "OutputSchema",
        "SourceUnits",
        "GRID_CELLS",
    ],
    "Statistics over the members": [
        "summarize_samples",
        "window_means",
        "pooled_statistics",
        "mixture_quantile",
        "read_model_weights",
        "ModelWeights",
        "UnweightedModelWeightsWarning",
    ],
    "Warming levels": [
        "read_gmst",
        "GMSTTable",
        "assign_warming_levels",
        "Period",
        "WarmingLevels",
        "LevelWindow",
    ],
    "Monte Carlo and delta-method trees": [
        "read_netcdf_leaf",
        "read_gradient_leaf",
        "window_gradient",
        "aggregate_gradient",
        "delta_method",
    ],
    "Generating weights": [
        "generate_weights",
        "load_config",
        "Config",
        "WeightsResult",
        "SumToOneError",
    ],
}

CONFIGS = [
    (
        "The weights config",
        "`cilreg run <config.toml>` and `generate_weights` read it.",
        Config,
    ),
    (
        "The pipeline config",
        "`cilreg pipeline <config.toml>` reads it.",
        MonteCarloPipelineConfig,
    ),
]

WIDTH = 79


def _definition(obj) -> ast.AST:
    """The node that defines ``obj``, parsed from its source file."""
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(obj)))
    return ast.parse(source).body[0]


def _annotation(node) -> str:
    """An annotation as written, without the quotes of a forward
    reference."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def _parameter(arg: ast.arg, default) -> str:
    if arg.annotation is None:
        return arg.arg + ("" if default is None else "=" + ast.unparse(default))
    text = f"{arg.arg}: {_annotation(arg.annotation)}"
    return text + ("" if default is None else " = " + ast.unparse(default))


def _parameters(args: ast.arguments) -> list[str]:
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = []
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        out.append(_parameter(arg, default))
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            out.append("/")
    if args.vararg is not None:
        out.append("*" + _parameter(args.vararg, None))
    elif args.kwonlyargs:
        out.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(_parameter(arg, default))
    if args.kwarg is not None:
        out.append("**" + _parameter(args.kwarg, None))
    return out


def _call(name: str, parameters: list[str], returns: str = "") -> str:
    """A signature on one line, or one parameter a line if it is long."""
    tail = f" -> {returns}" if returns else ""
    line = f"{name}({', '.join(parameters)}){tail}"
    if len(line) <= WIDTH or not parameters:
        return line
    body = "".join(f"    {p},\n" for p in parameters if p not in ("*", "/"))
    for marker in ("*", "/"):
        if marker in parameters:
            at = parameters.index(marker)
            lines = body.splitlines(keepends=True)
            lines.insert(at, f"    {marker},\n")
            body = "".join(lines)
    return f"{name}(\n{body}){tail}"


def _decorators(node) -> list[str]:
    return [ast.unparse(d) for d in node.decorator_list]


def _function(node: ast.FunctionDef, name: str, method: bool) -> str:
    parameters = _parameters(node.args)
    decorators = _decorators(node)
    if method and "staticmethod" not in decorators:
        parameters = parameters[1:]
    returns = _annotation(node.returns) if node.returns is not None else ""
    return _call(name, parameters, returns)


def _doc(node) -> str:
    return ast.get_docstring(node) or ""


def _block(code: str) -> str:
    return f"```python\n{code}\n```\n"


def _fields(node: ast.ClassDef) -> list[str]:
    """The annotated attributes of a class body, as written."""
    out = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            text = f"{item.target.id}: {_annotation(item.annotation)}"
            if item.value is not None:
                text += f" = {ast.unparse(item.value)}"
            out.append(text)
    return out


def _methods(node: ast.ClassDef, owner: str) -> list[str]:
    out = []
    for item in node.body:
        if not isinstance(item, ast.FunctionDef) or item.name.startswith("_"):
            continue
        decorators = _decorators(item)
        if "property" in decorators:
            returns = f": {_annotation(item.returns)}" if item.returns else ""
            head = f"{owner}.{item.name}{returns}"
        else:
            head = _function(item, f"{owner}.{item.name}", method=True)
            if "classmethod" in decorators:
                head = "@classmethod\n" + head
            elif "staticmethod" in decorators:
                head = "@staticmethod\n" + head
        text = f"#### `{owner}.{item.name}`\n\n{_block(head)}"
        if _doc(item):
            text += f"\n{_doc(item)}\n"
        out.append(text)
    return out


# where the keys of an exported config model are listed
KEYS_OF = {
    "Config": "under [The weights config](#the-weights-config)",
    "SourceUnitPolicies": "under `[policies]` of [The pipeline config]"
    "(#the-pipeline-config), and `apply_weights` takes them as arguments",
}


def _export(name: str, obj) -> str:
    """The reference entry of one exported name."""
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        return f"### `{name}`\n\n{_block(f'{name} = {obj!r}')}"
    node = _definition(obj)
    if inspect.isfunction(obj):
        text = f"### `{name}`\n\n{_block(_function(node, name, method=False))}"
        return text + (f"\n{_doc(node)}\n" if _doc(node) else "")
    bases = ", ".join(
        ast.unparse(b) for b in node.bases if not ast.unparse(b).startswith("_")
    )
    head = f"class {name}" + (f"({bases})" if bases else "")
    if issubclass(obj, BaseModel):
        text = f"### `{name}`\n\n{_block(head)}"
        if _doc(node):
            text += f"\n{_doc(node)}\n"
        return text + f"\nA pydantic model. Its keys are listed {KEYS_OF[name]}.\n"
    init = next(
        (
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    fields = _fields(node)
    if init is not None:
        head += "\n\n" + _function(init, name, method=True)
    elif fields:
        head += ":\n" + "".join(f"    {field}\n" for field in fields).rstrip("\n")
    text = f"### `{name}`\n\n{_block(head)}"
    if _doc(node):
        text += f"\n{_doc(node)}\n"
    for method in _methods(node, name):
        text += f"\n{method}"
    return text


def _type(t) -> str:
    """How a config key's type reads, from the type pydantic holds."""
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin is typing.Literal:
        return " or ".join(json.dumps(a) for a in args)
    if origin in (typing.Union, types.UnionType):
        return " or ".join(_type(a) for a in args if a is not type(None))
    if origin is typing.Annotated:
        return _type(args[0])
    if origin is not None:
        inner = ", ".join("..." if a is Ellipsis else _type(a) for a in args)
        return f"{getattr(origin, '__name__', str(origin))}[{inner}]"
    if t is typing.Any:
        return "any"
    return getattr(t, "__name__", str(t))


def _toml(value) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + "}"
    return repr(value)


def _limits(metadata) -> list[str]:
    out = []
    for item in metadata:
        if isinstance(item, annotated_types.MinLen):
            out.append(
                "not empty" if item.min_length == 1 else f"at least {item.min_length}"
            )
        elif isinstance(item, annotated_types.Gt):
            out.append(f"above {item.gt}")
        elif isinstance(item, annotated_types.Ge):
            out.append(f"at least {item.ge}")
        elif isinstance(item, annotated_types.Lt):
            out.append(f"below {item.lt}")
        elif isinstance(item, annotated_types.Le):
            out.append(f"at most {item.le}")
        else:
            out.append(str(item))
    return out


def _model_in(t):
    """The config model a field holds and whether it holds a list of
    them, or None if it holds a plain value."""
    origin, args = typing.get_origin(t), typing.get_args(t)
    # on Python 3.10 a parametrised list passes for a class
    if origin is None and inspect.isclass(t) and issubclass(t, BaseModel):
        return t, False
    if origin in (typing.Union, types.UnionType):
        models = [m for m in map(_model_in, args) if m is not None]
        return models[0] if len(models) == 1 else None
    if origin is list and len(args) == 1:
        found = _model_in(args[0])
        return (found[0], True) if found is not None else None
    return None


def _sections(model, path: str, many: bool, required: bool):
    yield path, model, many, required
    for name, field in model.model_fields.items():
        found = _model_in(field.annotation)
        if found is not None:
            sub, sub_many = found
            yield from _sections(
                sub, f"{path}.{name}" if path else name, sub_many, field.is_required()
            )


def _config(title: str, what: str, root) -> str:
    text = f"## {title}\n\n{what}\n"
    if _doc(_definition(root)):
        text += f"\n{_doc(_definition(root))}\n"
    for path, model, many, required in _sections(root, "", False, True):
        rows = []
        for name, field in model.model_fields.items():
            if _model_in(field.annotation) is not None:
                continue
            kind = ", ".join([_type(field.annotation), *_limits(field.metadata)])
            if field.is_required():
                default = "required"
            elif field.default_factory is not None:
                default = _toml(field.default_factory())
            else:
                default = _toml(field.default)
            rows.append(f"| `{name}` | {kind} | {default} |")
        if not path:
            if rows:
                text += "\n" + _table(rows)
            continue
        head = f"[[{path}]]" if many else f"[{path}]"
        text += f"\n### `{head}`\n\n"
        if many:
            text += "A list of tables, each with these keys.\n"
        else:
            text += f"{'Required.' if required else 'Can be left out.'}\n"
        if _doc(_definition(model)):
            text += f"\n{_doc(_definition(model))}\n"
        if rows:
            text += "\n" + _table(rows)
    return text


def _table(rows: list[str]) -> str:
    return "| key | type | default |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"


def render() -> str:
    """The text of docs/reference.md. Raises if a name the package root
    exports is in no group, or a group names something it does not
    export."""
    grouped = [name for names in GROUPS.values() for name in names]
    exported = [name for name in cilreg.__all__ if name != "__version__"]
    if sorted(grouped) != sorted(exported):
        raise ValueError(
            f"docs/make_reference.py groups {sorted(set(grouped) - set(exported))} "
            f"that the package does not export and leaves out "
            f"{sorted(set(exported) - set(grouped))} that it does"
        )
    parts = [
        "# Reference\n\n"
        "Written by `docs/make_reference.py` from the package itself. Do not\n"
        "edit it by hand. The names below are what `import cil_regionalization\n"
        "as cilreg` gives. Anything else in the package is internal and may\n"
        "change.\n\n"
        + "".join(
            f"- [{group}](#{group.lower().replace(' ', '-')})\n" for group in GROUPS
        )
        + "".join(
            f"- [{title}](#{title.lower().replace(' ', '-')})\n"
            for title, _, _ in CONFIGS
        )
    ]
    for group, names in GROUPS.items():
        parts.append(f"## {group}\n")
        parts.extend(_export(name, getattr(cilreg, name)) for name in names)
    parts.extend(_config(*config) for config in CONFIGS)
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
