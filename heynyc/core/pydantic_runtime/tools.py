from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jsonschema import Draft202012Validator
from pydantic_ai import (
    ModelRetry,
    RunContext,
)
from pydantic_ai.capabilities import (
    Capability,
)
from pydantic_ai.tools import Tool as PydanticTool

from heynyc.core.registry import Registry
from heynyc.core.tools.base import ResidentFact, Tool, ToolContext

_MISSING = object()


def _fact_leaves(value: object, path: str) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _fact_leaves(
                child,
                f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}",
            )
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _fact_leaves(child, f"{path}/{index}")
        ]
    return [(path, value)]


def _pointer_value(value: object, pointer: str) -> object:
    if not pointer.startswith("/"):
        return _MISSING
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdecimal():
            index = int(token)
            if index >= len(value):
                return _MISSING
            value = value[index]
        else:
            return _MISSING
    return value


def _scoped_fact_leaves(
    args: dict[str, object],
    scopes: Sequence[str],
) -> list[tuple[str, object]]:
    return [
        leaf
        for scope in scopes
        if (value := _pointer_value(args, scope)) is not _MISSING
        for leaf in _fact_leaves(value, scope)
    ]


def _resident_fact_errors(
    args: dict[str, object],
    ctx: ToolContext,
    scopes: Sequence[str],
) -> list[str]:
    return [
        path
        for path, value in _scoped_fact_leaves(args, scopes)
        if (fact := ctx.resident_facts.get(path)) is None
        or type(fact.value) is not type(value)
        or fact.value != value
    ]


def adapt_tool(tool: Tool) -> PydanticTool:
    """Wrap one existing HeyNYC tool without changing its handler or schema."""
    schema = tool._input_schema()
    validator = Draft202012Validator(schema)

    def validate(ctx: RunContext[ToolContext], **kwargs: object) -> None:
        errors = sorted(
            validator.iter_errors(kwargs), key=lambda error: list(error.path)
        )
        if errors:
            raise ModelRetry(f"Invalid arguments for {tool.name}: {errors[0].message}")
        unsupported = _resident_fact_errors(
            kwargs,
            ctx.deps,
            tool.resident_fact_scope,
        )
        if unsupported:
            paths = ", ".join(unsupported)
            raise ModelRetry(
                f"Resident evidence is missing or differs for {paths}. "
                f"Omit unknown optional fields or call confirm_{tool.name}_facts "
                "with the exact profile for resident confirmation."
            )

    async def invoke(ctx: RunContext[ToolContext], **kwargs: object) -> str:
        return await tool.handler(dict(kwargs), ctx.deps)

    adapted = PydanticTool.from_schema(
        invoke,
        name=tool.name,
        description=tool.description,
        json_schema=schema,
        takes_ctx=True,
        args_validator=validate,
    )
    adapted.requires_approval = tool.requires_approval
    adapted.strict = tool.strict
    if tool.resident_fact_scope:

        def expose_after_confirmation(ctx: RunContext[ToolContext], definition: Any):
            has_scoped_facts = any(
                any(
                    path == scope or path.startswith(f"{scope}/")
                    for scope in tool.resident_fact_scope
                )
                for path in ctx.deps.resident_facts
            )
            return definition if has_scoped_facts else None

        adapted.prepare = expose_after_confirmation
    adapted.metadata = {
        "title": tool.title or tool.name,
        "readOnlyHint": tool.read_only,
        "destructiveHint": tool.destructive,
        "idempotentHint": tool.idempotent,
        "openWorldHint": tool.open_world,
        "heynyc_module": tool.module,
    }
    return adapted


def resident_fact_confirmation_tool(tool: Tool) -> Tool:
    """Reuse a governed tool's schema for native structured fact confirmation."""
    if not tool.read_only or tool.destructive or not tool.idempotent:
        raise ValueError(
            "Resident fact confirmation can only wrap read-only idempotent tools"
        )

    async def confirm(args: dict, ctx: ToolContext) -> str:
        source_turn_id = f"turn-{len(ctx.user_turns)}"
        for path, value in _scoped_fact_leaves(args, tool.resident_fact_scope):
            ctx.resident_facts[path] = ResidentFact(
                value=value,
                source_turn_id=source_turn_id,
                status="confirmed",
            )
        return await tool.handler(args, ctx)

    return Tool(
        name=f"confirm_{tool.name}_facts",
        description=(
            f"Use after the resident provides a profile and asks to run {tool.name}. "
            f"{tool.name} is enabled but hidden until this review is approved. "
            "Once its required fields are supported by the conversation, use this "
            "confirmation immediately. Do not delay for optional fields; omit unknown "
            "optional values. Include only exact resident-provided or confirmed facts. "
            "This opens the exact structured facts for resident approval and runs the "
            "requested read-only check after approval."
        ),
        parameters=tool.parameters,
        handler=confirm,
        requires_approval=True,
        title=f"Confirm resident facts for {tool.title or tool.name}",
        module=tool.module,
    )


def build_module_capabilities(
    registry: Registry,
    tools: dict[str, Tool],
) -> tuple[list[PydanticTool], list[Capability[ToolContext]]]:
    """Derive deferred runtime capabilities from authoritative module manifests."""
    modules = {module.name: module for module in registry.modules}

    def root_name(module_name: str) -> str:
        seen: set[str] = set()
        module = modules[module_name]
        while module.parent and module.parent in modules and module.name not in seen:
            seen.add(module.name)
            module = modules[module.parent]
        return module.name

    governed = {
        tool.name: tool
        for tool in tools.values()
        if tool.resident_fact_scope
    }
    confirmation_for = {
        f"confirm_{name}_facts": name
        for name in governed
    }
    module_tools: dict[str, list[PydanticTool]] = {}
    governed_tools: dict[tuple[str, str], list[PydanticTool]] = {}
    shared_tools: list[PydanticTool] = []
    for tool in tools.values():
        if tool.name in governed:
            continue
        adapted = adapt_tool(tool)
        if tool.module in modules:
            root = root_name(tool.module)
            workflow = confirmation_for.get(tool.name)
            if workflow:
                governed_tools.setdefault((root, workflow), []).append(adapted)
            else:
                module_tools.setdefault(root, []).append(adapted)
        else:
            shared_tools.append(adapted)

    descendants: dict[str, list] = {}
    for module in registry.modules:
        descendants.setdefault(root_name(module.name), []).append(module)

    capabilities: list[Capability[ToolContext]] = []
    for module in registry.modules:
        if module.parent:
            continue
        instructions = "\n\n".join(
            member.prompt
            for member in descendants[module.name]
            if member.prompt.strip()
        )
        available_tools = module_tools.get(module.name, ())
        availability = (
            "Enabled module action tools: "
            + ", ".join(f"`{tool.name}`" for tool in available_tools)
            + "."
            if available_tools
            else "This capability has no module-specific action tools enabled."
        )
        availability += (
            " Other workflows may be available through the deferred capability catalog. "
            "Do not collect inputs for or claim to perform an action unless its tool is loaded."
        )
        instructions = "\n\n".join(part for part in (instructions, availability) if part)
        capabilities.append(
            Capability(
                id=module.name,
                description=module.description or f"NYC {module.category} help",
                instructions=instructions,
                tools=available_tools,
                defer_loading=True,
            )
        )
    for (module_name, tool_name), available_tools in governed_tools.items():
        tool = governed[tool_name]
        capability_id = f"{module_name}-{tool_name.replace('_', '-')}"
        purpose = tool.description.partition(". ")[0].rstrip(".")
        capabilities.append(
            Capability(
                id=capability_id,
                description=(
                    f"{purpose}. Use when the resident requests this workflow, asks for "
                    "its result, or accepts an offer to use it. If the resident requested "
                    "or accepted this workflow in a prior turn and the current turn "
                    "supplies or completes its required inputs, load this capability now. "
                    "Load before collecting its inputs; do not load merely to offer it."
                ),
                instructions=(
                    f"The resident explicitly asked for or accepted {tool.title or tool.name}. "
                    "Gather only the schema's required resident facts, a few at a time. "
                    "Omit unknown optional facts. Once the required facts are present, use "
                    "the confirmation tool; after approval it runs the governed read-only check."
                ),
                tools=available_tools,
                defer_loading=True,
            )
        )
    return shared_tools, capabilities
