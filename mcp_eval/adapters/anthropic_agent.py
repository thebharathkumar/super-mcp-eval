"""Single-shot Claude agent that picks one MCP tool for a natural-language task.

The agent forwards every MCP tool to the Anthropic Messages API as a tool
definition, sets ``tool_choice={"type": "any"}`` so the model is forced to use
one of them, and returns the first ``tool_use`` block it sees along with token
usage and an estimated cost. The eval suite grades on the picked tool name
and the picked arguments; tool execution is deliberately left to callers so
selection accuracy is measured independently of side effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from mcp_eval.adapters.mcp_client import ToolDef

DEFAULT_SYSTEM_PROMPT = (
    "You are an evaluation subject for an MCP tool-selection benchmark. "
    "Use exactly one of the provided tools to accomplish the user's task. "
    "Pick the most direct tool and the smallest set of arguments. "
    "Do not chain tool calls."
)

# Approximate per-million-token prices in USD. Override via the ``pricing``
# argument to AnthropicAgent. These rates are used for cost estimation only.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


@dataclass
class AgentDecision:
    picked_tool: str | None
    picked_args: dict[str, Any] | None
    stop_reason: str | None
    text: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)


class ToolPicker(Protocol):
    """Anything that can pick a tool given a task and a tool list.

    The eval depends on this Protocol rather than a concrete class so tests can
    swap in a deterministic fake without touching the Anthropic SDK.
    """

    model: str

    def pick_tool(self, task: str, tools: list[ToolDef]) -> AgentDecision: ...


def _to_anthropic_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def estimate_cost(
    model: str, tokens_in: int, tokens_out: int, pricing: dict[str, dict[str, float]]
) -> float:
    rates = pricing.get(model) or {"input": 3.0, "output": 15.0}
    return (tokens_in / 1_000_000.0) * rates["input"] + (tokens_out / 1_000_000.0) * rates["output"]


class AnthropicAgent:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        pricing: dict[str, dict[str, float]] | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt
        self.pricing = pricing or DEFAULT_PRICING
        self.max_tokens = max_tokens

    def pick_tool(self, task: str, tools: list[ToolDef]) -> AgentDecision:
        anthropic_tools = _to_anthropic_tools(tools)
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            tools=anthropic_tools,
            tool_choice={"type": "any"},
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": task}],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        picked_tool: str | None = None
        picked_args: dict[str, Any] | None = None
        text_pieces: list[str] = []

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and picked_tool is None:
                picked_tool = block.name
                picked_args = dict(block.input or {})
            elif block_type == "text":
                text_pieces.append(block.text)

        usage = response.usage
        tokens_in = int(getattr(usage, "input_tokens", 0))
        tokens_out = int(getattr(usage, "output_tokens", 0))

        return AgentDecision(
            picked_tool=picked_tool,
            picked_args=picked_args,
            stop_reason=response.stop_reason,
            text="\n".join(text_pieces) if text_pieces else None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out, self.pricing),
            latency_ms=elapsed_ms,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
