# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Locate and patch the interesting nodes in a ComfyUI API-format workflow.

We never build a graph ourselves -- we only find the nodes carrying the prompt
text and the sampler seed, and overwrite those two values. That keeps this
server model-agnostic: whatever checkpoint, custom nodes or loaders the
workflow uses are left exactly as they are.

API format looks like::

    {"3": {"class_type": "KSampler",
           "inputs": {"seed": 0, "positive": ["6", 0], ...},
           "_meta": {"title": "KSampler"}}, ...}

A value like ``["6", 0]`` is a link: output slot 0 of node "6".
"""

from __future__ import annotations

import copy
import re
from typing import Any

Graph = dict[str, dict[str, Any]]

# Explicit opt-in: rename a node's title in ComfyUI to one of these and it wins
# over every heuristic below. Use it when auto-detection picks the wrong node.
PROMPT_TITLES = {"mcp:prompt", "mcp_prompt", "prompt", "positive prompt", "positive"}

SAMPLER_RE = re.compile(r"KSampler|SamplerCustom|SamplerSelect", re.I)
SEED_KEYS = ("seed", "noise_seed")

# CLIPTextEncode calls it "text"; the string primitives that feed magic-prompt
# chains call it "value".
PROMPT_KEYS = ("text", "value")

# Where a sampler's conditioning comes in. SamplerCustomAdvanced has no
# "positive" -- it takes a guider that carries the conditioning instead.
CONDITIONING_INPUTS = ("positive", "guider")


class WorkflowError(RuntimeError):
    """The workflow could not be understood or patched."""


def is_link(value: Any) -> bool:
    """Report whether `value` is a ``[node_id, slot]`` link rather than a literal."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def load(raw: Any) -> Graph:
    """Validate that `raw` is an API-format workflow, not a UI export."""
    if not isinstance(raw, dict):
        raise WorkflowError("Workflow JSON must be an object.")
    if "nodes" in raw and "links" in raw:
        raise WorkflowError(
            "This is a UI-format workflow. Re-export it with "
            "'Workflow > Export (API)' -- the API format is a flat map of "
            "node id -> {class_type, inputs}."
        )
    for node_id, node in raw.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise WorkflowError(
                f"Node {node_id!r} has no 'class_type'; this does not look like "
                "an API-format workflow export."
            )
    if not raw:
        raise WorkflowError("Workflow is empty.")
    return raw


def _title(node: dict[str, Any]) -> str:
    return str(node.get("_meta", {}).get("title", "")).strip().lower()


def _prompt_key(node: dict[str, Any]) -> str | None:
    """Return the input holding this node's prompt string, if it has one."""
    inputs = node.get("inputs", {})
    return next((k for k in PROMPT_KEYS if isinstance(inputs.get(k), str)), None)


def _by_title(graph: Graph, titles: set[str]) -> str | None:
    for node_id, node in graph.items():
        if _title(node) in titles and _prompt_key(node):
            return node_id
    return None


def _samplers(graph: Graph) -> list[str]:
    return [nid for nid, n in graph.items() if SAMPLER_RE.search(str(n.get("class_type", "")))]


def _holds_prompt(inputs: dict[str, Any], blanks: bool) -> bool:
    """Whether this node's inputs are the prompt slot we are looking for.

    Normally only a literal "text" counts, never "value": the string
    primitives that call it "value" also hold the system templates driving
    magic-prompt chains, and overwriting one of those would quietly wreck the
    workflow.

    `blanks` relaxes that to empty "value" widgets only. A workflow that hands
    its prompt to a switch or a rewriting model leaves the user's box blank and
    fills the templates in, so an empty one is the box and a filled one never
    is -- which is the same distinction, just drawn where it still holds.
    """
    if isinstance(inputs.get("text"), str):
        return True
    return blanks and inputs.get("value") == ""


def _walk_to_text(
    graph: Graph, start: str, blanks: bool = False, seen: set[str] | None = None
) -> str | None:
    """Follow links upstream from `start` until we hit a literal prompt input.

    Handles chains like CLIPTextEncode -> FluxGuidance -> ConditioningCombine,
    where the sampler's `positive` input is several hops from the actual text.
    """
    seen = seen if seen is not None else set()
    if start in seen or start not in graph:
        return None
    seen.add(start)

    inputs = graph[start].get("inputs", {})
    if _holds_prompt(inputs, blanks):
        return start

    for value in inputs.values():
        if is_link(value):
            found = _walk_to_text(graph, str(value[0]), blanks, seen)
            if found:
                return found
    return None


def find_prompt_node(graph: Graph) -> str | None:
    """Return the id of the node whose text input is the prompt.

    Resolution order: explicit title marker, upstream from the sampler's
    positive conditioning input, the same walk again accepting a blank string
    widget, then a lone CLIPTextEncode.
    """
    explicit = _by_title(graph, PROMPT_TITLES)
    if explicit:
        return explicit

    # Strict first, across every sampler, before relaxing anywhere: a real
    # text input on one branch beats a blank widget on another.
    for blanks in (False, True):
        for sampler_id in _samplers(graph):
            for slot in CONDITIONING_INPUTS:
                link = graph[sampler_id].get("inputs", {}).get(slot)
                if is_link(link):
                    found = _walk_to_text(graph, str(link[0]), blanks)
                    if found:
                        return found

    # No sampler wired the usual way -- fall back to a single text-encode node.
    encoders = [
        nid
        for nid, n in graph.items()
        if "CLIPTextEncode" in str(n.get("class_type", "")) and _prompt_key(n)
    ]
    return encoders[0] if len(encoders) == 1 else None


def find_seed_nodes(graph: Graph) -> list[tuple[str, str]]:
    """Return every ``(node_id, input_name)`` holding a numeric seed."""
    found = []
    for node_id, node in graph.items():
        for key in SEED_KEYS:
            value = node.get("inputs", {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append((node_id, key))
    return found


def find_output_nodes(graph: Graph) -> list[str]:
    """Return the nodes that emit images, i.e. what makes the graph worth running."""
    return [
        nid
        for nid, n in graph.items()
        if str(n.get("class_type", "")) in ("SaveImage", "PreviewImage", "SaveImageWebsocket")
    ]


def describe(graph: Graph) -> dict[str, Any]:
    """Summarise what the patcher found: prompt node, seed inputs, outputs."""
    prompt_id = find_prompt_node(graph)

    def label(node_id: str | None) -> Any:
        if node_id is None:
            return None
        node = graph[node_id]
        return {
            "node": node_id,
            "class_type": node.get("class_type"),
            "title": node.get("_meta", {}).get("title"),
        }

    return {
        "node_count": len(graph),
        "prompt_node": label(prompt_id),
        "seed_inputs": [f"{nid}.{key}" for nid, key in find_seed_nodes(graph)],
        "output_nodes": [label(nid) for nid in find_output_nodes(graph)],
    }


def patch(
    graph: Graph,
    prompt: str,
    seed: int | None = None,
) -> Graph:
    """Return a copy of `graph` with the requested values written in."""
    out = copy.deepcopy(graph)

    prompt_id = find_prompt_node(out)
    if prompt_id is None:
        raise WorkflowError(
            "Could not find the prompt node. Open the workflow in ComfyUI, rename "
            "the positive prompt node's title to 'MCP:prompt' (right-click > Title), "
            "save it, and try again."
        )
    key = _prompt_key(out[prompt_id])
    assert key is not None  # find_prompt_node only returns nodes that have one
    out[prompt_id]["inputs"][key] = prompt

    if seed is not None:
        seed_inputs = find_seed_nodes(out)
        if not seed_inputs:
            raise WorkflowError("This workflow has no seed input to set.")
        for node_id, key in seed_inputs:
            out[node_id]["inputs"][key] = seed

    return out
