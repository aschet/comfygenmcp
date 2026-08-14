# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Where the API-format graph comes from.

ComfyUI's `/prompt` endpoint is stateless: every run must carry the whole graph,
and there is no "run the workflow named X" call. So a caller always names a
source, and we produce the graph:

  <name>            a workflow saved in ComfyUI, converted from UI to API format
  history:<id>      the graph the frontend posted on a past run
  image:<path>      the `prompt` chunk ComfyUI embeds in every PNG it saves
  <path.json>       a file exported with Workflow > Export (API)

Only the first needs converting. ComfyUI serves its saved workflows over
`/userdata` in UI format, and the UI -> API conversion (widget ordering,
control_after_generate, bypassed nodes, subgraph expansion) lives in ComfyUI's
frontend JavaScript, not the backend -- so we borrow comfy-cli's converter
rather than reimplement it. The other sources are already API format.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote

import httpx
from comfy_cli.workflow_to_api import convert_ui_to_api, is_api_format

from . import workflow as wf
from .comfy import ComfyClient

Graph = wf.Graph

# How many empty widgets to try when a workflow saved no prompt to match on.
_MAX_PROMPT_CANDIDATES = 8

# What a model file is called, for telling one apart from an ordinary choice.
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx")


class SourceError(RuntimeError):
    """The workflow source could not be resolved."""


class Resolved(NamedTuple):
    """A runnable graph and where it came from.

    `ui` is the UI-format graph, when the source had one. It is only safe to
    embed in the generated PNG after `patched_ui` has rewritten this
    generation's prompt and seed into it: as fetched, it still describes
    whatever was last saved in ComfyUI.
    """

    graph: Graph
    source: str
    ui: Any = None


async def list_saved(client: ComfyClient) -> list[str]:
    """Names of the workflows ComfyUI has saved on disk."""
    try:
        async with httpx.AsyncClient(base_url=client.base_url, timeout=30.0) as http:
            resp = await http.get("/userdata", params={"dir": "workflows"})
            resp.raise_for_status()
            return [n for n in resp.json() if isinstance(n, str)]
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not list ComfyUI workflows: {exc}") from exc


async def _object_info(client: ComfyClient) -> dict[str, Any]:
    """Node schemas, needed to map widget values onto named inputs.

    Fetched fresh each time. It is a couple of megabytes, but that costs tens of
    milliseconds against a generation measured in seconds, and it means newly
    installed custom nodes are picked up without restarting anything.
    """
    async with httpx.AsyncClient(base_url=client.base_url, timeout=120.0) as http:
        resp = await http.get("/object_info")
        resp.raise_for_status()
        schemas: dict[str, Any] = resp.json()
        return schemas


def _missing_models(graph: Graph, schemas: dict[str, Any]) -> list[str]:
    """Model files a graph names that ComfyUI does not actually have.

    ComfyUI advertises the installed files as the options of a combo input, so
    a literal value absent from that list is a file that is not there.
    """
    missing = []
    for node in graph.values():
        required = schemas.get(str(node.get("class_type")), {}).get("input", {}).get("required", {})
        for key, value in (node.get("inputs") or {}).items():
            if not isinstance(value, str):
                continue
            options = _combo_options(required.get(key))
            if options is None or value in options:
                continue
            # An empty combo is ambiguous -- either nothing of that kind is
            # installed, or the node fills its own choices in at runtime, as
            # CustomCombo does. Only a value naming a file settles it.
            if options or value.lower().endswith(MODEL_SUFFIXES):
                missing.append(value)
    return missing


def _combo_options(spec: Any) -> list[Any] | None:
    """Return the choices a combo input offers, across both shapes ComfyUI emits.

    Older nodes inline the list -- ["a.safetensors", ...] -- while newer ones
    say ["COMBO", {"options": [...]}]. Both appear in the same schema, so
    understanding only one silently skips half the checks.

    An empty list is an answer, not a shrug: a loader whose combo offers
    nothing has no files of that kind installed, and every value it names is
    missing. None means this input is not a combo at all.
    """
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        return options if isinstance(options, list) else None
    return None


async def check_saved(client: ComfyClient) -> list[tuple[str, str | None]]:
    """Report each saved workflow as (name, problem), problem None if runnable.

    Listing a name the user cannot actually generate with just invites a failed
    call, so this converts each workflow and checks both that a prompt node can
    be found and that the models it names are installed.
    """
    schemas = await _object_info(client)
    results: list[tuple[str, str | None]] = []
    for name in sorted(await list_saved(client)):
        try:
            found = await _from_saved(client, name)
        except SourceError as exc:
            results.append((name, f"could not be read ({exc})"))
            continue

        missing = _missing_models(found.graph, schemas)
        if missing:
            listed = ", ".join(sorted(set(missing))[:4])
            results.append((name, f"needs models that are not installed: {listed}"))
        elif wf.find_prompt_node(found.graph) is None:
            results.append((name, "no prompt node found; title one 'MCP:prompt' in ComfyUI"))
        else:
            results.append((name, None))
    return results


def _match_name(name: str, available: list[str]) -> str | None:
    """Resolve a workflow name, tolerating case and a missing .json suffix.

    People say "krea2" for a file saved as "Krea2.json", and a model passes
    through whatever the user typed. Exact matches win, so two workflows
    differing only in case stay distinguishable.
    """
    for candidate in (name, f"{name}.json"):
        if candidate in available:
            return candidate

    wanted = name.removesuffix(".json").casefold()
    loose = [a for a in available if a.removesuffix(".json").casefold() == wanted]
    return loose[0] if len(loose) == 1 else None


async def _from_saved(client: ComfyClient, name: str) -> Resolved:
    """Fetch a workflow ComfyUI has saved and convert it to API format.

    ComfyUI stores workflows in UI format and has no endpoint that converts
    them, so we use comfy-cli's converter -- the same code path `comfy run`
    uses, which handles subgraph expansion, bypassed nodes, reroutes and
    control_after_generate companions.
    """
    available = await list_saved(client)
    match = _match_name(name, available)
    if match is None:
        raise SourceError(
            f"ComfyUI has no saved workflow named {name!r}. Available: "
            + ", ".join(sorted(n.removesuffix(".json") for n in available))
        )

    try:
        async with httpx.AsyncClient(base_url=client.base_url, timeout=60.0) as http:
            resp = await http.get(f"/userdata/{quote(f'workflows/{match}', safe='')}")
            resp.raise_for_status()
            ui = resp.json()
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not read workflow {match!r}: {exc}") from exc

    if is_api_format(ui):
        # Already API format, so there is no UI graph to embed in the PNG.
        return Resolved(ui, f"saved:{match}")

    try:
        api = _convert(ui, await _object_info(client))
    except Exception as exc:
        raise SourceError(f"Could not convert {match!r} to API format: {exc}") from exc
    return Resolved(api, f"saved:{match}", ui)


def _convert(ui: Any, schemas: dict[str, Any]) -> Graph:
    """Convert a UI graph to API format, promoted subgraph widgets included.

    comfy-cli's converter expands a subgraph by copying its definition's nodes,
    which still hold whatever they were given when the subgraph was authored.
    ComfyUI runs the instance's values instead, so without this the graph we
    submit is not the workflow the user sees -- changing a promoted model
    picker in the editor would have no effect here.
    """
    api: Graph = convert_ui_to_api(ui, schemas)
    for node_id, overrides in _promoted_values(ui).items():
        inputs = api.get(node_id, {}).get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in overrides.items():
            # A linked input is driven by another node; the widget behind it is
            # inert, and overwriting the link would cut the graph in two.
            if key in inputs and not wf.is_link(inputs[key]):
                inputs[key] = value
    return api


def _promoted_values(ui: Any) -> dict[str, dict[str, Any]]:
    """Map each expanded node to the promoted widget values that reach it.

    Node ids follow comfy-cli's expansion, which prefixes an inner node with
    its instance: node 28 inside instance 57 becomes "57:28", and one level
    deeper "57:12:28".
    """
    definitions = {str(sg.get("id")): sg for sg in ui.get("definitions", {}).get("subgraphs", [])}
    out: dict[str, dict[str, Any]] = {}

    def walk(nodes: Any, prefix: str) -> None:
        for node in nodes or []:
            definition = definitions.get(str(node.get("type")))
            if definition is None:
                continue
            flat = f"{prefix}{node.get('id')}"
            _instance_values(node, definition, flat, out)
            walk(definition.get("nodes"), f"{flat}:")

    walk(ui.get("nodes"), "")
    return out


def _instance_values(node: Any, definition: Any, flat: str, out: dict[str, dict[str, Any]]) -> None:
    """Collect one instance's widget values, keyed by the inner node they feed.

    `widgets_values` is positional over the instance's widget-bearing inputs,
    and the definition records which internal input each one lands on -- so the
    values need no guessing at widget order, unlike the node schemas.
    """
    values = node.get("widgets_values")
    if not isinstance(values, list) or not values:
        return

    widget_inputs = [i for i in node.get("inputs") or [] if isinstance(i, dict) and i.get("widget")]
    named = {
        inp.get("name"): values[position]
        for position, inp in enumerate(widget_inputs)
        if position < len(values) and inp.get("link") is None
    }
    if not named:
        return

    links = {ln.get("id"): ln for ln in definition.get("links") or [] if isinstance(ln, dict)}
    inner = {n.get("id"): n for n in definition.get("nodes") or [] if isinstance(n, dict)}

    for in_def in definition.get("inputs") or []:
        if not isinstance(in_def, dict) or in_def.get("name") not in named:
            continue
        for link_id in in_def.get("linkIds") or []:
            link = links.get(link_id)
            if not isinstance(link, dict):
                continue
            target = inner.get(link.get("target_id"))
            slot = link.get("target_slot")
            slots = (target or {}).get("inputs") or []
            if target is None or not isinstance(slot, int) or slot >= len(slots):
                continue
            key = slots[slot].get("name")
            if key:
                out.setdefault(f"{flat}:{target.get('id')}", {})[key] = named[in_def["name"]]


def _replace_widget_values(node_list: Any, swaps: list[tuple[Any, Any]]) -> int:
    """Swap literal widget values in a list of UI nodes. Returns hits made."""
    hits = 0
    for node in node_list or []:
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue
        for index, current in enumerate(values):
            for old, new in swaps:
                if current == old:
                    values[index] = new
                    hits += 1
    return hits


async def patched_ui(client: ComfyClient, found: Resolved, prompt: str, seed: int) -> Any | None:
    """Return the UI graph with this generation's prompt and seed written in.

    ComfyUI copies this into the PNG's `workflow` chunk. Embedding the graph
    unpatched would label the image with whatever was last saved in ComfyUI
    rather than what produced it, so this rewrites the widget values -- and
    then proves the rewrite by converting the result back to API format and
    checking it yields the same prompt and seed. Returns None when that cannot
    be shown, because no workflow chunk beats a misleading one.

    Widgets are matched by their previous value rather than by position: the
    layout depends on widget ordering, control_after_generate companions and
    promoted subgraph inputs, and guessing at those is how you silently
    corrupt someone's workflow.
    """
    if found.ui is None:
        return None

    prompt_node = wf.find_prompt_node(found.graph)
    if prompt_node is None:
        return None
    key = wf._prompt_key(found.graph[prompt_node])
    was_prompt = found.graph[prompt_node]["inputs"][key] if key else None
    if not isinstance(was_prompt, str):
        return None

    seed_swaps: list[tuple[Any, Any]] = [
        (found.graph[node_id]["inputs"][seed_key], seed)
        for node_id, seed_key in wf.find_seed_nodes(found.graph)
    ]
    schemas = await _object_info(client)

    for candidate in _prompt_candidates(found.ui, was_prompt, prompt, seed_swaps):
        if await _matches(candidate, prompt_node, prompt, seed, schemas):
            return candidate
    return None


def _prompt_candidates(
    ui: Any, was_prompt: str, prompt: str, seed_swaps: list[tuple[Any, Any]]
) -> Iterator[Any]:
    """Yield UI graphs to try, each with the prompt written somewhere plausible.

    A workflow saved with a non-empty prompt gives one obvious candidate: swap
    that exact string. Saved with an empty prompt there is nothing distinctive
    to match, so every empty string widget is a candidate in turn -- the caller
    checks which one actually lands in the prompt node.
    """
    if was_prompt:
        candidate = copy.deepcopy(ui)
        hit = _replace_widget_values(_all_nodes(candidate), [(was_prompt, prompt), *seed_swaps])
        if hit and _patch_subgraph_instances(
            candidate, prompt, seed_swaps[0][1] if seed_swaps else 0
        ):
            yield candidate
        return

    for index in range(_MAX_PROMPT_CANDIDATES):
        candidate = copy.deepcopy(ui)
        _replace_widget_values(_all_nodes(candidate), seed_swaps)
        if not _fill_nth_empty_string(_all_nodes(candidate), index, prompt):
            return
        if _patch_subgraph_instances(candidate, prompt, seed_swaps[0][1] if seed_swaps else 0):
            yield candidate


def _patch_subgraph_instances(ui: Any, prompt: str, seed: int) -> bool:
    """Write prompt and seed into promoted subgraph widgets.

    A subgraph instance carries its own copy of the promoted inputs, and that
    copy -- not the inner node -- is what ComfyUI shows and runs. Patching only
    the inner nodes leaves the editor displaying the previous prompt.

    Instance `widgets_values` line up positionally with the definition's
    `inputs`, so the prompt is the sole STRING input and the seed the sole INT
    called seed. Returns False when that identification is ambiguous, since a
    wrong guess writes the prompt into some unrelated widget.
    """
    definitions = {str(sg.get("id")): sg for sg in ui.get("definitions", {}).get("subgraphs", [])}
    for node in ui.get("nodes") or []:
        definition = definitions.get(str(node.get("type")))
        values = node.get("widgets_values")
        if definition is None or not isinstance(values, list):
            continue

        if not values:
            # Nothing is promoted onto this instance, so the inner nodes carry
            # the prompt and seed by themselves and are already patched.
            continue

        inputs = definition.get("inputs") or []
        strings = [i for i, inp in enumerate(inputs) if inp.get("type") == "STRING"]
        seeds = [
            i
            for i, inp in enumerate(inputs)
            if inp.get("type") == "INT"
            and "seed" in f"{inp.get('name', '')}{inp.get('label', '')}".lower()
        ]
        if len(strings) > 1 or len(seeds) > 1:
            return False  # Ambiguous: writing into the wrong widget is worse.

        for indices, value in ((strings, prompt), (seeds, seed)):
            if indices and indices[0] < len(values):
                values[indices[0]] = value
    return True


def _all_nodes(ui: Any) -> list[Any]:
    """Every node in a UI graph, including those inside subgraph definitions."""
    nodes = list(ui.get("nodes") or [])
    for subgraph in ui.get("definitions", {}).get("subgraphs", []):
        nodes.extend(subgraph.get("nodes") or [])
    return nodes


def _fill_nth_empty_string(node_list: list[Any], wanted: int, value: str) -> bool:
    seen = 0
    for node in node_list:
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue
        for index, current in enumerate(values):
            if current == "":
                if seen == wanted:
                    values[index] = value
                    return True
                seen += 1
    return False


async def _matches(
    candidate: Any, prompt_node: str, prompt: str, seed: int, schemas: dict[str, Any]
) -> bool:
    """Whether this UI graph really does describe the generation we ran.

    The prompt has to land on the node we patched, named here rather than
    detected again: some workflows are only recognisable while their prompt box
    is still blank, so re-running detection on a filled-in graph would reject
    the very candidate that got it right. Node ids survive the conversion, so
    naming one costs nothing.

    Converting to API format is only a check: the graph that gets embedded is
    the original UI graph with widget values swapped, never anything
    reconstructed from the conversion.
    """
    try:
        roundtrip = _convert(candidate, schemas)
        node = roundtrip.get(prompt_node)
        key = wf._prompt_key(node) if node else None
        if node is None or not key or node["inputs"][key] != prompt:
            return False
        return all(roundtrip[n]["inputs"][k] == seed for n, k in wf.find_seed_nodes(roundtrip))
    except Exception:
        return False


async def _from_history(client: ComfyClient, prompt_id: str | None) -> Resolved:
    try:
        async with httpx.AsyncClient(base_url=client.base_url, timeout=30.0) as http:
            resp = await http.get("/history")
            resp.raise_for_status()
            history = resp.json()
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not read ComfyUI history: {exc}") from exc

    if prompt_id:
        entry = history.get(prompt_id)
        if entry is None:
            raise SourceError(f"No history entry with prompt id {prompt_id!r}.")
        graph = _graph_of(entry)
        if graph is None:
            raise SourceError(f"History entry {prompt_id!r} carries no prompt graph.")
        return Resolved(graph, f"history:{prompt_id}")

    # Newest first: prompt[0] is ComfyUI's monotonically increasing queue number.
    ranked = sorted(
        ((pid, e) for pid, e in history.items()),
        key=lambda kv: _queue_number(kv[1]),
        reverse=True,
    )
    for pid, entry in ranked:
        graph = _graph_of(entry)
        if graph is not None and _has_images(entry):
            return Resolved(graph, f"history:{pid}")
    for pid, entry in ranked:
        graph = _graph_of(entry)
        if graph is not None:
            return Resolved(graph, f"history:{pid}")

    raise SourceError(
        "ComfyUI's history is empty, so there is no workflow to pick up. Run "
        "your workflow once in the ComfyUI web UI, then try again. (History is "
        "cleared when ComfyUI restarts.)"
    )


def _queue_number(entry: dict[str, Any]) -> int:
    prompt = entry.get("prompt")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return prompt[0]
    return -1


def _graph_of(entry: dict[str, Any]) -> Graph | None:
    """History stores prompt as [number, id, graph, extra_data, outputs]."""
    prompt = entry.get("prompt")
    if isinstance(prompt, list) and len(prompt) > 2 and isinstance(prompt[2], dict):
        return prompt[2]
    return None


def _has_images(entry: dict[str, Any]) -> bool:
    return any(o.get("images") for o in (entry.get("outputs") or {}).values())


def _from_image(path: Path) -> Resolved:
    if not path.exists():
        raise SourceError(f"Image not found: {path}")
    try:
        from PIL import Image

        info = Image.open(path).info
    except Exception as exc:
        raise SourceError(f"Could not read {path}: {exc}") from exc

    raw = info.get("prompt")
    if not raw:
        raise SourceError(
            f"{path.name} has no embedded ComfyUI prompt. Only PNGs written by "
            "ComfyUI's SaveImage carry one -- images that were re-encoded or "
            "exported from an editor lose it."
        )
    try:
        return Resolved(json.loads(raw), f"image:{path}")
    except json.JSONDecodeError as exc:
        raise SourceError(f"Embedded prompt in {path.name} is not valid JSON: {exc}") from exc


def _from_file(path: Path) -> Resolved:
    if not path.exists():
        raise SourceError(f"Workflow file not found: {path}")
    try:
        return Resolved(json.loads(path.read_text()), f"file:{path}")
    except json.JSONDecodeError as exc:
        raise SourceError(f"Workflow file is not valid JSON: {exc}") from exc


async def resolve(spec: str, client: ComfyClient) -> Resolved:
    """Turn a workflow spec into a validated API graph plus a label for it.

    The spec is always used as given and never silently substituted: if you
    name a workflow or file that cannot be read, that is an error, not a reason
    to generate from some other graph.
    """
    if spec.startswith("history:"):
        found = await _from_history(client, spec.split(":", 1)[1])
    elif spec.startswith("image:"):
        found = _from_image(Path(spec.split(":", 1)[1]).expanduser())
    elif spec.startswith("saved:"):
        found = await _from_saved(client, spec.split(":", 1)[1])
    else:
        path = Path(spec).expanduser()
        # Anything written as a path is a path: a missing file must report
        # itself as missing, not get retried as a workflow name. A bare word
        # like "Krea2" is a name, and so is "Krea2.json" if no such file exists.
        written_as_path = "/" in spec or spec.startswith("~") or path.is_absolute()
        if path.suffix.lower() == ".png":
            found = _from_image(path)
        elif written_as_path or path.exists():
            found = _from_file(path)
        else:
            found = await _from_saved(client, spec)

    return found._replace(graph=wf.load(found.graph))
