# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""MCP server: generate an image on ComfyUI and hand it back inline."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import random
from dataclasses import dataclass
from typing import Annotated, Any

import mcp.types as types
from mcp.server import MCPServer
from PIL import Image
from pydantic import Field

from . import __version__, sources
from . import workflow as wf
from .comfy import ComfyClient, ComfyError

# ComfyUI seeds are uint64, but the UI caps its randomiser here and some nodes
# choke on anything wider.
MAX_SEED = 2**32 - 1

# Claude scales images down to a 1568 px long edge before the model sees them,
# so anything larger is detail discarded in transit.
MAX_IMAGE_EDGE = 1568

# Claude Desktop rejects images over 1 MB. Base64 inflates by 4/3, so the raw
# bytes must stay under ~750 KB; 700 KB leaves margin for the JSON envelope.
MAX_IMAGE_BYTES = 700_000


@dataclass
class Settings:
    """Process-wide configuration, populated from the command line."""

    comfy_url: str = "http://127.0.0.1:8188"
    timeout: float = 300.0


settings = Settings()
mcp = MCPServer(
    "comfygen",
    version=__version__,
    instructions=(
        "Generates images by running a workflow on a ComfyUI server.\n\n"
        "ComfyUI must already have a saved workflow that produces an image. "
        "This server reuses it, substituting only the prompt and seed -- "
        "everything else is whatever that workflow defines and cannot be "
        "changed here.\n\n"
        "When the user asks for an image, produce one rather than asking them to "
        "fill in details: pass their description as the prompt, and use the "
        "workflow they named. Call list_workflows only when they named none or "
        "the name was not found, then pick one ready workflow. Ask only when "
        "they have named no subject at all.\n\n"
        "One generation per request. Never run the same prompt through several "
        "workflows to compare -- each call ties up the user's GPU for many "
        "seconds, and they asked for one image.\n\n"
        "Always tell the user the seed a result came back with, so they can "
        "reproduce it. Use comfy_status to diagnose failures."
    ),
)


@mcp.tool(structured_output=False)
async def list_workflows() -> str:
    """List the workflows saved in ComfyUI, ready ones first.

    Call this instead of asking the user which workflows they have -- they
    expect you to look.

    The reply separates workflows that can generate from those that cannot,
    because the models they reference are not installed or no prompt node was
    found. Only use the ready ones; naming another will fail.
    """
    client = ComfyClient(settings.comfy_url, timeout=180.0)
    try:
        checked = await sources.check_saved(client)
    except sources.SourceError as exc:
        return str(exc)
    if not checked:
        return "ComfyUI has no saved workflows."

    usable = [f"  {n.removesuffix('.json')}" for n, problem in checked if problem is None]
    broken = [f"  {n.removesuffix('.json')} -- {problem}" for n, problem in checked if problem]

    lines = []
    if usable:
        lines += ["Ready to generate with:", *usable]
    else:
        lines += ["No workflow here can currently generate an image."]
    if broken:
        lines += ["", "Present but not usable:", *broken]
    return "\n".join(lines)


def _fit(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale `image` down so its longest edge is `long_edge`."""
    scale = long_edge / max(image.size)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _prepare(image: Image.Image) -> Image.Image:
    """Convert to a mode WebP can save, preserving any alpha channel.

    Unlike JPEG, WebP has an RGBA mode, so a cutout workflow's transparency
    survives untouched rather than being flattened onto an assumed background.
    """
    if image.mode in ("RGB", "RGBA"):
        return image
    if image.mode in ("LA", "PA") or "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def _webp(image: Image.Image, quality: int) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=6)
    return buffer


def _encode(data: bytes) -> tuple[str, str, tuple[int, int]]:
    """Encode the image small enough for the client to accept.

    Chat clients reject oversized images outright, so this keeps trying --
    lower quality, then fewer pixels -- rather than attempting once and hoping.

    Returns (base64 payload, mime type, the size before any downscaling).
    """
    opened = Image.open(io.BytesIO(data))
    opened.load()
    original = opened.size

    image = opened if max(opened.size) <= MAX_IMAGE_EDGE else _fit(opened, MAX_IMAGE_EDGE)
    image = _prepare(image)

    for quality in (88, 75, 60):
        buffer = _webp(image, quality)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            return base64.b64encode(buffer.getvalue()).decode(), "image/webp", original

    # Quality alone was not enough -- give up pixels instead.
    for _ in range(6):
        image = _fit(image, max(64, int(max(image.size) * 0.75)))
        buffer = _webp(image, 75)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            break
    return base64.b64encode(buffer.getvalue()).decode(), "image/webp", original


# structured_output=False keeps the return value as raw content blocks, so the
# image reaches the client as an image rather than serialised JSON.
@mcp.tool(structured_output=False)
async def generate_image(
    prompt: Annotated[
        str,
        Field(description="What to draw, in plain language."),
    ],
    workflow: Annotated[
        str,
        Field(
            description=(
                "Name of a workflow saved in ComfyUI, exactly as list_workflows "
                "returns it. If you do not have a name yet, call list_workflows and "
                "use the first ready one rather than asking the user to choose."
            )
        ),
    ],
    seed: Annotated[
        int | None,
        Field(
            description="Reuse a seed from an earlier result to reproduce it. Omit for a new one."
        ),
    ] = None,
    fetch_image: Annotated[
        bool,
        Field(
            description=(
                "Whether to include the image in the reply, not just its seed and "
                "URL. Default true. Pass false if this client cannot render an "
                "inline image and would otherwise receive it as raw base64 text."
            )
        ),
    ] = True,
) -> list[types.TextContent | types.ImageContent]:
    """Generate an image on ComfyUI and return it inline.

    Returns the seed used and a URL to the full-resolution original on
    ComfyUI, and -- unless fetch_image is false -- the image itself.

    Use exactly the workflow name the user asked for. Similar names are
    different workflows -- "Krea2" and "Krea2_App.app" are not interchangeable.

    Clients usually collapse tool results, so the image is easy to miss. Pass
    the seed and the URL on in your reply: they survive the collapse, and the
    URL is how the user sees the picture when no image is returned. Write the
    URL bare so the client turns it into a link -- code formatting or
    backticks around it make it unclickable.

    Fails if the workflow needs an input image or has no prompt node to patch.
    """
    client = ComfyClient(settings.comfy_url, timeout=settings.timeout)
    found = await sources.resolve(workflow, client)
    used_seed = random.randint(0, MAX_SEED) if seed is None else int(seed)

    patched = wf.patch(found.graph, prompt=prompt, seed=used_seed)
    # Embedded only when it provably describes this generation; see patched_ui.
    embed = await sources.patched_ui(client, found, prompt, used_seed)
    refs = await client.generate(patched, embed)

    if not refs:
        raise ComfyError(
            "The workflow ran but produced no images. Make sure it ends in a "
            "SaveImage or PreviewImage node."
        )

    blocks: list[types.TextContent | types.ImageContent] = []
    # Lead with a plain statement that this finished, and name the workflow it
    # used: without that, a model can read a bare metadata line as an
    # unsatisfying result and try the remaining workflows in turn.
    notes = [f"Done: generated with {workflow}", f"seed {used_seed}"]

    for ref in refs:
        if ref.filename:
            notes.append(client.view_url(ref))
        if not fetch_image:
            continue

        raw = await client.fetch(ref)
        payload, mime, original = _encode(raw)
        if max(original) > MAX_IMAGE_EDGE:
            notes.append(f"downscaled from {original[0]}x{original[1]}")
        blocks.append(
            types.ImageContent(
                type="image",
                data=payload,
                mime_type=mime,
                # The picture is the point of the call, not a detail the model
                # should merely reason over. Clients that honour annotations
                # surface it rather than tucking it inside the tool result.
                annotations=types.Annotations(audience=["user", "assistant"], priority=1.0),
            )
        )

    blocks.insert(0, types.TextContent(type="text", text=" · ".join(notes)))
    return blocks


@mcp.tool(structured_output=False)
async def comfy_status(
    workflow: Annotated[
        str | None,
        Field(description="Also check this workflow, naming the nodes it would patch."),
    ] = None,
) -> str:
    """Check the ComfyUI connection, and optionally inspect one workflow.

    Reports whether ComfyUI is reachable and what hardware it has. Given a
    workflow name, also reports which nodes would receive the prompt and seed --
    use that to diagnose a generation that failed or patched the wrong node.
    """
    report: dict[str, Any] = {"comfy_url": settings.comfy_url}

    client = ComfyClient(settings.comfy_url, timeout=30.0)
    try:
        stats = await client.stats()
        system = stats.get("system", {})
        report["comfyui"] = {
            "reachable": True,
            "version": system.get("comfyui_version"),
            "python": system.get("python_version", "").split()[0] or None,
        }
        report["devices"] = [
            {"name": d.get("name"), "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1)}
            for d in stats.get("devices", [])
        ]
    except Exception as exc:
        report["comfyui"] = {"reachable": False, "error": str(exc)}

    if workflow is not None:
        try:
            found = await sources.resolve(workflow, client)
            report["workflow_resolved_from"] = found.source
            report["workflow"] = wf.describe(found.graph)
        except Exception as exc:
            report["workflow"] = {"error": str(exc)}

    return json.dumps(report, indent=2)


def main() -> None:
    """Run the MCP server. Entry point for the `comfygenmcp` command."""
    parser = argparse.ArgumentParser(prog="comfygenmcp", description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for clients that launch the server, http to listen on a port.",
    )
    parser.add_argument(
        "--comfy-url",
        default="http://127.0.0.1:8188",
        help="Address of the ComfyUI server to generate on.",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:8000",
        metavar="HOST:PORT",
        help="Address this server listens on, with --transport http.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for a generation before giving up and interrupting.",
    )
    args = parser.parse_args()

    # httpx logs a line per request; on stdio that just clutters the client log.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings.comfy_url = args.comfy_url
    settings.timeout = args.timeout

    if args.transport == "http":
        host, _, port = args.listen.rpartition(":")
        if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
            parser.error(f"--listen must be HOST:PORT with a port in 1-65535, got {args.listen!r}")
        mcp.run(transport="streamable-http", host=host, port=int(port))
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
