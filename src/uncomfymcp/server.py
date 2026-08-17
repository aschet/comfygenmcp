# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""MCP server: generate an image on ComfyUI and hand it back inline."""

from __future__ import annotations

import argparse
import base64
import io
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
    "uncomfy",
    version=__version__,
    instructions=(
        "Generates images by running a workflow on a ComfyUI server.\n\n"
        "ComfyUI must already have a saved workflow that produces an image. "
        "This server reuses it, substituting only the prompt and seed -- "
        "everything else is whatever that workflow defines and cannot be "
        "changed here.\n\n"
        "Any request describing a picture counts as asking for an image, however "
        "it's phrased -- a scene, a portrait, an art style or visual effect, not "
        'only a literal "generate/draw/make me an image". Produce one rather '
        "than asking them to fill in details or treating it as a stylistic "
        "description: pass their description as the prompt, and use the "
        "workflow they named. Call list_workflows only when they named none or "
        "the name was not found, then pick one ready workflow. Ask only when "
        "they have named no subject at all.\n\n"
        "One generation per request. Never run the same prompt through several "
        "workflows to compare -- each call ties up the user's GPU for many "
        "seconds, and they asked for one image.\n\n"
        "Always tell the user the seed a result came back with, so they can "
        "reproduce it."
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


def _int_type_not_anyof(schema: dict[str, Any]) -> None:
    """Flatten `int | None` to a plain integer type in the exposed schema.

    Some MCP clients (AnythingLLM among them) show no type at all for a field
    schema'd as `anyOf: [integer, null]`, but render a plain `type: integer`
    fine. Omitting `seed` from `required` already tells a client it need not
    pass one; the schema does not also need to spell out that null would be
    accepted.
    """
    schema.pop("anyOf", None)
    schema["type"] = "integer"


# structured_output=False stops the SDK from auto-deriving an outputSchema and
# serialising our whole return value -- including the image -- into JSON. We
# build the CallToolResult ourselves instead, so content stays raw blocks
# while structured_content still carries a real map alongside it.
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
            description="Reuse a seed from an earlier result to reproduce it. Omit for a new one.",
            json_schema_extra=_int_type_not_anyof,
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
) -> types.CallToolResult:
    """Generate an image on ComfyUI and return it inline.

    Returns "<workflow> · seed <seed> · <url>" plus -- unless fetch_image is
    false -- the image itself. The same facts are also in structured_content,
    for a client that parses rather than reads.

    Clients usually collapse tool results, so the image is easy to miss. Pass
    the URL on in your reply: it survives the collapse, and is how the user
    sees the picture when no image is returned. Write it bare so the client
    turns it into a link -- code formatting or backticks make it unclickable.

    Use exactly the workflow name the user asked for. Similar names are
    different workflows -- "Krea2" and "Krea2+Upscale" are not interchangeable.

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

    blocks: list[types.ContentBlock] = []
    images: list[dict[str, Any]] = []
    notes: list[str] = []

    for ref in refs:
        image: dict[str, Any] = {}
        if ref.filename:
            url = client.view_url(ref)
            image["url"] = url
            notes.append(f"{workflow} · seed {used_seed} · {url}")
        if not fetch_image:
            if image:
                images.append(image)
            continue

        raw = await client.fetch(ref)
        payload, mime, original = _encode(raw)
        if max(original) > MAX_IMAGE_EDGE:
            image["downscaled_from"] = f"{original[0]}x{original[1]}"
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
        if image:
            images.append(image)

    if notes:
        blocks.insert(0, types.TextContent(type="text", text="\n".join(notes)))
    elif not blocks:
        # No ref carried a filename to build a URL from, and fetch_image is
        # false -- content would otherwise be empty, indistinguishable from
        # a silent failure to a client that never looks at structured_content.
        blocks.append(types.TextContent(type="text", text=f"seed {used_seed}"))

    return types.CallToolResult(
        content=blocks,
        structured_content={"workflow": workflow, "seed": used_seed, "images": images},
    )


def main() -> None:
    """Run the MCP server. Entry point for the `uncomfymcp` command."""
    parser = argparse.ArgumentParser(prog="uncomfymcp", description=__doc__)
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
