# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""End-to-end tests against a live ComfyUI.

Skipped automatically when nothing is listening. These use EmptyImage so they
exercise the full queue -> poll -> fetch path without loading any model.
"""

import asyncio
import base64
import io
import os

import httpx
import pytest
from PIL import Image

from uncomfymcp.comfy import ComfyClient, ComfyError

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")


def _reachable() -> bool:
    try:
        return httpx.get(f"{COMFY_URL}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"no ComfyUI at {COMFY_URL}")

# A red 64x64 image straight to disk -- no checkpoint, no VAE, no sampler.
BLANK_GRAPH = {
    "1": {
        "class_type": "EmptyImage",
        "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0xFF0000},
    },
    # PreviewImage, not SaveImage: this writes to ComfyUI's temp directory,
    # so running the tests does not litter the user's real output folder.
    "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
}


def test_stats():
    stats = asyncio.run(ComfyClient(COMFY_URL).stats())
    assert "devices" in stats or "system" in stats


def test_generate_roundtrip_returns_the_actual_pixels():
    client = ComfyClient(COMFY_URL, timeout=60)

    async def run():
        refs = await client.generate(BLANK_GRAPH)
        assert refs, "workflow produced no images"
        return await client.fetch(refs[0])

    data = asyncio.run(run())
    image = Image.open(io.BytesIO(data))
    assert image.size == (64, 64)
    assert image.convert("RGB").getpixel((32, 32)) == (255, 0, 0)


def test_bad_graph_raises_a_useful_error():
    broken = {"1": {"class_type": "NoSuchNodeType", "inputs": {}}}
    with pytest.raises(ComfyError) as exc:
        asyncio.run(ComfyClient(COMFY_URL, timeout=30).generate(broken))
    assert "rejected" in str(exc.value).lower()


@pytest.mark.anyio
async def test_saved_workflows_are_listable_and_convertible():
    """The durable path: fetch a UI workflow off the server and convert it."""
    from uncomfymcp import sources
    from uncomfymcp import workflow as wf

    client = ComfyClient(COMFY_URL, timeout=120)
    names = await sources.list_saved(client)
    assert names, "ComfyUI reports no saved workflows"

    converted = 0
    for name in names:
        try:
            found = await sources.resolve(name, client)
        except sources.SourceError:
            continue  # Some workflows use custom nodes that are not installed.
        assert found.source.startswith("saved:")
        wf.load(found.graph)
        assert all("class_type" in n for n in found.graph.values())
        converted += 1

    assert converted, f"none of {names} could be converted to API format"


def test_encode_fits_the_payload_budget_and_stays_decodable():
    from uncomfymcp import server

    big = Image.new("RGB", (4000, 2000), (10, 200, 30))
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")

    payload, mime, original = server._encode(buffer.getvalue())
    assert original == (4000, 2000)
    assert mime == "image/webp"
    assert len(payload) < 1_000_000, "must fit the client's 1 MB image limit"

    decoded = Image.open(io.BytesIO(base64.b64decode(payload)))
    assert max(decoded.size) <= server.MAX_IMAGE_EDGE


def test_transparency_survives_encoding():
    """WebP has an alpha channel, so a cutout's transparency is not flattened."""
    from uncomfymcp import server

    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    im.paste((200, 30, 30, 255), (20, 20, 44, 44))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")

    payload, _, _ = server._encode(buffer.getvalue())
    decoded = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA")
    assert decoded.getpixel((2, 2))[3] == 0, "corner was transparent"

    r, g, b, a = decoded.getpixel((32, 32))
    assert a == 255, "center was opaque"
    # Lossy compression rounds slightly; a black background would round to
    # near-zero, not near the original red.
    assert (r, g, b) == pytest.approx((200, 30, 30), abs=5)


def test_view_url_omits_default_parameters():
    """Ampersands are what barebones renderers mangle; skip needless ones."""
    from uncomfymcp.comfy import ComfyClient, ImageRef

    client = ComfyClient("http://127.0.0.1:8188")
    plain = client.view_url(ImageRef("out.png", "", "output"))
    assert plain == "http://127.0.0.1:8188/view?filename=out.png"
    assert "&" not in plain

    # Anything non-default still has to be carried, or the URL 404s.
    temp = client.view_url(ImageRef("out.png", "sub", "temp"))
    assert "subfolder=sub" in temp
    assert "type=temp" in temp
