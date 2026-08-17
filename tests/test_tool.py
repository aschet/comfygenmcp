# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Tool-level tests with a stubbed ComfyUI, covering what the client receives."""

import base64
import io
import json

import mcp.types as types
import pytest
from PIL import Image

from uncomfymcp import server
from uncomfymcp.comfy import ImageRef

from .test_workflow import KREA_GRAPH


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point the server at a real workflow file and a fake ComfyUI."""
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(KREA_GRAPH))

    sent = {"workflow": str(path)}

    buffer = io.BytesIO()
    Image.new("RGB", (128, 96), (0, 128, 255)).save(buffer, format="PNG")
    png = buffer.getvalue()

    class StubClient:
        def __init__(self, url, timeout=300.0):
            pass

        async def generate(self, graph, ui_workflow=None):
            sent["graph"] = graph
            sent["ui_workflow"] = ui_workflow
            return [ImageRef("out.png", "", "output")]

        async def fetch(self, ref):
            return png

        def view_url(self, ref):
            return f"http://comfy.test/view?filename={ref.filename}"

    monkeypatch.setattr(server, "ComfyClient", StubClient)
    return sent


@pytest.mark.anyio
async def test_returns_text_then_image(wired):
    blocks = await server.generate_image(prompt="a fox", seed=1234, workflow=wired["workflow"])

    assert isinstance(blocks[0], types.TextContent)
    assert "seed 1234" in blocks[0].text

    image = blocks[1]
    assert isinstance(image, types.ImageContent)
    assert image.mime_type == "image/webp"
    decoded = Image.open(io.BytesIO(base64.b64decode(image.data)))
    assert decoded.size == (128, 96)


@pytest.mark.anyio
async def test_prompt_and_seed_reach_the_graph(wired):
    await server.generate_image(prompt="a fox in snow", seed=77, workflow=wired["workflow"])
    graph = wired["graph"]
    assert graph["6"]["inputs"]["text"] == "a fox in snow"
    assert graph["3"]["inputs"]["seed"] == 77


@pytest.mark.anyio
async def test_omitted_seed_is_random_and_reported(wired):
    first = await server.generate_image(prompt="x", workflow=wired["workflow"])
    seed_a = wired["graph"]["3"]["inputs"]["seed"]
    second = await server.generate_image(prompt="x", workflow=wired["workflow"])
    seed_b = wired["graph"]["3"]["inputs"]["seed"]

    assert seed_a != seed_b, "omitting the seed must not repeat the same image"
    assert f"seed {seed_a}" in first[0].text
    assert f"seed {seed_b}" in second[0].text
    assert 0 <= seed_a <= server.MAX_SEED


@pytest.mark.anyio
async def test_missing_workflow_file_explains_itself(tmp_path):
    with pytest.raises(Exception, match="not found"):
        await server.generate_image(prompt="x", workflow=str(tmp_path / "nope.json"))


@pytest.mark.anyio
async def test_status_reports_detected_nodes(wired, monkeypatch):
    async def fake_resolve(spec, client):
        return server.sources.Resolved(KREA_GRAPH, "saved:test.json")

    monkeypatch.setattr(server.sources, "resolve", fake_resolve)

    report = json.loads(await server.comfy_status(workflow="test"))
    assert report["workflow_resolved_from"] == "saved:test.json"
    assert report["workflow"]["prompt_node"]["node"] == "6"
    assert report["workflow"]["seed_inputs"] == ["3.seed"]


@pytest.mark.anyio
async def test_fetch_image_false_returns_only_text(wired):
    """The URL has to carry the result when the image is not returned.

    Some clients hand the base64 to the model as text, which no small context
    window survives.
    """
    blocks = await server.generate_image(
        prompt="x", seed=5, workflow=wired["workflow"], fetch_image=False
    )

    assert [type(b) for b in blocks] == [types.TextContent]
    assert "seed 5" in blocks[0].text
    assert "http://comfy.test/view?filename=out.png" in blocks[0].text


@pytest.mark.anyio
async def test_nothing_is_embedded_when_there_is_no_ui_graph(wired):
    """An API-format .json source has no UI graph, so there is none to embed."""
    await server.generate_image(prompt="x", seed=5, workflow=wired["workflow"])
    assert wired["ui_workflow"] is None
