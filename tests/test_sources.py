# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for resolving the API graph without a local export."""

import json

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from uncomfymcp import sources
from uncomfymcp.comfy import ComfyClient

from .test_workflow import KREA_GRAPH


def _png_with_prompt(path, graph):
    meta = PngInfo()
    meta.add_text("prompt", json.dumps(graph))
    Image.new("RGB", (32, 32), (1, 2, 3)).save(path, pnginfo=meta)
    return path


@pytest.mark.anyio
async def test_image_source_reads_embedded_graph(tmp_path):
    png = _png_with_prompt(tmp_path / "gen.png", KREA_GRAPH)
    found = await sources.resolve(f"image:{png}", ComfyClient("http://x"))
    assert found.graph["6"]["inputs"]["text"] == "a surreal illustration"
    assert found.source.startswith("image:")


@pytest.mark.anyio
async def test_bare_png_path_is_treated_as_an_image_source(tmp_path):
    png = _png_with_prompt(tmp_path / "gen.png", KREA_GRAPH)
    found = await sources.resolve(str(png), ComfyClient("http://x"))
    assert "6" in found.graph


@pytest.mark.anyio
async def test_png_without_metadata_says_why(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(sources.SourceError, match="no embedded ComfyUI prompt"):
        await sources.resolve(f"image:{path}", ComfyClient("http://x"))


@pytest.mark.anyio
async def test_json_file_source(tmp_path):
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(KREA_GRAPH))
    found = await sources.resolve(str(path), ComfyClient("http://x"))
    assert found.source.startswith("file:")
    assert "3" in found.graph


@pytest.mark.anyio
async def test_ui_format_is_rejected_whatever_the_source(tmp_path):
    path = tmp_path / "ui.json"
    path.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(Exception, match="Export \\(API\\)"):
        await sources.resolve(str(path), ComfyClient("http://x"))


@pytest.mark.anyio
async def test_a_missing_path_reports_itself_missing(tmp_path):
    """Must not be retried as a saved-workflow name -- that error would confuse."""
    with pytest.raises(sources.SourceError, match="Workflow file not found"):
        await sources.resolve(str(tmp_path / "nope.json"), ComfyClient("http://x"))


@pytest.mark.anyio
async def test_a_bare_name_is_looked_up_on_the_server(monkeypatch):
    seen = {}

    async def fake_saved(client, name):
        seen["name"] = name
        return sources.Resolved(KREA_GRAPH, f"saved:{name}")

    monkeypatch.setattr(sources, "_from_saved", fake_saved)
    found = await sources.resolve("Krea2", ComfyClient("http://x"))
    assert seen["name"] == "Krea2"
    assert found.source == "saved:Krea2"


@pytest.mark.anyio
async def test_a_local_file_wins_over_a_same_named_saved_workflow(tmp_path, monkeypatch):
    path = tmp_path / "Krea2.json"
    path.write_text(json.dumps(KREA_GRAPH))
    monkeypatch.chdir(tmp_path)

    async def boom(client, name):
        raise AssertionError("should not hit the server when the file exists")

    monkeypatch.setattr(sources, "_from_saved", boom)
    found = await sources.resolve("Krea2.json", ComfyClient("http://x"))
    assert found.source.startswith("file:")


def _saved(names):
    async def fake(client):
        return list(names)

    return fake


@pytest.mark.anyio
async def test_check_saved_flags_missing_models_and_absent_prompt_nodes(monkeypatch):
    """Listing a workflow the user cannot generate with only invites a failure."""
    schemas = {
        "UNETLoader": {"input": {"required": {"unet_name": [["installed.safetensors"], {}]}}},
        "KSampler": {"input": {"required": {"seed": ["INT", {}]}}},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}},
    }

    good = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "installed.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["3", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}},
    }
    absent_model = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "nope.safetensors"}}}
    no_prompt = {"2": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    graphs = {"good.json": good, "model.json": absent_model, "prompt.json": no_prompt}

    async def fake_saved(client, name):
        return sources.Resolved(graphs[name], f"saved:{name}")

    async def fake_object_info(client):
        return schemas

    monkeypatch.setattr(sources, "list_saved", _saved(list(graphs)))
    monkeypatch.setattr(sources, "_from_saved", fake_saved)
    monkeypatch.setattr(sources, "_object_info", fake_object_info)

    result = dict(await sources.check_saved(ComfyClient("http://x")))
    assert result["good.json"] is None
    assert "not installed" in str(result["model.json"])
    assert "nope.safetensors" in str(result["model.json"])
    assert "no prompt node" in str(result["prompt.json"])


@pytest.mark.parametrize("typed", ["Krea2", "krea2", "KREA2", "krea2.json", "Krea2.json"])
def test_workflow_names_tolerate_case_and_suffix(typed):
    """Users say "krea2"; the file is "Krea2.json"."""
    assert sources._match_name(typed, ["Krea2.json", "Ideogram 4.json"]) == "Krea2.json"


def test_an_exact_match_wins_over_a_case_insensitive_one():
    available = ["photo.json", "Photo.json"]
    assert sources._match_name("Photo", available) == "Photo.json"
    assert sources._match_name("photo", available) == "photo.json"


def test_an_ambiguous_case_fold_is_refused():
    """Two names differing only in case: guessing would pick the wrong one."""
    assert sources._match_name("PHOTO", ["photo.json", "Photo.json"]) is None


UI_WITH_SUBGRAPH = {
    "nodes": [
        {"id": 29, "type": "SaveImage", "widgets_values": ["out"]},
        {"id": 30, "type": "sub-uuid", "widgets_values": ["OLD PROMPT", 1024, 1024, 111]},
    ],
    "definitions": {
        "subgraphs": [
            {
                "id": "sub-uuid",
                "inputs": [
                    {"name": "value", "type": "STRING"},
                    {"name": "width_1", "label": "width", "type": "INT"},
                    {"name": "height_1", "label": "height", "type": "INT"},
                    {"name": "seed_1", "label": "seed", "type": "INT"},
                ],
                "nodes": [
                    {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["OLD PROMPT"]},
                    {"id": 3, "type": "KSampler", "widgets_values": [111, "randomize", 8]},
                ],
            }
        ]
    },
}


def test_promoted_subgraph_widgets_are_patched():
    """These, not the inner nodes, are what ComfyUI shows and runs."""
    ui = json.loads(json.dumps(UI_WITH_SUBGRAPH))
    assert sources._patch_subgraph_instances(ui, "NEW PROMPT", 999)
    instance = ui["nodes"][1]["widgets_values"]
    assert instance[0] == "NEW PROMPT"
    assert instance[3] == 999
    assert instance[1:3] == [1024, 1024], "size widgets must be left alone"


def test_ambiguous_promoted_widgets_are_refused():
    """Two STRING inputs: writing the prompt into the wrong one is worse."""
    ui = json.loads(json.dumps(UI_WITH_SUBGRAPH))
    ui["definitions"]["subgraphs"][0]["inputs"].append({"name": "extra", "type": "STRING"})
    assert sources._patch_subgraph_instances(ui, "NEW PROMPT", 999) is False


def test_widget_swap_leaves_everything_else_untouched():
    ui = json.loads(json.dumps(UI_WITH_SUBGRAPH))
    swapped = sources._replace_widget_values(sources._all_nodes(ui), [("OLD PROMPT", "NEW")])
    assert swapped == 2, "the instance copy and the inner node both hold it"
    assert ui["nodes"][0]["widgets_values"] == ["out"]


def test_a_subgraph_promoting_nothing_needs_no_instance_patch():
    """Its inner nodes hold the prompt outright, and are patched by value."""
    ui = json.loads(json.dumps(UI_WITH_SUBGRAPH))
    ui["nodes"][1]["widgets_values"] = []
    ui["definitions"]["subgraphs"][0]["inputs"] = [{"name": "width", "type": "INT"}]
    assert sources._patch_subgraph_instances(ui, "NEW PROMPT", 999) is True


PROMOTED_UI = {
    "nodes": [
        {
            "id": 57,
            "type": "sub-uuid",
            "widgets_values": ["NEW PROMPT", "new_model.safetensors"],
            "inputs": [
                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                {
                    "name": "unet_name",
                    "type": "COMBO",
                    "widget": {"name": "unet_name"},
                    "link": None,
                },
            ],
        }
    ],
    "definitions": {
        "subgraphs": [
            {
                "id": "sub-uuid",
                "inputs": [
                    {"name": "text", "type": "STRING", "linkIds": [34]},
                    {"name": "unet_name", "type": "COMBO", "linkIds": [73]},
                ],
                "links": [
                    {
                        "id": 34,
                        "origin_id": -10,
                        "origin_slot": 0,
                        "target_id": 27,
                        "target_slot": 1,
                    },
                    {
                        "id": 73,
                        "origin_id": -10,
                        "origin_slot": 1,
                        "target_id": 28,
                        "target_slot": 0,
                    },
                ],
                "nodes": [
                    {
                        "id": 27,
                        "type": "CLIPTextEncode",
                        "widgets_values": ["STALE PROMPT"],
                        "inputs": [
                            {"name": "clip", "type": "CLIP", "link": 28},
                            {"name": "text", "type": "STRING", "widget": {"name": "text"}},
                        ],
                    },
                    {
                        "id": 28,
                        "type": "UNETLoader",
                        "widgets_values": ["stale_model.safetensors", "default"],
                        "inputs": [
                            {"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}},
                            {
                                "name": "weight_dtype",
                                "type": "COMBO",
                                "widget": {"name": "weight_dtype"},
                            },
                        ],
                    },
                ],
            }
        ]
    },
}


def test_promoted_subgraph_widgets_reach_the_api_graph():
    """The instance's values are what ComfyUI runs, not the definition's."""
    assert sources._promoted_values(PROMOTED_UI) == {
        "57:27": {"text": "NEW PROMPT"},
        "57:28": {"unet_name": "new_model.safetensors"},
    }


def test_an_externally_linked_promotion_is_not_treated_as_a_widget_value():
    """Its widget is inert -- the value comes down the link instead."""
    ui = json.loads(json.dumps(PROMOTED_UI))
    ui["nodes"][0]["inputs"][0]["link"] = 62
    assert sources._promoted_values(ui) == {"57:28": {"unet_name": "new_model.safetensors"}}


def test_convert_leaves_a_linked_input_alone(monkeypatch):
    """Overwriting a link with a literal would cut the graph in two."""
    api = {"57:28": {"class_type": "UNETLoader", "inputs": {"unet_name": ["99", 0]}}}
    monkeypatch.setattr(sources, "convert_ui_to_api", lambda *_: api)
    assert sources._convert(PROMOTED_UI, {})["57:28"]["inputs"]["unet_name"] == ["99", 0]


@pytest.mark.anyio
async def test_the_roundtrip_checks_the_node_we_patched(monkeypatch):
    """Filling the blank box is what makes it undetectable.

    Re-running detection on the filled-in graph would reject the one candidate
    that got it right, so the check names the node instead.
    """
    roundtrip = {
        "5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a paper crane"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 7}},
    }
    monkeypatch.setattr(sources, "convert_ui_to_api", lambda *_: roundtrip)
    assert await sources._matches({}, "5", "a paper crane", 7, {}) is True
    assert await sources._matches({}, "3", "a paper crane", 7, {}) is False


def test_missing_models_understands_both_combo_shapes():
    """ComfyUI emits both; understanding one silently skips half the checks."""
    schemas = {
        "Old": {"input": {"required": {"ckpt_name": [["have.safetensors"], {}]}}},
        "New": {
            "input": {"required": {"model_name": ["COMBO", {"options": ["have_upscaler.pth"]}]}}
        },
    }
    graph = {
        "1": {"class_type": "Old", "inputs": {"ckpt_name": "absent.safetensors"}},
        "2": {"class_type": "New", "inputs": {"model_name": "absent_upscaler.pth"}},
    }
    assert sorted(sources._missing_models(graph, schemas)) == [
        "absent.safetensors",
        "absent_upscaler.pth",
    ]


def test_a_combo_offering_nothing_means_everything_it_names_is_missing():
    """No checkpoints installed at all is the case that reads as "can't tell"."""
    schemas = {"Old": {"input": {"required": {"ckpt_name": [[], {}]}}}}
    graph = {"1": {"class_type": "Old", "inputs": {"ckpt_name": "wanted.safetensors"}}}
    assert sources._missing_models(graph, schemas) == ["wanted.safetensors"]


def test_a_combo_that_fills_itself_in_at_runtime_is_not_a_missing_model():
    """CustomCombo advertises no choices; 'Quality' is a setting, not a file."""
    schemas = {"CustomCombo": {"input": {"required": {"choice": [[], {}]}}}}
    graph = {"1": {"class_type": "CustomCombo", "inputs": {"choice": "Quality"}}}
    assert sources._missing_models(graph, schemas) == []
