# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Detection and patching tests across the graph shapes people actually export."""

import json

import pytest

from comfygenmcp import workflow as wf

# Classic SD: sampler -> two CLIPTextEncode, seed on the KSampler.
SD_GRAPH = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 111,
            "steps": 20,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
}

# Flux-ish: the prompt sits two hops upstream behind FluxGuidance, seed lives on
# a separate RandomNoise node, and the latent node is an SD3 one.
FLUX_GRAPH = {
    "13": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["25", 0], "guider": ["22", 0]},
    },
    "22": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["26", 0]}},
    "26": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["11", 0]}},
    "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "12": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors"},
    },
    "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "a", "clip_name2": "b"}},
    "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
}


# The real Krea2 subgraph, flattened the way an API export delivers it. The
# sampler's negative goes through ConditioningZeroOut back to the *positive*
# text node -- there is no separate negative prompt here.
KREA_GRAPH = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 735915477938686,
            "steps": 8,
            "cfg": 1,
            "sampler_name": "euler",
            "model": ["10", 0],
            "positive": ["6", 0],
            "negative": ["13", 0],
            "latent_image": ["5", 0],
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a surreal illustration", "clip": ["11", 0]},
    },
    "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
    "10": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors"},
    },
    "11": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors"},
    },
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
    "12": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    "29": {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "Krea2_turbo"},
    },
}


def test_krea_positive_prompt_found():
    assert wf.find_prompt_node(KREA_GRAPH) == "6"
    assert wf.find_seed_nodes(KREA_GRAPH) == [("3", "seed")]
    assert wf.find_output_nodes(KREA_GRAPH) == ["29"]


def test_krea_patch_roundtrip():
    out = wf.patch(KREA_GRAPH, prompt="a fox in snow", seed=99)
    assert out["6"]["inputs"]["text"] == "a fox in snow"
    assert out["3"]["inputs"]["seed"] == 99
    # Size is the workflow's own business and must be left exactly as found.
    assert out["5"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}


def test_sd_graph_detection():
    assert wf.find_prompt_node(SD_GRAPH) == "6"
    assert wf.find_seed_nodes(SD_GRAPH) == [("3", "seed")]
    assert wf.find_output_nodes(SD_GRAPH) == ["9"]


def test_flux_graph_walks_through_guidance():
    """The sampler's conditioning is 3 hops from the text -- follow it."""
    assert wf.find_prompt_node(FLUX_GRAPH) == "6"
    assert wf.find_seed_nodes(FLUX_GRAPH) == [("25", "noise_seed")]


def test_patch_writes_values_and_leaves_original_alone():
    out = wf.patch(FLUX_GRAPH, prompt="a dog", seed=7)
    assert out["6"]["inputs"]["text"] == "a dog"
    assert out["25"]["inputs"]["noise_seed"] == 7
    # Source graph untouched.
    assert FLUX_GRAPH["6"]["inputs"]["text"] == "a fox"
    assert FLUX_GRAPH["25"]["inputs"]["noise_seed"] == 42


COUNCIL = {
    "9": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["22", 0]}},
    "22": {"class_type": "BasicGuider", "inputs": {"conditioning": ["6", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a hare", "clip": ["11", 0]}},
    "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x"}},
}


def test_conditioning_reached_through_a_guider():
    """SamplerCustomAdvanced has no `positive` -- follow `guider` instead."""
    assert wf.find_prompt_node(COUNCIL) == "6"


def test_string_primitives_are_not_auto_detected():
    """They also hold magic-prompt system templates; overwriting one wrecks it."""
    graph = {
        "9": {"class_type": "KSampler", "inputs": {"positive": ["7", 0]}},
        "7": {"class_type": "StringReplace", "inputs": {"string": ["5", 0]}},
        "5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "[SYSTEM] ..."}},
    }
    assert wf.find_prompt_node(graph) is None


# The shape ComfyUI's Ernie blueprints use: the positive text comes from a
# switch between the user's box and a model that rewrites it, so nothing
# upstream of the sampler holds a literal `text`.
SWITCHED = {
    "1": {"class_type": "KSampler", "inputs": {"positive": ["2", 0], "negative": ["3", 0]}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["4", 0]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "4": {
        "class_type": "ComfySwitchNode",
        "inputs": {"switch": ["7", 0], "on_false": ["5", 0], "on_true": ["6", 0]},
    },
    "5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": ""}},
    "6": {"class_type": "TextGenerate", "inputs": {"prompt": ["8", 0]}},
    "7": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
    "8": {
        "class_type": "StringReplace",
        "inputs": {"string": "[SYSTEM] rewrite {prompt}", "replace": ["5", 0]},
    },
}


def test_a_blank_string_widget_is_the_prompt_box_when_nothing_else_is():
    """Not the system template beside it, and not the negative encode."""
    assert wf.find_prompt_node(SWITCHED) == "5"


def test_a_real_text_input_beats_a_blank_widget_on_another_branch():
    """Relaxing the rule must not start outranking the ordinary case."""
    graph = json.loads(json.dumps(SWITCHED))
    graph["2"]["inputs"]["text"] = "a hare in a hedge"
    assert wf.find_prompt_node(graph) == "2"


def test_a_titled_string_primitive_is_patched_on_its_value_input():
    graph = {
        "9": {"class_type": "KSampler", "inputs": {"positive": ["7", 0]}},
        "7": {"class_type": "StringReplace", "inputs": {"string": ["5", 0], "replace": ["6", 0]}},
        "5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "[SYSTEM] ..."}},
        "6": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": ""},
            "_meta": {"title": "MCP:prompt"},
        },
    }
    out = wf.patch(graph, prompt="a paper crane")
    assert out["6"]["inputs"]["value"] == "a paper crane"
    assert out["5"]["inputs"]["value"] == "[SYSTEM] ...", "template must survive"


def test_title_marker_overrides_heuristic():
    graph = json.loads(json.dumps(SD_GRAPH))
    graph["7"]["_meta"] = {"title": "MCP:prompt"}
    assert wf.find_prompt_node(graph) == "7"


def test_ui_format_export_is_rejected_with_guidance():
    with pytest.raises(wf.WorkflowError, match="Export \\(API\\)"):
        wf.load({"nodes": [], "links": [], "last_node_id": 3})


def test_patch_touches_only_prompt_and_seed():
    """Everything else -- size, steps, cfg, sampler, filenames -- stays put."""
    out = wf.patch(KREA_GRAPH, prompt="new", seed=1)
    changed = {
        nid: [k for k, v in n["inputs"].items() if KREA_GRAPH[nid]["inputs"][k] != v]
        for nid, n in out.items()
    }
    assert {nid: keys for nid, keys in changed.items() if keys} == {
        "6": ["text"],
        "3": ["seed"],
    }


def test_describe_is_json_safe():
    out = wf.describe(FLUX_GRAPH)
    json.dumps(out)
    assert out["prompt_node"]["node"] == "6"
    assert out["seed_inputs"] == ["25.noise_seed"]
