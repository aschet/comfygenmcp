# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""MCP server exposing ComfyUI image generation.

Drives workflows you already have in ComfyUI rather than constructing node
graphs, so it works with whatever models and custom nodes you run.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.1.0"
