# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ComfyUI HTTP client that do not need a live ComfyUI."""

import pytest

from uncomfymcp.comfy import ComfyClient, ComfyError


@pytest.mark.anyio
async def test_a_dead_connection_raises_a_clear_error_not_a_raw_one():
    """A wrong host or a ComfyUI that isn't running must not surface as a bare traceback."""
    client = ComfyClient("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(ComfyError, match="Could not reach ComfyUI"):
        await client.stats()
