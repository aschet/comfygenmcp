# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Thin async client for the ComfyUI HTTP API."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


class ComfyError(RuntimeError):
    """ComfyUI rejected the prompt or failed while executing it."""


@dataclass
class ImageRef:
    """One image in ComfyUI's output store, as `/view` addresses it."""

    filename: str
    subfolder: str
    type: str


class ComfyClient:
    """Async client for the handful of ComfyUI endpoints this server needs."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def stats(self) -> dict[str, Any]:
        """Return ComfyUI's `/system_stats`: version, python, devices, VRAM."""
        async with await self._client() as http:
            resp = await http.get("/system_stats")
            resp.raise_for_status()
            stats: dict[str, Any] = resp.json()
            return stats

    async def queue(self, graph: dict[str, Any], ui_workflow: Any = None) -> str:
        """Submit a graph, returning the prompt id.

        `ui_workflow` is the UI-format graph. ComfyUI copies it into the PNG's
        `workflow` chunk, which is what lets an image be dragged back into the
        editor. Without it the file carries only the API prompt.
        """
        body: dict[str, Any] = {"prompt": graph, "client_id": self.client_id}
        if ui_workflow is not None:
            body["extra_data"] = {"extra_pnginfo": {"workflow": ui_workflow}}
        async with await self._client() as http:
            resp = await http.post("/prompt", json=body)
            if resp.status_code >= 400:
                raise ComfyError(_format_queue_error(resp))
            accepted: dict[str, Any] = resp.json()
            return str(accepted["prompt_id"])

    async def wait(self, prompt_id: str, poll: float = 0.7) -> dict[str, Any]:
        """Poll /history until the prompt finishes; return its history entry."""
        deadline = asyncio.get_event_loop().time() + self.timeout
        async with await self._client() as http:
            while True:
                resp = await http.get(f"/history/{prompt_id}")
                resp.raise_for_status()
                history: dict[str, dict[str, Any]] = resp.json()
                entry = history.get(prompt_id)
                if entry is not None:
                    status = entry.get("status", {})
                    # 'completed' is absent on older builds; outputs implies done.
                    if status.get("completed") or entry.get("outputs"):
                        if status.get("status_str") == "error":
                            raise ComfyError(_format_exec_error(status))
                        return entry
                    if status.get("status_str") == "error":
                        raise ComfyError(_format_exec_error(status))

                if asyncio.get_event_loop().time() > deadline:
                    await self.interrupt()
                    raise ComfyError(
                        f"Generation timed out after {self.timeout:.0f}s. Raise "
                        "--timeout if your workflow is genuinely this slow."
                    )
                await asyncio.sleep(poll)

    async def interrupt(self) -> None:
        """Ask ComfyUI to abort the running prompt. Best effort; never raises."""
        try:
            async with await self._client() as http:
                await http.post("/interrupt")
        except httpx.HTTPError:
            pass  # Best effort -- we are already failing.

    def view_url(self, ref: ImageRef) -> str:
        """Browser-openable URL for the full-resolution original on ComfyUI.

        Only non-default parameters are included. ComfyUI assumes an empty
        subfolder and type=output, so the usual case is a single-parameter URL
        with no ampersands -- which survives clients that render links
        barebones, and is far easier to copy by hand.
        """
        params = {"filename": ref.filename}
        if ref.subfolder:
            params["subfolder"] = ref.subfolder
        if ref.type and ref.type != "output":
            params["type"] = ref.type
        return f"{self.base_url}/view?{urlencode(params)}"

    async def fetch(self, ref: ImageRef) -> bytes:
        """Download one generated image."""
        async with await self._client() as http:
            resp = await http.get(
                "/view",
                params={
                    "filename": ref.filename,
                    "subfolder": ref.subfolder,
                    "type": ref.type,
                },
            )
            resp.raise_for_status()
            return resp.content

    async def generate(self, graph: dict[str, Any], ui_workflow: Any = None) -> list[ImageRef]:
        """Queue a graph, wait for it, and return references to its images."""
        prompt_id = await self.queue(graph, ui_workflow)
        entry = await self.wait(prompt_id)
        return collect_images(entry)


def collect_images(entry: dict[str, Any]) -> list[ImageRef]:
    """Pull image references out of a history entry's outputs.

    Prefers saved 'output' images over 'temp' previews when a workflow emits
    both, so we don't return the same picture twice.
    """
    refs: list[ImageRef] = []
    for node_output in entry.get("outputs", {}).values():
        for image in node_output.get("images", []) or []:
            refs.append(
                ImageRef(
                    filename=image.get("filename", ""),
                    subfolder=image.get("subfolder", ""),
                    type=image.get("type", "output"),
                )
            )
    saved = [r for r in refs if r.type == "output"]
    return saved or refs


def _format_queue_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"ComfyUI rejected the prompt (HTTP {resp.status_code}): {resp.text[:500]}"

    error = body.get("error", {})
    parts = [error.get("message") or f"HTTP {resp.status_code}"]
    if error.get("details"):
        parts.append(str(error["details"]))
    for node_id, info in (body.get("node_errors") or {}).items():
        for err in info.get("errors", []):
            parts.append(
                f"node {node_id} ({info.get('class_type', '?')}): "
                f"{err.get('message')} {err.get('details', '')}".strip()
            )
    return "ComfyUI rejected the prompt -- " + "; ".join(p for p in parts if p)


def _format_exec_error(status: dict[str, Any]) -> str:
    for kind, payload in status.get("messages", []) or []:
        if kind == "execution_error":
            return (
                f"ComfyUI failed in node {payload.get('node_id')} "
                f"({payload.get('node_type')}): {payload.get('exception_type')}: "
                f"{payload.get('exception_message')}"
            )
    return "ComfyUI reported an execution error."
