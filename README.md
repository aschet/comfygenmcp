# uncomfymcp

A Model Context Protocol (MCP) server that connects an AI assistant to a running
ComfyUI instance. It fetches a workflow saved in ComfyUI, writes the prompt and
seed into it, runs it, and returns the image inline in the chat — so it only
works with text-to-image workflows that have a single prompt node and seed to
inject into.

## Install

Requires Python 3.10+ and a running ComfyUI.

Linux and macOS:

```bash
git clone https://github.com/aschet/uncomfymcp.git
cd uncomfymcp
python -m venv .venv
.venv/bin/pip install -e .
```

Windows:

```bat
git clone https://github.com/aschet/uncomfymcp.git
cd uncomfymcp
python -m venv .venv
.venv\Scripts\pip install -e .
```

That gives you the command `.venv/bin/uncomfymcp` (`.venv\Scripts\uncomfymcp.exe`
on Windows), which a client launches for you — nothing to start by hand beyond
ComfyUI itself.

## Connecting a Client

Any MCP client works. By default the server speaks stdio, meaning the client
launches `.venv/bin/uncomfymcp` itself and talks to it over the process
pipes. With `--transport http` it instead listens on
`http://127.0.0.1:8000/mcp` for clients that connect over the network — with
no authentication, so don't expose it without one in front.

Add this `mcpServers` entry to the client's config. Flags go in an `"args"`
array, e.g. `"args": ["--comfy-url", "http://host:8188", "--timeout", "600"]`
— add the first if ComfyUI isn't on localhost, and raise the timeout if your
workflows are slow or an agent chains several generations.

```json
{
  "mcpServers": {
    "uncomfy": {
      "command": "/path/to/uncomfymcp/.venv/bin/uncomfymcp"
    }
  }
}
```

On Windows, `command` is the `.venv\Scripts\uncomfymcp.exe` path instead.

- Claude Desktop: add it to `claude_desktop_config.json`. The image only
  shows inside the collapsed tool card — expand it, or use the URL instead.
- AnythingLLM: add it to `anythingllm_mcp_servers.json`
  (`~/.config/anythingllm-desktop/storage/plugins/` on Linux), then start it
  from Settings → Agent Skills → MCP Servers and invoke with `@agent`. It
  renders no image at all, so tell the agent to always pass
  `fetch_image: false`.

## Use

Ask the assistant what's available:

```
you:    which image workflows do I have?
claude: [list_workflows]
        Ready to generate with:
          Krea2
          Z-Image Turbo

        Present but not usable:
          Ideogram 4 -- needs models that are not installed:
          flux2-vae.safetensors, gemma4_e4b_it_fp8_scaled.safetensors,
          ideogram4_fp8_scaled.safetensors
```

Then generate with one of your own names:

```
you:    generate a red fox in deep snow using Krea2
claude: [image]  Done: generated with Krea2 · seed 12345 ·
        http://127.0.0.1:8188/view?filename=Krea2_turbo_00007_.png
```

Three tools are exposed:

| Tool | Description |
| --- | --- |
| `generate_image(prompt, workflow, seed?, fetch_image?)` | Generate and return the image. Seeds are random unless you pass one, and every result reports the seed it used. `fetch_image` defaults to true; a client that cannot render one can pass false and get only the seed and URL. |
| `list_workflows()` | The workflows saved in ComfyUI, split into those ready to run and those that cannot. |
| `comfy_status(workflow?)` | Connection check; with a name, the nodes it would patch. |

## Limitations

- Only the prompt and seed change — no width, height, steps or sampler; those
  come from the workflow. A sampler set to "fixed" in ComfyUI is not honoured.
- Node detection can pick the wrong node when a workflow has several prompt
  boxes. `comfy_status` shows what it found; set a node's Title to
  `MCP:prompt` to override.
- Images are sent as WebP, downscaled and compressed to fit a 1 MB limit,
  with transparency preserved. The full-resolution original stays in
  ComfyUI's output folder.

## Configuration

All settings are command-line flags.

| Flag | Default | Description |
| --- | --- | --- |
| `--comfy-url` | `http://127.0.0.1:8188` | Address of the ComfyUI server to generate on |
| `--transport` | `stdio` | `stdio`, or `http` to listen on a port |
| `--listen` | `127.0.0.1:8000` | Address this server binds to, with `--transport http` |
| `--timeout` | `300` | Seconds before giving up on a generation |
