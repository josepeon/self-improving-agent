"""HTTP inference server.

Exposes SIA's currently-promoted model behind an OpenAI-compatible
``/v1/chat/completions`` endpoint. The production adapter is loaded at
startup from the model registry and reloaded when the registry changes.

Run via the CLI:

    sia serve --host 0.0.0.0 --port 8000

Importable for tests / embedding in other apps:

    from self_improving_agent.serving import app
"""

from self_improving_agent.serving.server import app

__all__ = ["app"]
