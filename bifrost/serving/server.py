"""
HTTP serving layer for Clifford HLM models.

Uses only the Python standard library (http.server).  The server mounts
basic /generate and /embed endpoints plus delegates to the OpenAI-compat
and native-Clifford routers.
"""

from __future__ import annotations

import json
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Tuple
from io import BytesIO

import numpy as np


# ---------------------------------------------------------------------------
# Stub model interface — the real HLM model plugs in here
# ---------------------------------------------------------------------------
class _StubModel:
    """Minimal placeholder so the server can run without a real model."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        return f"[stub] echo: {prompt[:64]}"

    def embed(self, text: str) -> np.ndarray:
        """Return a dummy 8-component multivector embedding."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(8).astype(np.float64)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def grade_energy(self, text: str) -> Dict[int, float]:
        mv = self.embed(text)
        return {
            0: float(mv[0] ** 2),
            1: float(np.sum(mv[1:4] ** 2)),
            2: float(np.sum(mv[4:7] ** 2)),
            3: float(mv[7] ** 2),
        }

    def attention_map(self, text: str) -> np.ndarray:
        tokens = text.split()
        n = max(len(tokens), 1)
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        raw = rng.rand(n, n)
        # softmax rows
        exp = np.exp(raw - raw.max(axis=-1, keepdims=True))
        return (exp / exp.sum(axis=-1, keepdims=True)).tolist()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class _CliffordHandler(BaseHTTPRequestHandler):
    """Routes requests to the correct endpoint handler."""

    server: "CliffordHTTPServer"  # type hint for our subclass

    def log_message(self, fmt: str, *args: Any) -> None:
        # quieten default stderr logging; override if verbose needed
        pass

    # ---- helpers ---------------------------------------------------------
    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, chunks: List[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for chunk in chunks:
            event = f"data: {chunk}\n\n"
            self.wfile.write(event.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- GET -------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "model": "clifford-hlm"})
        else:
            self._send_json({"error": "not found"}, 404)

    # ---- POST ------------------------------------------------------------
    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        try:
            body = self._read_json()
        except Exception as exc:
            self._send_json({"error": f"bad request: {exc}"}, 400)
            return

        handler = self._route_post(route)
        if handler is None:
            self._send_json({"error": "not found"}, 404)
            return
        try:
            handler(body)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _route_post(self, route: str) -> Optional[Callable]:
        routes: Dict[str, Callable] = {
            "/generate": self._handle_generate,
            "/embed": self._handle_embed,
            # OpenAI compat
            "/v1/chat/completions": self._handle_chat_completions,
            # Clifford-native
            "/v1/clifford/embed": self._handle_clifford_embed,
            "/v1/clifford/grade_analysis": self._handle_grade_analysis,
            "/v1/clifford/attention_map": self._handle_attention_map,
        }
        return routes.get(route)

    # ---- endpoint implementations ----------------------------------------
    def _handle_generate(self, body: Dict) -> None:
        model = self.server.model
        text = body.get("text", body.get("prompt", ""))
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        result = model.generate(
            text, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
        self._send_json({"generated_text": result})

    def _handle_embed(self, body: Dict) -> None:
        model = self.server.model
        text = body.get("text", "")
        if isinstance(text, list):
            embeddings = model.embed_batch(text)
            self._send_json({"embeddings": embeddings.tolist()})
        else:
            embedding = model.embed(text)
            self._send_json({"embedding": embedding.tolist()})

    def _handle_chat_completions(self, body: Dict) -> None:
        from .openai_compat import handle_chat_completions
        handle_chat_completions(self, body, self.server.model)

    def _handle_clifford_embed(self, body: Dict) -> None:
        from .clifford_api import handle_clifford_embed
        handle_clifford_embed(self, body, self.server.model)

    def _handle_grade_analysis(self, body: Dict) -> None:
        from .clifford_api import handle_grade_analysis
        handle_grade_analysis(self, body, self.server.model)

    def _handle_attention_map(self, body: Dict) -> None:
        from .clifford_api import handle_attention_map
        handle_attention_map(self, body, self.server.model)


# ---------------------------------------------------------------------------
# Extended HTTPServer to carry model reference
# ---------------------------------------------------------------------------
class CliffordHTTPServer(ThreadingHTTPServer):
    def __init__(self, addr: Tuple[str, int], model: Any) -> None:
        self.model = model
        super().__init__(addr, _CliffordHandler)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
class CliffordServer:
    """High-level wrapper around the serving stack.

    Parameters
    ----------
    model : object or None
        Any object implementing generate / embed / embed_batch /
        grade_energy / attention_map.  Defaults to a stub.
    host : str
    port : int
    """

    def __init__(
        self,
        model: Any = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.model = model or _StubModel()
        self.host = host
        self.port = port
        self._httpd: Optional[CliffordHTTPServer] = None

    def start(self, blocking: bool = True) -> None:
        self._httpd = CliffordHTTPServer((self.host, self.port), self.model)
        if blocking:
            self._httpd.serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
