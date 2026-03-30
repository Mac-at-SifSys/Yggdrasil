"""Tests for the HTTP serving layer."""

from __future__ import annotations

import sys
import os
import json
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bifrost.serving.server import CliffordServer


def _post(url: str, data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    req = Request(url)
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


class TestCliffordServer(unittest.TestCase):
    server: CliffordServer

    @classmethod
    def setUpClass(cls):
        cls.server = CliffordServer(port=18923)
        cls.server.start(blocking=False)
        # Wait for server to be ready
        for _ in range(20):
            try:
                _get(f"http://127.0.0.1:18923/health")
                break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    # ---- health ----------------------------------------------------------
    def test_health(self):
        resp = _get("http://127.0.0.1:18923/health")
        self.assertEqual(resp["status"], "ok")

    # ---- /generate -------------------------------------------------------
    def test_generate(self):
        resp = _post("http://127.0.0.1:18923/generate", {"text": "hello world"})
        self.assertIn("generated_text", resp)
        self.assertIsInstance(resp["generated_text"], str)

    # ---- /embed ----------------------------------------------------------
    def test_embed_single(self):
        resp = _post("http://127.0.0.1:18923/embed", {"text": "test"})
        self.assertIn("embedding", resp)
        self.assertEqual(len(resp["embedding"]), 8)

    def test_embed_batch(self):
        resp = _post("http://127.0.0.1:18923/embed", {"text": ["a", "b", "c"]})
        self.assertIn("embeddings", resp)
        self.assertEqual(len(resp["embeddings"]), 3)
        for emb in resp["embeddings"]:
            self.assertEqual(len(emb), 8)

    # ---- /v1/chat/completions (non-streaming) ----------------------------
    def test_chat_completions(self):
        resp = _post(
            "http://127.0.0.1:18923/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )
        self.assertIn("choices", resp)
        self.assertEqual(len(resp["choices"]), 1)
        self.assertIn("message", resp["choices"][0])

    # ---- /v1/clifford/embed ----------------------------------------------
    def test_clifford_embed(self):
        resp = _post(
            "http://127.0.0.1:18923/v1/clifford/embed", {"text": "test"}
        )
        self.assertIn("embeddings", resp)
        emb = resp["embeddings"][0]
        self.assertEqual(len(emb["multivector"]), 8)
        self.assertIn("grades", emb)
        self.assertIn("0", emb["grades"])

    # ---- /v1/clifford/grade_analysis -------------------------------------
    def test_grade_analysis(self):
        resp = _post(
            "http://127.0.0.1:18923/v1/clifford/grade_analysis",
            {"text": "test"},
        )
        self.assertIn("grade_energy", resp)
        self.assertIn("total_energy", resp)
        self.assertIn("dominant_grade", resp)
        total_frac = sum(
            v["fraction"] for v in resp["grade_energy"].values()
        )
        self.assertAlmostEqual(total_frac, 1.0, places=5)

    # ---- /v1/clifford/attention_map --------------------------------------
    def test_attention_map(self):
        resp = _post(
            "http://127.0.0.1:18923/v1/clifford/attention_map",
            {"text": "the cat sat"},
        )
        self.assertIn("tokens", resp)
        self.assertEqual(len(resp["tokens"]), 3)
        self.assertEqual(resp["shape"], [3, 3])
        self.assertEqual(len(resp["attention_scores"]), 3)

    # ---- 404 -------------------------------------------------------------
    def test_not_found(self):
        with self.assertRaises(HTTPError) as ctx:
            _post("http://127.0.0.1:18923/nonexistent", {})
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
