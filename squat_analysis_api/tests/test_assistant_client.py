from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from squat_analysis_api.assistant_client import (
    DEFAULT_ASSISTANT_TIMEOUT,
    AssistantClient,
    AssistantClientError,
    apply_assistant_response,
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/analyze-squat-video":
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self._send_json(
                {
                    "overall_summary": "Looks stable.",
                    "rep_feedback": [
                        {
                            "rep_id": 1,
                            "summary": "Good rep.",
                            "cues": ["Keep tempo steady."],
                            "severity": "info",
                        }
                    ],
                    "detections": [],
                    "model_names": {"falcon": "fake", "gemma": "fake"},
                }
            )
        elif self.path == "/bad-json":
            body = b"not json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

    def _send_json(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


class AssistantClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_health_and_analyze(self):
        client = AssistantClient(self.base_url)
        self.assertEqual(client.health()["status"], "ok")

        report = {
            "video": {},
            "stats": {},
            "reps": [{"rep_id": 1, "assistant_feedback": None}],
        }
        response, latency = client.analyze(report)
        updated = apply_assistant_response(report, response, service_url=self.base_url, latency_seconds=latency)

        self.assertEqual(updated["assistant"]["status"], "completed")
        self.assertEqual(updated["reps"][0]["assistant_feedback"]["summary"], "Good rep.")

    def test_unavailable_health_fails(self):
        client = AssistantClient("http://127.0.0.1:1", timeout=0.1)
        with self.assertRaises(AssistantClientError):
            client.health()

    def test_default_timeout_uses_env(self):
        with unittest.mock.patch.dict("os.environ", {"SQUAT_ASSISTANT_CLIENT_TIMEOUT": "123"}):
            self.assertEqual(AssistantClient(self.base_url).timeout, 123)

    def test_default_timeout_fallback(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(AssistantClient(self.base_url).timeout, DEFAULT_ASSISTANT_TIMEOUT)

    def test_malformed_json_fails(self):
        client = AssistantClient(self.base_url)
        with self.assertRaises(AssistantClientError):
            client._request("POST", "bad-json", {})


if __name__ == "__main__":
    unittest.main()
