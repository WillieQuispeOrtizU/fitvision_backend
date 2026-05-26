from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class AssistantClientError(RuntimeError):
    pass


DEFAULT_ASSISTANT_TIMEOUT = 600.0


def default_assistant_timeout() -> float:
    raw_timeout = os.getenv("SQUAT_ASSISTANT_CLIENT_TIMEOUT")
    if not raw_timeout:
        return DEFAULT_ASSISTANT_TIMEOUT
    try:
        return float(raw_timeout)
    except ValueError as exc:
        raise AssistantClientError("SQUAT_ASSISTANT_CLIENT_TIMEOUT must be a number of seconds") from exc


class AssistantClient:
    def __init__(self, base_url: str, *, timeout: float | None = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = default_assistant_timeout() if timeout is None else timeout

    def health(self) -> dict[str, Any]:
        data = self._request("GET", "health")
        if not isinstance(data, dict):
            raise AssistantClientError("Assistant health response was not a JSON object")
        return data

    def analyze(self, report: dict[str, Any]) -> tuple[dict[str, Any], float]:
        payload = {
            "video": report["video"],
            "stats": report["stats"],
            "reps": report["reps"],
        }
        start = time.perf_counter()
        response = self._request("POST", "analyze-squat-video", payload)
        latency = time.perf_counter() - start
        if not isinstance(response, dict):
            raise AssistantClientError("Assistant analysis response was not a JSON object")
        return response, latency

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            urljoin(self.base_url, path),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AssistantClientError(f"Assistant service returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise AssistantClientError(f"Assistant service is unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AssistantClientError("Assistant service request timed out") from exc
        except OSError as exc:
            raise AssistantClientError(f"Assistant service connection failed: {exc}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssistantClientError("Assistant service returned invalid JSON") from exc


def apply_assistant_response(
    report: dict[str, Any],
    response: dict[str, Any],
    *,
    service_url: str,
    latency_seconds: float,
) -> dict[str, Any]:
    feedback_by_rep = {
        int(item["rep_id"]): item
        for item in response.get("rep_feedback", [])
        if isinstance(item, dict) and "rep_id" in item
    }
    for rep in report.get("reps", []):
        feedback = feedback_by_rep.get(int(rep["rep_id"]))
        if feedback is not None:
            rep["assistant_feedback"] = {
                "summary": feedback.get("summary", ""),
                "cues": feedback.get("cues", []),
                "severity": feedback.get("severity", "info"),
            }

    report["assistant"] = {
        "enabled": True,
        "service_url": service_url,
        "status": "completed",
        "model_names": response.get("model_names", {}),
        "latency_seconds": latency_seconds,
        "errors": [],
        "overall_summary": response.get("overall_summary"),
        "detections": response.get("detections", []),
    }
    return report
