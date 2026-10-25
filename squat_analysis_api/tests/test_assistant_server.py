from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from squat_analysis_api.assistant_server import (
    EmptyFalconAdapter,
    GENERAL_TECHNIQUE_CUE,
    HeuristicFeedbackAdapter,
    _build_gemma_prompt,
    _feedback_from_model_text,
    _heuristic_feedback,
    _select_key_frames,
    analyze_squat_payload,
)


class _FakeFalcon:
    model_id = "fake-falcon"

    def detect(self, frame_paths):
        return [
            {
                "frame_path": frame_paths[0] if frame_paths else "none",
                "query": "person",
                "bbox": {"xy": {"x": 0.5, "y": 0.5}, "hw": {"h": 0.8, "w": 0.4}},
                "confidence": None,
            }
        ]


class _FakeGemma:
    model_id = "fake-gemma"

    def generate(self, payload, detections):
        return {
            "overall_summary": "Se revisó una repetición.",
            "rep_feedback": [
                {
                    "rep_id": 1,
                    "summary": "La repetición fue revisada.",
                    "cues": ["Mantén el abdomen firme."],
                    "severity": "info",
                }
            ],
        }


class AssistantServerTest(unittest.TestCase):
    def test_analyze_squat_payload_uses_adapters(self):
        payload = {
            "stats": {"total_reps": 1, "correct": 1, "incorrect": 0},
            "reps": [{"rep_id": 1, "key_frames": {"bottom": "/missing.jpg"}}],
        }

        result = analyze_squat_payload(payload, falcon_adapter=_FakeFalcon(), gemma_adapter=_FakeGemma())

        self.assertEqual(result["overall_summary"], "Se revisó una repetición.")
        self.assertEqual(result["model_names"]["falcon"], "fake-falcon")
        self.assertEqual(result["model_names"]["gemma"], "fake-gemma")
        self.assertEqual(result["rep_feedback"][0]["rep_id"], 1)

    def test_analyze_squat_payload_preserves_response_contract(self):
        payload = _feedback_payload()

        result = analyze_squat_payload(
            payload,
            falcon_adapter=EmptyFalconAdapter(),
            gemma_adapter=HeuristicFeedbackAdapter(),
        )

        self.assertEqual(
            set(result),
            {"overall_summary", "rep_feedback", "detections", "model_names"},
        )
        self.assertIsInstance(result["overall_summary"], str)
        self.assertEqual(result["detections"], [])
        self.assertEqual(
            result["model_names"],
            {"falcon": "empty-test-adapter", "gemma": "heuristic-test-adapter"},
        )
        self.assertEqual(len(result["rep_feedback"]), 2)
        for item in result["rep_feedback"]:
            self.assertEqual(set(item), {"rep_id", "summary", "cues", "severity"})
            self.assertIsInstance(item["rep_id"], int)
            self.assertIsInstance(item["summary"], str)
            self.assertIsInstance(item["cues"], list)
            self.assertIn(item["severity"], {"info", "warning"})


class SpanishFeedbackTest(unittest.TestCase):
    def test_heuristic_feedback_is_spanish_for_every_supported_issue(self):
        payload = _feedback_payload()

        result = _heuristic_feedback(payload)

        self.assertEqual(set(result), {"overall_summary", "rep_feedback"})
        self.assertEqual(
            result["overall_summary"],
            "Se analizaron 2 repeticiones: 1 correcta y 1 incorrecta.",
        )
        self.assertEqual(
            result["rep_feedback"],
            [
                {
                    "rep_id": 1,
                    "summary": "La repetición fue clasificada como correcta según las métricas de MediaPipe.",
                    "cues": [],
                    "severity": "info",
                },
                {
                    "rep_id": 2,
                    "summary": "La repetición fue clasificada como incorrecta según las métricas de MediaPipe.",
                    "cues": [
                        "Evita inclinar el torso hacia atrás y mantén una postura estable.",
                        "Mantén el abdomen firme y evita que el pecho caiga durante el descenso.",
                        "Controla el desplazamiento de las rodillas y mantén el apoyo estable sobre el pie.",
                        "Reduce ligeramente la profundidad si la postura comienza a perder estabilidad.",
                        "Desciende un poco más antes de iniciar el ascenso.",
                    ],
                    "severity": "warning",
                },
            ],
        )
        self.assertEqual(
            payload["reps"][1]["issue_tags"],
            ["bend_backwards", "bend_forward", "knee_over_toe", "squat_too_deep", "lower_hips"],
        )

    def test_overall_summary_uses_singular_concordance(self):
        result = _heuristic_feedback(
            {
                "stats": {"total_reps": 1, "correct": 1, "incorrect": 0},
                "reps": [],
            }
        )

        self.assertEqual(
            result["overall_summary"],
            "Se analizó 1 repetición: 1 correcta y 0 incorrectas.",
        )

    def test_overall_summary_uses_plural_concordance(self):
        result = _heuristic_feedback(
            {
                "stats": {"total_reps": 4, "correct": 2, "incorrect": 2},
                "reps": [],
            }
        )

        self.assertEqual(
            result["overall_summary"],
            "Se analizaron 4 repeticiones: 2 correctas y 2 incorrectas.",
        )

    def test_heuristic_feedback_does_not_contain_legacy_english_copy(self):
        serialized = json.dumps(_heuristic_feedback(_feedback_payload()), ensure_ascii=False)
        legacy_copy = (
            "Analyzed 2 reps: 1 correct and 1 incorrect.",
            "Counted as a correct rep based on the MediaPipe angle thresholds.",
            "Counted as an incorrect rep based on the MediaPipe angle thresholds.",
            "Keep your ribs stacked over your hips instead of leaning back.",
            "Brace and keep your chest from dropping during the descent.",
            "Control knee travel and keep pressure through the mid-foot.",
            "Stop slightly higher if depth causes posture to break.",
            "Aim for a little more depth before standing up.",
        )

        for text in legacy_copy:
            with self.subTest(text=text):
                self.assertNotIn(text, serialized)

    def test_invalid_model_json_uses_complete_spanish_fallback(self):
        payload = _feedback_payload()

        result = _feedback_from_model_text("This is not valid JSON.", payload)

        self.assertEqual(result, _heuristic_feedback(payload))
        self.assertNotIn("This is not valid JSON.", json.dumps(result, ensure_ascii=False))

    def test_unknown_issue_uses_general_recommendation(self):
        result = _heuristic_feedback(
            {
                "stats": {"total_reps": 1, "correct": 0, "incorrect": 1},
                "reps": [
                    {
                        "rep_id": 1,
                        "correct": False,
                        "issue_tags": ["future_issue"],
                    }
                ],
            }
        )

        self.assertEqual(
            result["rep_feedback"][0]["cues"],
            [GENERAL_TECHNIQUE_CUE],
        )

    def test_incorrect_rep_without_issue_uses_general_recommendation(self):
        result = _heuristic_feedback(
            {
                "stats": {"total_reps": 1, "correct": 0, "incorrect": 1},
                "reps": [{"rep_id": 1, "correct": False, "issue_tags": []}],
            }
        )

        self.assertEqual(
            result["rep_feedback"][0]["cues"],
            [GENERAL_TECHNIQUE_CUE],
        )


class GemmaPromptTest(unittest.TestCase):
    def test_prompt_requires_spanish_json_and_preserves_internal_identifiers(self):
        payload = _feedback_payload()
        prompt = _build_gemma_prompt(payload, [{"query": "person"}])
        prompt_lower = prompt.lower()

        self.assertIn("siempre en español", prompt_lower)
        self.assertIn("json válido", prompt_lower)
        self.assertIn("mediapipe", prompt_lower)
        self.assertIn("fuente de verdad", prompt_lower)
        self.assertIn("contexto visual adicional", prompt_lower)
        self.assertIn("no contradigas", prompt_lower)
        self.assertIn("consejos médicos", prompt_lower)
        for key in ("overall_summary", "rep_feedback", "rep_id", "summary", "cues", "severity"):
            with self.subTest(key=key):
                self.assertIn(key, prompt)

        json_start = prompt.rfind("\n{")
        self.assertGreaterEqual(json_start, 0)
        analysis_data = json.loads(prompt[json_start + 1 :])
        self.assertEqual(
            analysis_data["reps"][1]["issue_tags"],
            ["bend_backwards", "bend_forward", "knee_over_toe", "squat_too_deep", "lower_hips"],
        )

    def test_prompt_allows_only_maintenance_for_correct_rep_without_issues(self):
        payload = {
            "stats": {"total_reps": 1, "correct": 1, "incorrect": 0},
            "reps": [{"rep_id": 1, "correct": True, "issue_tags": []}],
        }

        prompt = _build_gemma_prompt(payload, [{"query": "person"}])
        prompt_lower = prompt.lower()

        self.assertIn("issue_tags como la lista exhaustiva", prompt_lower)
        self.assertIn("correct es true e issue_tags está vacío", prompt_lower)
        self.assertIn('[' + '"Mantén esta ejecución controlada y estable."' + ']', prompt)
        self.assertIn("no recomiendes bajar más", prompt_lower)
        self.assertIn("ni insinúes defectos", prompt_lower)
        self.assertIn("detecciones de falcon", prompt_lower)
        self.assertIn("no pueden justificar ni inventar errores biomecánicos", prompt_lower)
        self.assertIn("frases completas y naturales", prompt_lower)
        self.assertIn("espacios normales entre todas las palabras", prompt_lower)

        analysis_data = _analysis_data_from_prompt(prompt)
        self.assertIs(analysis_data["reps"][0]["correct"], True)
        self.assertEqual(analysis_data["reps"][0]["issue_tags"], [])

    def test_prompt_restricts_feedback_to_the_single_reported_issue(self):
        payload = {
            "stats": {"total_reps": 1, "correct": 0, "incorrect": 1},
            "reps": [
                {
                    "rep_id": 1,
                    "correct": False,
                    "issue_tags": ["bend_forward"],
                    "worst_issue": "bend_forward",
                }
            ],
        }

        prompt = _build_gemma_prompt(payload, [{"query": "person"}])
        prompt_lower = prompt.lower()

        self.assertIn("debe corresponder exclusivamente", prompt_lower)
        self.assertIn("un defecto ausente", prompt_lower)
        self.assertIn("no estén respaldadas por issue_tags", prompt_lower)
        analysis_data = _analysis_data_from_prompt(prompt)
        self.assertEqual(
            analysis_data["reps"][0]["issue_tags"],
            ["bend_forward"],
        )


def _feedback_payload():
    return {
        "stats": {"total_reps": 2, "correct": 1, "incorrect": 1},
        "reps": [
            {
                "rep_id": 1,
                "correct": True,
                "issue_tags": [],
                "angles": {"min_knee": 82.0},
                "duration": 1.2,
                "key_frames": {"bottom": "/missing-1.jpg"},
            },
            {
                "rep_id": 2,
                "correct": False,
                "issue_tags": [
                    "bend_backwards",
                    "bend_forward",
                    "knee_over_toe",
                    "squat_too_deep",
                    "lower_hips",
                ],
                "worst_issue": "bend_forward",
                "angles": {"min_knee": 121.0},
                "duration": 1.4,
                "key_frames": {"bottom": "/missing-2.jpg"},
            },
        ],
    }


def _analysis_data_from_prompt(prompt):
    json_start = prompt.rfind("\n{")
    if json_start < 0:
        raise AssertionError("Prompt does not contain embedded analysis JSON")
    return json.loads(prompt[json_start + 1 :])


class SelectKeyFramesTest(unittest.TestCase):
    def test_prefers_worst_issue_dedupes_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("start", "bottom", "worst", "r2bottom"):
                path = root / f"{name}.jpg"
                path.write_bytes(b"x")
                paths[name] = str(path)

            reps = [
                {"key_frames": {"start": paths["start"], "bottom": paths["bottom"], "worst_issue": paths["worst"]}},
                {"key_frames": {"start": paths["start"], "bottom": paths["r2bottom"]}},
            ]

            # limit=1 keeps only the first rep's most informative (worst_issue) frame.
            self.assertEqual(_select_key_frames(reps, 1), [paths["worst"]])
            # limit=5 falls back to bottom for rep 2 and skips the duplicate start frame.
            self.assertEqual(_select_key_frames(reps, 5), [paths["worst"], paths["r2bottom"]])

    def test_skips_missing_paths(self):
        reps = [{"key_frames": {"bottom": "/does/not/exist.jpg"}}]
        self.assertEqual(_select_key_frames(reps, 4), [])


if __name__ == "__main__":
    unittest.main()

