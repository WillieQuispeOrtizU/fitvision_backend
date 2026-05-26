from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_FALCON_MODEL = "tiiuae/Falcon-Perception-300M"
DEFAULT_GEMMA_MODEL = "google/gemma-4-E4B-it"
DEFAULT_MLX_GEMMA_MODEL = "mlx-community/gemma-4-e4b-it-8bit"


def _is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class FalconBoxAdapter:
    def __init__(self, model_id: str | None = None):
        if _is_macos_arm64():
            self._adapter = MlxFalconBoxAdapter(model_id)
        else:
            self._adapter = TorchFalconBoxAdapter(model_id)

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        return self._adapter.detect(frame_paths)


class TorchFalconBoxAdapter:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("SQUAT_ASSISTANT_FALCON_MODEL", DEFAULT_FALCON_MODEL)
        self._model = None
        self._tokenizer = None
        self._args = None
        self._device = None

    def _load(self):
        if self._model is None:
            from falcon_perception import load_and_prepare_model, setup_torch_config

            setup_torch_config()
            self._model, self._tokenizer, self._args = load_and_prepare_model(
                hf_model_id=self.model_id,
                hf_revision=os.getenv("FALCON_HF_REVISION", "main"),
                hf_local_dir=os.getenv("FALCON_HF_LOCAL_DIR") or None,
                device=os.getenv("CUDA_DEVICE") or None,
                dtype=os.getenv("FALCON_TORCH_DTYPE", "bfloat16"),
                compile=os.getenv("FALCON_TORCH_COMPILE", "0").lower() not in ("0", "false", "no"),
            )
            self._device = self._model.device

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        from PIL import Image

        self._load()
        detections = []
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            for query in ("person", "barbell", "dumbbell", "exercise equipment"):
                for pred in self._detect_one(image, query):
                    detections.append(
                        {
                            "frame_path": frame_path,
                            "query": query,
                            "bbox": {
                                "xy": pred.get("xy"),
                                "hw": pred.get("hw"),
                            },
                            "confidence": None,
                        }
                    )
        return detections

    def _detect_one(self, image: Any, query: str) -> list[dict[str, Any]]:
        import torch
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate

        prompt = build_prompt_for_task(query, "detection")
        max_len = getattr(self._args, "max_seq_len", None) or 4096
        patch = getattr(self._args, "spatial_patch_size", 16)
        batch = process_batch_and_generate(
            self._tokenizer,
            [(image.convert("RGB"), prompt)],
            max_length=max_len,
            min_dimension=256,
            max_dimension=1024,
            patch_size=patch,
        )
        batch = {key: (value.to(self._device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        eos = self._tokenizer.eos_token_id
        end_query = getattr(self._tokenizer, "end_of_query_token_id", None)
        stop_token_ids = [eos] if end_query is None else [eos, end_query]
        engine = BatchInferenceEngine(self._model, self._tokenizer)
        _, aux = engine.generate(
            **batch,
            max_new_tokens=100,
            temperature=0.0,
            stop_token_ids=stop_token_ids,
            seed=42,
        )

        predictions = []
        bboxes = aux[0].bboxes_raw
        idx = 0
        while idx < len(bboxes):
            box = {}
            if "x" in bboxes[idx]:
                box["xy"] = {"x": bboxes[idx]["x"], "y": bboxes[idx]["y"]}
                idx += 1
            if idx < len(bboxes) and "h" in bboxes[idx]:
                box["hw"] = {"h": bboxes[idx]["h"], "w": bboxes[idx]["w"]}
                idx += 1
            else:
                idx += 1
            if "xy" in box and "hw" in box:
                predictions.append(box)
        return predictions


class GemmaFeedbackAdapter:
    def __init__(self, model_id: str | None = None):
        if _is_macos_arm64():
            self._adapter = MlxGemmaFeedbackAdapter(model_id)
        else:
            self._adapter = TorchGemmaFeedbackAdapter(model_id)

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    def generate(self, payload: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        return self._adapter.generate(payload, detections)


class TorchGemmaFeedbackAdapter:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("SQUAT_ASSISTANT_VLM_MODEL", DEFAULT_GEMMA_MODEL)
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                dtype="auto",
                device_map="auto",
            )

    def generate(self, payload: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from PIL import Image

        self._load()
        prompt = _build_gemma_prompt(payload, detections)
        image_paths = _frame_paths_from_reps(payload.get("reps", []))[:4]
        content = [{"type": "text", "text": prompt}]
        if image_paths:
            content.insert(0, {"type": "image", "image": Image.open(image_paths[0]).convert("RGB")})

        inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=1200, temperature=0.1, do_sample=True)
        text = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        parsed = _parse_json_object(text)
        if parsed is None:
            parsed = _heuristic_feedback(payload)
            parsed["overall_summary"] = text.strip() or parsed["overall_summary"]
        return parsed


class MlxFalconBoxAdapter:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("SQUAT_ASSISTANT_FALCON_MODEL", DEFAULT_FALCON_MODEL)
        self._model = None
        self._tokenizer = None
        self._args = None

    def _load(self) -> None:
        if self._model is None:
            from falcon_perception import load_from_hf_export_mlx

            self._model, self._tokenizer, self._args = load_from_hf_export_mlx(
                hf_model_id=self.model_id,
                hf_local_dir=os.getenv("FALCON_HF_LOCAL_DIR") or None,
                dtype=os.getenv("FALCON_MLX_DTYPE", "float16"),
            )

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        from PIL import Image

        self._load()
        detections = []
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            for query in ("person", "barbell", "dumbbell", "exercise equipment"):
                for pred in self._detect_one(image, query):
                    detections.append(
                        {
                            "frame_path": frame_path,
                            "query": query,
                            "bbox": {
                                "xy": pred.get("xy"),
                                "hw": pred.get("hw"),
                            },
                            "confidence": None,
                        }
                    )
        return detections

    def _detect_one(self, image: Any, query: str) -> list[dict[str, Any]]:
        from falcon_perception import build_prompt_for_task
        from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate

        task = "detection"
        prompt = build_prompt_for_task(query, task)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.convert("RGB").save(tmp.name)
            batch = process_batch_and_generate(
                self._tokenizer,
                [(tmp.name, prompt)],
                max_length=self._args.max_seq_len,
                min_dimension=256,
                max_dimension=1024,
                patch_size=self._args.spatial_patch_size,
            )
            engine = BatchInferenceEngine(self._model, self._tokenizer)
            _, aux = engine.generate(
                tokens=batch["tokens"],
                pos_t=batch["pos_t"],
                pos_hw=batch["pos_hw"],
                pixel_values=batch["pixel_values"],
                pixel_mask=batch["pixel_mask"],
                max_new_tokens=100,
                temperature=0.0,
                task=task,
            )
        finally:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)

        predictions = []
        bboxes = aux[0].bboxes_raw
        idx = 0
        while idx < len(bboxes):
            box = {}
            if "x" in bboxes[idx]:
                box["xy"] = {"x": bboxes[idx]["x"], "y": bboxes[idx]["y"]}
                idx += 1
            if idx < len(bboxes) and "h" in bboxes[idx]:
                box["hw"] = {"h": bboxes[idx]["h"], "w": bboxes[idx]["w"]}
                idx += 1
            else:
                idx += 1
            if "xy" in box and "hw" in box:
                predictions.append(box)
        return predictions


class MlxGemmaFeedbackAdapter:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("SQUAT_ASSISTANT_VLM_MODEL", DEFAULT_MLX_GEMMA_MODEL)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is None:
            from mlx_vlm import load

            self._model, self._processor = load(self.model_id)

    def generate(self, payload: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        image_paths = _frame_paths_from_reps(payload.get("reps", []))[:1]
        if not image_paths:
            return _heuristic_feedback(payload)

        self._load()
        prompt = _build_gemma_prompt(payload, detections)
        config = load_config(self.model_id)
        formatted_prompt = apply_chat_template(self._processor, config, prompt, num_images=1)
        result = generate(
            self._model,
            self._processor,
            formatted_prompt,
            image_paths,
            verbose=False,
            max_tokens=1200,
            temperature=0.1,
        )
        text = result.text if hasattr(result, "text") else str(result)
        parsed = _parse_json_object(text)
        if parsed is None:
            parsed = _heuristic_feedback(payload)
            parsed["overall_summary"] = text.strip() or parsed["overall_summary"]
        return parsed


class HeuristicFeedbackAdapter:
    def __init__(self, model_id: str = "heuristic-test-adapter"):
        self.model_id = model_id

    def generate(self, payload: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        return _heuristic_feedback(payload)


class EmptyFalconAdapter:
    model_id = "empty-test-adapter"

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        return []


def analyze_squat_payload(
    payload: dict[str, Any],
    *,
    falcon_adapter: Any,
    gemma_adapter: Any,
) -> dict[str, Any]:
    frame_paths = _frame_paths_from_reps(payload.get("reps", []))
    detections = falcon_adapter.detect(frame_paths)
    feedback = gemma_adapter.generate(payload, detections)
    feedback.setdefault("overall_summary", "")
    feedback.setdefault("rep_feedback", [])
    feedback.setdefault("detections", detections)
    feedback["detections"] = detections
    feedback["model_names"] = {
        "falcon": getattr(falcon_adapter, "model_id", DEFAULT_FALCON_MODEL),
        "gemma": getattr(gemma_adapter, "model_id", DEFAULT_GEMMA_MODEL),
    }
    return feedback


def create_app(falcon_adapter: Any | None = None, gemma_adapter: Any | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install assistant extras before running the server: uv sync --extra assistant") from exc

    falcon = falcon_adapter or FalconBoxAdapter()
    gemma = gemma_adapter or GemmaFeedbackAdapter()
    app = FastAPI(title="Squat Assistant Service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_names": {
                "falcon": getattr(falcon, "model_id", DEFAULT_FALCON_MODEL),
                "gemma": getattr(gemma, "model_id", DEFAULT_GEMMA_MODEL),
            },
        }

    @app.post("/analyze-squat-video")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return analyze_squat_payload(payload, falcon_adapter=falcon, gemma_adapter=gemma)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install assistant extras before running the server: uv sync --extra assistant") from exc

    host = os.getenv("SQUAT_ASSISTANT_HOST", "127.0.0.1")
    port = int(os.getenv("SQUAT_ASSISTANT_PORT", "7862"))
    uvicorn.run(create_app(), host=host, port=port)


def _frame_paths_from_reps(reps: list[dict[str, Any]]) -> list[str]:
    seen = set()
    paths = []
    for rep in reps:
        for path in rep.get("key_frames", {}).values():
            if path and path not in seen and Path(path).exists():
                seen.add(path)
                paths.append(path)
    return paths


def _build_gemma_prompt(payload: dict[str, Any], detections: list[dict[str, Any]]) -> str:
    compact_payload = {
        "stats": payload.get("stats", {}),
        "reps": [
            {
                "rep_id": rep.get("rep_id"),
                "correct": rep.get("correct"),
                "issue_tags": rep.get("issue_tags", []),
                "worst_issue": rep.get("worst_issue"),
                "angles": rep.get("angles", {}),
                "duration": rep.get("duration"),
            }
            for rep in payload.get("reps", [])
        ],
        "detections": detections[:50],
    }
    return (
        "You are a concise strength-coaching assistant. MediaPipe metrics are the source of truth. "
        "Use the images only to add visual context. Do not provide medical advice. "
        "Return only valid JSON with keys: overall_summary, rep_feedback. "
        "rep_feedback must contain objects with rep_id, summary, cues, severity.\n\n"
        f"Analysis data:\n{json.dumps(compact_payload, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _heuristic_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    issue_cues = {
        "bend_backwards": "Keep your ribs stacked over your hips instead of leaning back.",
        "bend_forward": "Brace and keep your chest from dropping during the descent.",
        "knee_over_toe": "Control knee travel and keep pressure through the mid-foot.",
        "squat_too_deep": "Stop slightly higher if depth causes posture to break.",
        "lower_hips": "Aim for a little more depth before standing up.",
    }
    rep_feedback = []
    for rep in payload.get("reps", []):
        issues = rep.get("issue_tags", [])
        cues = [issue_cues[issue] for issue in issues if issue in issue_cues]
        if rep.get("correct"):
            summary = "Counted as a correct rep based on the MediaPipe angle thresholds."
            severity = "info"
        else:
            summary = "Counted as an incorrect rep based on the MediaPipe angle thresholds."
            severity = "warning"
        rep_feedback.append(
            {
                "rep_id": rep.get("rep_id"),
                "summary": summary,
                "cues": cues,
                "severity": severity,
            }
        )

    stats = payload.get("stats", {})
    overall = (
        f"Analyzed {stats.get('total_reps', 0)} reps: "
        f"{stats.get('correct', 0)} correct and {stats.get('incorrect', 0)} incorrect."
    )
    return {"overall_summary": overall, "rep_feedback": rep_feedback}


if __name__ == "__main__":
    main()
