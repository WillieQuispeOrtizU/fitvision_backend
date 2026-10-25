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

# Object queries Falcon runs per frame. "exercise equipment" was dropped from the
# default set: it is vague and roughly doubled inference cost for little signal.
DEFAULT_DETECTION_QUERIES = ("person", "barbell", "dumbbell")
DEFAULT_DETECT_FRAME_LIMIT = 4
DEFAULT_VLM_IMAGE_LIMIT = 1
GENERAL_TECHNIQUE_CUE = (
    "Revisa tu postura y repite el movimiento de forma controlada manteniendo "
    "una alineación estable durante toda la ejecución."
)


def _is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _detection_queries() -> tuple[str, ...]:
    raw = os.getenv("SQUAT_ASSISTANT_DETECT_QUERIES")
    if not raw:
        return DEFAULT_DETECTION_QUERIES
    queries = tuple(query.strip() for query in raw.split(",") if query.strip())
    return queries or DEFAULT_DETECTION_QUERIES


def _select_key_frames(reps: list[dict[str, Any]], limit: int) -> list[str]:
    """Pick the most informative existing key frame per rep, deduped and capped.

    Falcon detection and the Gemma VLM both run on these. Preferring the
    worst-issue (or bottom) frame over the first available frame avoids analysing
    the standing "start" pose, which carries the least form signal, and caps the
    number of model invocations to keep latency bounded.
    """
    seen: set[str] = set()
    selected: list[str] = []
    for rep in reps:
        frames = rep.get("key_frames", {})
        path = frames.get("worst_issue") or frames.get("bottom")
        if not path:
            path = next((candidate for candidate in frames.values() if candidate), None)
        if path and path not in seen and Path(path).exists():
            seen.add(path)
            selected.append(path)
        if len(selected) >= limit:
            break
    return selected


class FalconBoxAdapter:
    def __init__(self, model_id: str | None = None):
        if _is_macos_arm64():
            self._adapter = MlxFalconBoxAdapter(model_id)
        else:
            self._adapter = TorchFalconBoxAdapter(model_id)

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    def warmup(self) -> None:
        self._adapter._load()

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        return self._adapter.detect(frame_paths)


class _FalconBoxAdapterBase:
    """Shared Falcon detection flow. Subclasses provide _load() and _detect_one()
    for their backend (Torch or MLX); the per-frame/per-query loop and the raw
    bbox parsing are identical across both."""

    def detect(self, frame_paths: list[str]) -> list[dict[str, Any]]:
        from PIL import Image

        self._load()
        detections = []
        queries = _detection_queries()
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            for query in queries:
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

    @staticmethod
    def _parse_bboxes(bboxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictions = []
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


class TorchFalconBoxAdapter(_FalconBoxAdapterBase):
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
            task="detection",
        )
        return self._parse_bboxes(aux[0].bboxes_raw)


class GemmaFeedbackAdapter:
    def __init__(self, model_id: str | None = None):
        if _is_macos_arm64():
            self._adapter = MlxGemmaFeedbackAdapter(model_id)
        else:
            self._adapter = TorchGemmaFeedbackAdapter(model_id)

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    def warmup(self) -> None:
        self._adapter._load()

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
        image_paths = _select_key_frames(payload.get("reps", []), _positive_int_env("SQUAT_ASSISTANT_VLM_IMAGE_LIMIT", DEFAULT_VLM_IMAGE_LIMIT))
        content = [{"type": "text", "text": prompt}]
        for image_path in reversed(image_paths):
            content.insert(0, {"type": "image", "image": Image.open(image_path).convert("RGB")})

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
        return _feedback_from_model_text(text, payload)


class MlxFalconBoxAdapter(_FalconBoxAdapterBase):
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

        return self._parse_bboxes(aux[0].bboxes_raw)


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

        image_paths = _select_key_frames(payload.get("reps", []), _positive_int_env("SQUAT_ASSISTANT_VLM_IMAGE_LIMIT", DEFAULT_VLM_IMAGE_LIMIT))
        if not image_paths:
            return _heuristic_feedback(payload)

        self._load()
        prompt = _build_gemma_prompt(payload, detections)
        config = load_config(self.model_id)
        formatted_prompt = apply_chat_template(self._processor, config, prompt, num_images=len(image_paths))
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
        return _feedback_from_model_text(text, payload)


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
    frame_paths = _select_key_frames(
        payload.get("reps", []),
        _positive_int_env("SQUAT_ASSISTANT_DETECT_FRAME_LIMIT", DEFAULT_DETECT_FRAME_LIMIT),
    )
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

    # Optionally load the models at startup so the first user request is not the
    # one that pays the multi-minute model-load cost. Off by default.
    if os.getenv("SQUAT_ASSISTANT_WARMUP", "0").lower() not in ("0", "false", "no"):
        for adapter in (falcon, gemma):
            warmup = getattr(adapter, "warmup", None)
            if callable(warmup):
                try:
                    warmup()
                except Exception:  # noqa: BLE001 - warmup must never block startup
                    pass

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
        "Eres un asistente breve de entrenamiento de fuerza para usuarios de gimnasio. "
        "Responde siempre en español claro, natural y conciso. "
        "Devuelve únicamente un objeto JSON válido, sin markdown, bloques de código ni texto adicional. "
        "Conserva exactamente en inglés las claves JSON overall_summary, rep_feedback, rep_id, "
        "summary, cues y severity. rep_feedback debe ser una lista de objetos que contengan "
        "rep_id, summary, cues y severity. Todo el contenido textual visible de overall_summary, "
        "summary y cada elemento de cues debe estar en español. "
        "Escribe frases completas y naturales, con espacios normales entre todas las palabras; "
        "no juntes palabras ni omitas espacios. "
        "Mantén en inglés y sin modificar los identificadores internos como issue_tags y sus valores, "
        "así como los valores internos de severity. "
        "Los valores correct, issue_tags y las métricas calculadas por MediaPipe son la fuente de verdad: "
        "no contradigas ni cambies stats, correct, rep_id, angles, issue_tags ni la clasificación de cada repetición. "
        "Considera issue_tags como la lista exhaustiva de defectos técnicos permitidos para cada repetición. "
        "Cada observación o recomendación técnica de summary y cues debe corresponder exclusivamente a un "
        "issue_tag presente en esa misma repetición; nunca menciones ni insinúes un defecto ausente. "
        "Si correct es true e issue_tags está vacío, presenta la repetición como correcta y usa en cues exactamente "
        "[\"Mantén esta ejecución controlada y estable.\"]. En ese caso no recomiendes bajar más, corregir la "
        "alineación, cambiar la inclinación, modificar la profundidad ni insinúes defectos. "
        "Usa las imágenes y las detecciones de Falcon únicamente como contexto visual adicional: no pueden justificar "
        "ni inventar errores biomecánicos nuevos. worst_issue, angles, imágenes y detecciones tampoco autorizan "
        "correcciones que no estén respaldadas por issue_tags. No añadas consejos médicos.\n\n"
        f"Datos del análisis:\n{json.dumps(compact_payload, indent=2)}"
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


def _feedback_from_model_text(
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Use Gemma's JSON or a fully Spanish heuristic fallback.

    Free-form model output is never exposed when JSON parsing fails because it
    may ignore the requested language, schema, or safety constraints.
    """

    parsed = _parse_json_object(text)
    return parsed if parsed is not None else _heuristic_feedback(payload)


def _heuristic_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    issue_cues = {
        "bend_backwards": "Evita inclinar el torso hacia atrás y mantén una postura estable.",
        "bend_forward": "Mantén el abdomen firme y evita que el pecho caiga durante el descenso.",
        "knee_over_toe": "Controla el desplazamiento de las rodillas y mantén el apoyo estable sobre el pie.",
        "squat_too_deep": "Reduce ligeramente la profundidad si la postura comienza a perder estabilidad.",
        "lower_hips": "Desciende un poco más antes de iniciar el ascenso.",
    }
    rep_feedback = []
    for rep in payload.get("reps", []):
        issues = rep.get("issue_tags", [])
        cues = [issue_cues[issue] for issue in issues if issue in issue_cues]
        if any(issue not in issue_cues for issue in issues):
            cues.append(GENERAL_TECHNIQUE_CUE)
        if not cues and not rep.get("correct"):
            cues.append(GENERAL_TECHNIQUE_CUE)
        if rep.get("correct"):
            summary = "La repetición fue clasificada como correcta según las métricas de MediaPipe."
            severity = "info"
        else:
            summary = "La repetición fue clasificada como incorrecta según las métricas de MediaPipe."
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
    total_reps = stats.get("total_reps", 0)
    correct_reps = stats.get("correct", 0)
    incorrect_reps = stats.get("incorrect", 0)
    analyzed = (
        "Se analizó 1 repetición"
        if total_reps == 1
        else f"Se analizaron {total_reps} repeticiones"
    )
    correct_label = "correcta" if correct_reps == 1 else "correctas"
    incorrect_label = "incorrecta" if incorrect_reps == 1 else "incorrectas"
    overall = (
        f"{analyzed}: {correct_reps} {correct_label} y "
        f"{incorrect_reps} {incorrect_label}."
    )
    return {"overall_summary": overall, "rep_feedback": rep_feedback}


if __name__ == "__main__":
    main()
