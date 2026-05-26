from __future__ import annotations

import argparse
import platform
from pathlib import Path


DEFAULT_MODEL_DIR = Path("assistant_models")
DEFAULT_FALCON_MODEL = "tiiuae/Falcon-Perception-300M"
DEFAULT_MLX_GEMMA_MODEL = "mlx-community/gemma-4-e4b-it-8bit"
DEFAULT_TORCH_GEMMA_MODEL = "google/gemma-4-E4B-it"


def default_gemma_model() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return DEFAULT_MLX_GEMMA_MODEL
    return DEFAULT_TORCH_GEMMA_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squat-assistant-download",
        description="Download Falcon/Gemma assistant models to local folders for offline-ish reuse.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory where model snapshots will be stored.",
    )
    parser.add_argument(
        "--falcon-model",
        default=DEFAULT_FALCON_MODEL,
        help="Falcon Hugging Face model id to download.",
    )
    parser.add_argument(
        "--gemma-model",
        default=default_gemma_model(),
        help="Gemma Hugging Face model id to download. Defaults to MLX on Apple Silicon and Torch/Transformers elsewhere.",
    )
    parser.add_argument(
        "--skip-falcon",
        action="store_true",
        help="Only download the Gemma model.",
    )
    parser.add_argument(
        "--skip-gemma",
        action="store_true",
        help="Only download the Falcon model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install assistant extras first: uv sync --extra assistant") from exc

    falcon_dir = model_dir / safe_dir_name(args.falcon_model)
    gemma_dir = model_dir / safe_dir_name(args.gemma_model)

    if not args.skip_falcon:
        print(f"Downloading Falcon model {args.falcon_model} to {falcon_dir}")
        snapshot_download(
            repo_id=args.falcon_model,
            local_dir=falcon_dir,
            local_dir_use_symlinks=False,
        )

    if not args.skip_gemma:
        print(f"Downloading Gemma model {args.gemma_model} to {gemma_dir}")
        snapshot_download(
            repo_id=args.gemma_model,
            local_dir=gemma_dir,
            local_dir_use_symlinks=False,
        )

    print("\nAdd these values to .env:")
    if not args.skip_falcon:
        print(f"SQUAT_ASSISTANT_FALCON_MODEL={args.falcon_model}")
        print(f"FALCON_HF_LOCAL_DIR={falcon_dir}")
    if not args.skip_gemma:
        print(f"SQUAT_ASSISTANT_VLM_MODEL={gemma_dir}")
    return 0


def safe_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")


if __name__ == "__main__":
    raise SystemExit(main())
