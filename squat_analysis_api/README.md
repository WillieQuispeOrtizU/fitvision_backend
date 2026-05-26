# Squat Analysis API

Python backend for squat-video analysis.

It provides:

- MediaPipe squat rep counting and annotated video generation.
- JSON reports with per-rep metrics and key frames.
- Optional Falcon/Gemma assistant feedback.
- A local FastAPI upload/job API for the React frontend.

## Prerequisites

- Python 3.12
- `uv`
- macOS Apple Silicon for the MLX assistant path, or Windows/Linux with a Torch-compatible GPU for the Torch assistant path.

## Install

Base MediaPipe analyzer only:

```bash
uv sync
```

Analyzer plus assistant/upload API dependencies:

```bash
uv sync --extra assistant
```

## Batch CLI

Put `.mp4` files in `input_video/`, then run:

```bash
uv run squat-analysis
```

Annotated videos are written to `output_video/` as `<name>_analyzed.mp4`.

Useful options:

```bash
uv run squat-analysis --mode pro
uv run squat-analysis --input input_video --output output_video
uv run squat-analysis --file input_video/example.mp4
uv run squat-analysis --file input_video/example.mp4 --report
uv run squat-analysis --file input_video/example.mp4 --assistant-url http://127.0.0.1:7862
```

## macOS Apple Silicon Setup

Install dependencies and create `.env`:

```bash
uv sync --extra assistant
cp .env.example .env
```

Download local assistant models:

```bash
uv run squat-assistant-download
```

Copy the printed local paths into `.env`. They should look like:

```env
SQUAT_ASSISTANT_FALCON_MODEL=tiiuae/Falcon-Perception-300M
FALCON_HF_LOCAL_DIR=/Users/dele/Documents/Projects/Willie_TP/squat_analysis_api/assistant_models/tiiuae__Falcon-Perception-300M
SQUAT_ASSISTANT_VLM_MODEL=/Users/dele/Documents/Projects/Willie_TP/squat_analysis_api/assistant_models/mlx-community__gemma-4-e4b-it-8bit
```

Start the Falcon/Gemma assistant server:

```bash
set -a
source .env
set +a
uv run squat-assistant-server
```

In another terminal, start the upload API:

```bash
set -a
source .env
set +a
uv run squat-analysis-api-server
```

## Windows PowerShell Setup

Install dependencies and create `.env`:

```powershell
uv sync --extra assistant
Copy-Item .env.example .env
```

Edit `.env`. For Windows, use a Torch/Transformers-compatible Gemma model id or local Hugging Face model folder:

```env
SQUAT_ASSISTANT_FALCON_MODEL=tiiuae/Falcon-Perception-300M
SQUAT_ASSISTANT_VLM_MODEL=google/gemma-4-E4B-it
SQUAT_ASSISTANT_URL=http://127.0.0.1:7862
SQUAT_ANALYSIS_API_HOST=127.0.0.1
SQUAT_ANALYSIS_API_PORT=8000
SQUAT_ANALYSIS_JOBS_DIR=jobs
```

Load `.env` into the current PowerShell process:

```powershell
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
```

Start the Falcon/Gemma assistant server:

```powershell
uv run squat-assistant-server
```

In another PowerShell terminal, load `.env` again and start the upload API:

```powershell
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
uv run squat-analysis-api-server
```

## Upload API

The upload API listens on `http://127.0.0.1:8000` by default.

Endpoints:

- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/report`
- `GET /api/jobs/{job_id}/files/{filename}`

Each upload is stored under:

```text
jobs/<job_id>/input/
jobs/<job_id>/output/
jobs/<job_id>/frames/
```

`jobs/` is ignored by git.

## JSON Reports

`--report` or assistant mode writes:

- annotated video
- analysis JSON
- rep key frames

The JSON report includes video metadata, aggregate counts, per-rep angle ranges, issue tags, key-frame paths, Falcon detections, and Gemma feedback when assistant mode is enabled.

## Troubleshooting

- If the frontend gets a report `404`, restart `uv run squat-analysis-api-server` after code changes.
- If the assistant service times out on first use, increase `SQUAT_ASSISTANT_CLIENT_TIMEOUT` in `.env`.
- If you see `Download complete: 0.00B`, the model loader checked cached files; it did not download full model weights.
- On macOS, set `FALCON_HF_LOCAL_DIR` and `SQUAT_ASSISTANT_VLM_MODEL` to the local `assistant_models/...` folders to avoid repo-id loading.
- On Windows, use PowerShell and load `.env` separately in every terminal.

