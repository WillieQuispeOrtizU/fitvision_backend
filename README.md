# Squat Analysis Workspace

This workspace has two local apps:

- `squat_analysis_api`: Python backend for MediaPipe squat analysis, Falcon/Gemma assistant feedback, upload jobs, and generated reports.
- `squat_analysis_web`: React/Vite frontend for uploading a video and viewing the annotated video plus feedback.

## Prerequisites

- Python 3.12
- `uv`
- Node.js 22 LTS or 24 LTS
- `npm`

Platform notes:

- macOS Apple Silicon uses MLX for Gemma/Falcon assistant inference.
- Windows uses the Torch/Transformers path. An NVIDIA GPU is strongly recommended for Falcon/Gemma feedback.

## macOS Quick Start

Terminal 1, assistant model server:

```bash
cd squat_analysis_api
uv sync --extra assistant
cp .env.example .env
uv run squat-assistant-download
```

Edit `squat_analysis_api/.env` and set the local model paths printed by the download command:

```env
FALCON_HF_LOCAL_DIR=/Users/dele/Documents/Projects/Willie_TP/squat_analysis_api/assistant_models/tiiuae__Falcon-Perception-300M
SQUAT_ASSISTANT_VLM_MODEL=/Users/dele/Documents/Projects/Willie_TP/squat_analysis_api/assistant_models/mlx-community__gemma-4-e4b-it-8bit
```

Start the assistant server:

```bash
set -a
source .env
set +a
uv run squat-assistant-server
```

Terminal 2, upload API:

```bash
cd squat_analysis_api
set -a
source .env
set +a
uv run squat-analysis-api-server
```

Terminal 3, frontend:

```bash
cd squat_analysis_web
npm install
cp .env.example .env
npm run dev
```

Open the Vite URL shown in the terminal and upload a squat video.

## Windows Quick Start

Use PowerShell.

Terminal 1, assistant model server:

```powershell
cd squat_analysis_api
uv sync --extra assistant
Copy-Item .env.example .env
```

For Windows, use a Torch/Transformers-compatible Gemma model id or local HF model folder. In `.env`, set:

```env
SQUAT_ASSISTANT_FALCON_MODEL=tiiuae/Falcon-Perception-300M
SQUAT_ASSISTANT_VLM_MODEL=google/gemma-4-E4B-it
```

If you download local models manually, point `SQUAT_ASSISTANT_VLM_MODEL` to the local Hugging Face/Transformers folder.

Load `.env` into the current PowerShell process:

```powershell
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
```

Start the assistant server:

```powershell
uv run squat-assistant-server
```

Terminal 2, upload API:

```powershell
cd squat_analysis_api
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
uv run squat-analysis-api-server
```

Terminal 3, frontend:

```powershell
cd squat_analysis_web
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL shown in the terminal and upload a squat video.

## Services

- Assistant server: `http://127.0.0.1:7862`
- Upload API server: `http://127.0.0.1:8000`
- Frontend dev server: usually `http://127.0.0.1:5173`

The assistant server can take several minutes on the first analysis request while models load into memory.

