# Squat Analysis Web

React/Vite frontend for uploading squat videos and viewing:

- annotated MediaPipe video
- rep counts
- per-rep metrics
- Falcon/Gemma assistant feedback

## Prerequisites

- Node.js 22 LTS or 24 LTS
- `npm`
- `squat_analysis_api` assistant server running on `http://127.0.0.1:7862`
- `squat_analysis_api` upload API running on `http://127.0.0.1:8000`

## Setup

macOS/Linux:

```bash
npm install
cp .env.example .env
```

Windows PowerShell:

```powershell
npm install
Copy-Item .env.example .env
```

Default `.env`:

```env
VITE_SQUAT_API_URL=http://127.0.0.1:8000
```

## Run on macOS

Terminal 1, assistant model server:

```bash
cd ../squat_analysis_api
set -a
source .env
set +a
uv run squat-assistant-server
```

Terminal 2, upload API:

```bash
cd ../squat_analysis_api
set -a
source .env
set +a
uv run squat-analysis-api-server
```

Terminal 3, frontend:

```bash
cd ../squat_analysis_web
npm run dev
```

## Run on Windows PowerShell

Terminal 1, assistant model server:

```powershell
cd ..\squat_analysis_api
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
uv run squat-assistant-server
```

Terminal 2, upload API:

```powershell
cd ..\squat_analysis_api
Get-Content .env | Where-Object { $_ -match '^[^#].+=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
}
uv run squat-analysis-api-server
```

Terminal 3, frontend:

```powershell
cd ..\squat_analysis_web
npm run dev
```

Open the Vite URL shown in the terminal. It is usually `http://127.0.0.1:5173`.

## Build and Test

```bash
npm run build
npm test
```

## Troubleshooting

- If uploads fail, confirm `VITE_SQUAT_API_URL` points to `http://127.0.0.1:8000`.
- If report loading returns `404`, restart the upload API server.
- If the Vite URL uses a different port, make sure `SQUAT_ANALYSIS_CORS_ORIGINS` in `squat_analysis_api/.env` includes that origin.

