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
- FFmpeg installed with both `ffmpeg` and `ffprobe` available on `PATH`
- PostgreSQL 16-compatible database provisioned with `schema.sql`
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

Before OpenCV and MediaPipe analyze a video, the backend uses `ffprobe` and
`ffmpeg` to physically normalize mobile rotation metadata when necessary. The
original upload is preserved. If either tool is unavailable or normalization
fails, processing falls back to the original file and logs a warning.

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

Authentication, profile, job, and evaluation endpoints require the existing
PostgreSQL schema and the following environment variables:

```env
DATABASE_URL=postgresql+psycopg://fitvision:password@127.0.0.1:5432/fitvision
AUTH_SESSION_TTL_HOURS=720
```

`DATABASE_URL` is required for authentication, job ownership, evaluation
history, and progress persistence. `AUTH_SESSION_TTL_HOURS` is optional and
defaults to 720 hours. The application validates connectivity but does not call
`Base.metadata.create_all()` or otherwise create production tables.

Endpoints:

- `GET /api/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout` (Bearer required)
- `GET /api/users/me` (Bearer required)
- `PATCH /api/users/me` (Bearer required)
- `POST /api/jobs` (Bearer required)
- `GET /api/jobs/{job_id}` (Bearer required; owner only)
- `GET /api/jobs/{job_id}/report` (Bearer required; owner only)
- `GET /api/evaluations` (Bearer required)
- `GET /api/evaluations/{evaluation_id}` (Bearer required; owner only)
- `GET /api/jobs/{job_id}/files/{filename}` (public)

Bearer sessions use an opaque token returned only to the client. PostgreSQL
stores its SHA-256 digest in `auth_session.token_hash`; passwords are stored as
Argon2id hashes.

Example authentication flow:

```bash
API_URL=http://127.0.0.1:8000

curl -X POST "$API_URL/api/auth/register" \
  -H 'Content-Type: application/json' \
  -d '{"email":"usuario@email.com","password":"password","full_name":"Nombre","birth_date":"2002-01-01","height_cm":173,"weight_kg":65,"training_level":"beginner"}'

curl -X POST "$API_URL/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{"email":"usuario@email.com","password":"password"}'

curl "$API_URL/api/users/me" \
  -H "Authorization: Bearer $TOKEN"

curl -X POST "$API_URL/api/auth/logout" \
  -H "Authorization: Bearer $TOKEN"
```

### Password recovery (local development)

Password recovery is intentionally disabled until its database migration and
email settings are provided. Do not expose this flow through the current
cleartext HTTP deployment: login credentials, recovery tokens, and codes must
travel over HTTPS in any public environment.

Apply the migration to a **local/development** PostgreSQL database before
enabling the endpoints (do not run it against production without a reviewed
deployment plan and backup):

```bash
psql "$PSQL_DATABASE_URL" \
  -f migrations/20261113_add_password_reset_request.sql
```

`PSQL_DATABASE_URL` in that command means a libpq-compatible DSN such as
`postgresql://user:password@127.0.0.1:5432/fitvision`. It is deliberately
separate from the application's SQLAlchemy `DATABASE_URL`, which may contain
the `postgresql+psycopg://` driver suffix that the `psql` CLI does not accept.

Configure a server-only HMAC secret and an SMTP provider in `.env`; the real
file is Git-ignored and credentials must never be committed. Relevant values
and safe defaults are documented in `.env.example`. For Gmail, enable two-step
verification, create an application password, and use:

```env
PASSWORD_RESET_EMAIL_PROVIDER=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURITY=starttls
SMTP_USERNAME=sender@example.com
SMTP_PASSWORD=<application-password>
SMTP_FROM_EMAIL=sender@example.com
```

Use a unique random `PASSWORD_RESET_PEPPER` of at least 32 bytes. A convenient
PowerShell generator is:

```powershell
[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

The flow exposes three public endpoints:

- `POST /api/auth/password-reset/request` with `{ "email": "..." }` returns
  `202` and the same generic message for known and unknown addresses.
- `POST /api/auth/password-reset/verify` with `request_token` plus the six-digit
  `code` returns a short-lived one-time `reset_token`.
- `POST /api/auth/password-reset/complete` with `reset_token` and
  `new_password` changes the Argon2id hash and revokes every previous session.

Codes and reset tokens expire after ten minutes by default. Requests enforce a
60-second resend cooldown, hourly per-address/per-IP limits, and at most five
failed code attempts. Only hashes/fingerprints are persisted. SMTP delivery is
performed after the generic response; a delivery failure invalidates that
challenge without revealing whether the address exists.

The current local implementation uses FastAPI `BackgroundTasks`, not a durable
mail queue. A process stop immediately after a `202` response can therefore
prevent delivery; the user may retry after the cooldown. Before a public/high-
availability deployment, move delivery to a durable worker and schedule a
retention job for old expired/used recovery rows.

When Nginx is used, configure it to overwrite `X-Forwarded-For`, then list only
the trusted proxy address (typically `127.0.0.1,::1`) in
`PASSWORD_RESET_TRUSTED_PROXY_IPS`. Never trust forwarding headers from an
arbitrary public peer.

For a local end-to-end check, request a code for a non-production test account,
confirm that it arrives, verify it once, complete the reset, and confirm that
the new password logs in while the old password and all earlier Bearer sessions
are rejected. No real SMTP credentials are required by the automated suite;
it injects an in-memory sender.

Upload a squat video with the token returned by register or login:

```bash
TOKEN='TOKEN_RETURNED_BY_REGISTER_OR_LOGIN'

curl -X POST "$API_URL/api/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -F 'video=@input_video/example.mp4' \
  -F 'mode=beginner'
```

List the authenticated user's evaluations, newest first, and retrieve one
evaluation including its complete persisted report:

```bash
curl "$API_URL/api/evaluations" \
  -H "Authorization: Bearer $TOKEN"

EVALUATION_ID=1
curl "$API_URL/api/evaluations/$EVALUATION_ID" \
  -H "Authorization: Bearer $TOKEN"
```

Each evaluation response also includes `xp_earned`, calculated as the sum of
`xp_transaction.amount` rows linked to that evaluation. It includes the base
completion reward and any challenge rewards triggered by that completion.

Retrieve the authenticated user's complete PostgreSQL-backed gamification
state:

```bash
curl "$API_URL/api/gamification" \
  -H "Authorization: Bearer $TOKEN"
```

The response contains `progress`, `achievements`, and `challenges`. XP, level,
streaks, counters, unlock timestamps, challenge status, and challenge rewards
are authoritative backend values. Calendar streaks use `America/Lima` days.

## Gamification catalogs and historical backfill

The canonical achievement/challenge catalog is upserted by stable `code`; an
existing row keeps its database ID and user progress is never deleted by the
seed. Seed it explicitly during deployment:

```bash
uv run --extra assistant fitvision-gamification seed
```

Existing completed evaluations can be replayed without changing their report,
classification, media URLs, or timestamps. Selection is mandatory so the
operation can never target the production database accidentally without an
explicit scope:

Before a non-dry-run backfill, stop the API process and every analysis worker
after allowing in-flight jobs to finish. Take a PostgreSQL backup, then run the
seed/backfill while those processes remain stopped. This prevents a worker
from completing a new evaluation between the historical snapshot and replay.

```bash
# Inspect one user without persisting changes.
uv run --extra assistant fitvision-gamification backfill --user-id 123 --dry-run

# Rebuild one user.
uv run --extra assistant fitvision-gamification backfill --user-id 123

# One-time deployment backfill for every user.
uv run --extra assistant fitvision-gamification backfill --all
```

The backfill uses one transaction per user. It deletes and reconstructs only
derived gamification rows (`xp_transaction`, `user_achievement`,
`user_challenge`, and aggregate `user_progress` values) from completed
evaluations in chronological order. It is safe to rerun and does not execute
automatically when the API starts.

Job creation, status, and report access are tied to the authenticated user. A
user receives `404` when requesting a job or evaluation owned by another user.
The files endpoint deliberately remains public for now because Flutter video
playback consumes the returned media URLs without attaching a Bearer header.
Existing URL formats such as
`/api/jobs/{job_id}/files/{filename}` remain unchanged.

## Test and deploy

Install exactly the locked API dependencies and run the complete test suite:

```bash
uv sync --frozen --extra assistant
uv run --extra assistant python -m unittest discover -s tests -v
```

On the server, provide `DATABASE_URL` through the existing process manager and
restart that process after updating the checkout. The repository does not
contain a systemd/Nginx service definition, so its external service name is not
assumed here.

Evaluation persistence reuses the existing `evaluation` and `user_progress`
tables plus the existing gamification tables from `schema.sql`. It requires no
schema migration, new dependency, or new environment variable.

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

