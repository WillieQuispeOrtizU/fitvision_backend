# MediaPipe Pose Estimation

Simple webcam pose estimation demo using Python, MediaPipe, OpenCV, and `uv`.

## Prerequisites

- `uv` installed
- A working webcam
- Camera permission for your terminal app, especially on macOS

MediaPipe supports Python 3.9 through 3.12, so this project pins Python to `>=3.12,<3.13`.

## Setup

Create the virtual environment and install dependencies:

```bash
uv sync
```

If `uv` asks to download Python 3.12, allow it.

## Run

```bash
uv run python pose_estimation.py
```

A webcam window should open with pose landmarks drawn when a person is visible. Press `q` while the window is focused to quit.

## Troubleshooting

- If the camera does not open, close other apps that may be using it and try again.
- On macOS, grant camera access to your terminal in System Settings > Privacy & Security > Camera.
- If dependency installation fails because of Python version support, run `uv python install 3.12` and then run `uv sync` again.
