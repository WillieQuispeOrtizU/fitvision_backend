def get_thresholds_beginner() -> dict:
    return {
        "HIP_KNEE_VERT": {
            "NORMAL": (0, 32),
            "TRANS": (35, 65),
            "PASS": (70, 95),
        },
        "HIP_THRESH": [10, 50],
        "ANKLE_THRESH": 45,
        "KNEE_THRESH": [50, 70, 95],
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
        "CNT_FRAME_THRESH": 50,
    }


def get_thresholds_pro() -> dict:
    return {
        "HIP_KNEE_VERT": {
            "NORMAL": (0, 32),
            "TRANS": (35, 65),
            "PASS": (80, 95),
        },
        "HIP_THRESH": [15, 50],
        "ANKLE_THRESH": 30,
        "KNEE_THRESH": [50, 80, 95],
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
        "CNT_FRAME_THRESH": 50,
    }


def get_thresholds(mode: str) -> dict:
    if mode == "beginner":
        return get_thresholds_beginner()
    if mode == "pro":
        return get_thresholds_pro()
    raise ValueError(f"Unsupported mode: {mode}")
