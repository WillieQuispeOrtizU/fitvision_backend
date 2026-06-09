def get_thresholds_beginner() -> dict:
    return {
        "HIP_KNEE_VERT": {
            # TRANS and PASS are contiguous: the old (65, 70) gap mapped legitimate
            # bottom-of-squat knee angles to "no state", so valid-depth reps were
            # scored as too shallow. PASS lowered to 60 to match real squat depth.
            "NORMAL": (0, 32),
            "TRANS": (35, 60),
            "PASS": (60, 95),
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
            # Contiguous bands (was a 65-80 dead zone). Pro stays stricter than
            # beginner: requires a deeper knee angle (>=65) to register full depth.
            "NORMAL": (0, 32),
            "TRANS": (35, 65),
            "PASS": (65, 95),
        },
        "HIP_THRESH": [15, 50],
        # Knee-over-toe flagged on shin lean > 40 deg. The old 30 was below the
        # natural forward shin travel of valid deep squats (31-40 deg here), so it
        # false-flagged good reps; stays stricter than beginner (45).
        "ANKLE_THRESH": 40,
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
