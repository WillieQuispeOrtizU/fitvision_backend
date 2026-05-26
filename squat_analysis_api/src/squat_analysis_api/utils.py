from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


def draw_rounded_rect(
    img: np.ndarray,
    rect_start: tuple[int, int],
    rect_end: tuple[int, int],
    corner_width: int,
    box_color: tuple[int, int, int],
) -> np.ndarray:
    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

    cv2.ellipse(img, (x1 + w, y1 + w), (w, w), 0, -90, -180, box_color, -1)
    cv2.ellipse(img, (x2 - w, y1 + w), (w, w), 0, 0, -90, box_color, -1)
    cv2.ellipse(img, (x1 + w, y2 - w), (w, w), 0, 90, 180, box_color, -1)
    cv2.ellipse(img, (x2 - w, y2 - w), (w, w), 0, 0, 90, box_color, -1)
    return img


def draw_dotted_line(
    frame: np.ndarray,
    lm_coord: np.ndarray,
    start: int,
    end: int,
    line_color: tuple[int, int, int],
) -> np.ndarray:
    for y_pos in range(start, end + 1, 8):
        cv2.circle(frame, (int(lm_coord[0]), y_pos), 2, line_color, -1, cv2.LINE_AA)
    return frame


def draw_text(
    img: np.ndarray,
    msg: str,
    *,
    width: int = 8,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    pos: tuple[int, int] = (0, 0),
    font_scale: float = 1,
    font_thickness: int = 2,
    text_color: tuple[int, int, int] = (0, 255, 0),
    text_color_bg: tuple[int, int, int] = (0, 0, 0),
    box_offset: tuple[int, int] = (20, 10),
) -> tuple[int, int]:
    x, y = pos
    text_w, text_h = cv2.getTextSize(msg, font, font_scale, font_thickness)[0]
    rec_start = (x - box_offset[0], y - box_offset[1])
    rec_end = (x + text_w - 25, y + text_h)

    draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)
    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    return text_w, text_h


def find_angle(p1: np.ndarray, p2: np.ndarray, ref_pt: np.ndarray) -> int:
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt
    denominator = np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref)
    if denominator == 0:
        return 0
    cos_theta = np.dot(p1_ref, p2_ref) / denominator
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return int((180 / np.pi) * theta)


def get_landmark_array(pose_landmark, key: int, frame_width: int, frame_height: int) -> np.ndarray:
    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)
    return np.array([denorm_x, denorm_y])


def get_landmark_features(
    kp_results,
    dict_features: dict,
    feature: str,
    frame_width: int,
    frame_height: int,
):
    if feature == "nose":
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    if feature in {"left", "right"}:
        side_features = dict_features[feature]
        return (
            get_landmark_array(kp_results, side_features["shoulder"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["elbow"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["wrist"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["hip"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["knee"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["ankle"], frame_width, frame_height),
            get_landmark_array(kp_results, side_features["foot"], frame_width, frame_height),
        )

    raise ValueError("feature needs to be either 'nose', 'left' or 'right'")


def get_mediapipe_pose(
    *,
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    return mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
