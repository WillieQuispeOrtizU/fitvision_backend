from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from squat_analysis_api.reporting import RepAnalysis
from squat_analysis_api.utils import (
    draw_dotted_line,
    draw_text,
    find_angle,
    get_landmark_features,
)


@dataclass(frozen=True)
class SquatStats:
    correct: int
    incorrect: int


class SquatFrameAnalyzer:
    def __init__(self, thresholds: dict, flip_frame: bool = False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.colors = {
            "blue": (0, 127, 255),
            "red": (255, 50, 50),
            "green": (0, 255, 127),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "light_blue": (102, 204, 255),
            "light_green": (100, 233, 127),
        }
        self.dict_features = {
            "left": {
                "shoulder": 11,
                "elbow": 13,
                "wrist": 15,
                "hip": 23,
                "knee": 25,
                "ankle": 27,
                "foot": 31,
            },
            "right": {
                "shoulder": 12,
                "elbow": 14,
                "wrist": 16,
                "hip": 24,
                "knee": 26,
                "ankle": 28,
                "foot": 32,
            },
            "nose": 0,
        }
        # Timing is driven by the video timestamp passed into process(), not the
        # wall clock, so rep counting is deterministic and independent of how fast
        # frames are decoded. self._now holds the current frame's video time.
        self._now = 0.0
        self.state_tracker = {
            "state_seq": [],
            "start_inactive_time": 0.0,
            "start_inactive_time_front": 0.0,
            "INACTIVE_TIME": 0.0,
            "INACTIVE_TIME_FRONT": 0.0,
            "DISPLAY_TEXT": np.full((4,), False),
            "COUNT_FRAMES": np.zeros((4,), dtype=np.int64),
            "LOWER_HIPS": False,
            "INCORRECT_POSTURE": False,
            "prev_state": None,
            "curr_state": None,
            "SQUAT_COUNT": 0,
            "IMPROPER_SQUAT": 0,
        }
        self.feedback_id_map = {
            0: ("BEND BACKWARDS", 215, (0, 153, 255)),
            1: ("BEND FORWARD", 215, (0, 153, 255)),
            2: ("KNEE FALLING OVER TOE", 170, (255, 80, 80)),
            3: ("SQUAT TOO DEEP", 125, (255, 80, 80)),
        }
        self.feedback_issue_map = {
            0: "bend_backwards",
            1: "bend_forward",
            2: "knee_over_toe",
            3: "squat_too_deep",
        }
        self.issue_priority = (
            "squat_too_deep",
            "knee_over_toe",
            "bend_forward",
            "bend_backwards",
            "lower_hips",
        )
        self.completed_reps: list[RepAnalysis] = []
        self.key_frame_images: dict[int, dict[str, np.ndarray]] = {}
        self._current_rep: dict | None = None
        self._next_rep_id = 1
        self._internal_frame_index = 0

    @property
    def stats(self) -> SquatStats:
        # Derive the authoritative totals from the completed reps rather than the
        # live SQUAT_COUNT/IMPROPER_SQUAT counters. Those counters are reset to 0
        # by the on-screen inactivity logic (a live-overlay behavior), which would
        # otherwise corrupt the totals written to the report for a recorded video.
        correct = sum(1 for rep in self.completed_reps if rep.correct)
        incorrect = sum(1 for rep in self.completed_reps if not rep.correct)
        return SquatStats(correct=correct, incorrect=incorrect)

    @property
    def reps(self) -> list[RepAnalysis]:
        return list(self.completed_reps)

    def _get_state(self, knee_angle: int) -> str | None:
        knee = None
        if self.thresholds["HIP_KNEE_VERT"]["NORMAL"][0] <= knee_angle <= self.thresholds["HIP_KNEE_VERT"]["NORMAL"][1]:
            knee = 1
        elif self.thresholds["HIP_KNEE_VERT"]["TRANS"][0] <= knee_angle <= self.thresholds["HIP_KNEE_VERT"]["TRANS"][1]:
            knee = 2
        elif self.thresholds["HIP_KNEE_VERT"]["PASS"][0] <= knee_angle <= self.thresholds["HIP_KNEE_VERT"]["PASS"][1]:
            knee = 3
        return f"s{knee}" if knee else None

    def _update_state_sequence(self, state: str | None) -> None:
        if state == "s2":
            no_s3_first_s2 = "s3" not in self.state_tracker["state_seq"] and self.state_tracker["state_seq"].count("s2") == 0
            after_s3_second_s2 = "s3" in self.state_tracker["state_seq"] and self.state_tracker["state_seq"].count("s2") == 1
            if no_s3_first_s2 or after_s3_second_s2:
                self.state_tracker["state_seq"].append(state)
        elif state == "s3" and state not in self.state_tracker["state_seq"] and "s2" in self.state_tracker["state_seq"]:
            self.state_tracker["state_seq"].append(state)

    def _show_feedback(self, frame: np.ndarray) -> np.ndarray:
        if self.state_tracker["LOWER_HIPS"]:
            draw_text(
                frame,
                "LOWER YOUR HIPS",
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                text_color_bg=(255, 255, 0),
            )

        for idx in np.where(self.state_tracker["COUNT_FRAMES"])[0]:
            message, y_pos, color = self.feedback_id_map[int(idx)]
            draw_text(
                frame,
                message,
                pos=(30, y_pos),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=color,
            )
        return frame

    def _draw_counters(self, frame: np.ndarray, frame_width: int) -> None:
        draw_text(
            frame,
            f"CORRECT: {self.state_tracker['SQUAT_COUNT']}",
            pos=(int(frame_width * 0.68), 30),
            text_color=(255, 255, 230),
            font_scale=0.7,
            text_color_bg=(18, 185, 0),
        )
        draw_text(
            frame,
            f"INCORRECT: {self.state_tracker['IMPROPER_SQUAT']}",
            pos=(int(frame_width * 0.68), 80),
            text_color=(255, 255, 230),
            font_scale=0.7,
            text_color_bg=(221, 0, 0),
        )

    def process(
        self,
        frame: np.ndarray,
        pose,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> np.ndarray:
        if frame_index is None:
            frame_index = self._internal_frame_index
        if timestamp is None:
            timestamp = 0.0
        self._now = timestamp
        self._internal_frame_index = max(self._internal_frame_index, frame_index + 1)

        frame_height, frame_width, _ = frame.shape
        keypoints = pose.process(frame)

        if not keypoints.pose_landmarks:
            return self._process_missing_pose(frame, frame_width)

        landmarks = keypoints.pose_landmarks.landmark
        nose_coord = get_landmark_features(landmarks, self.dict_features, "nose", frame_width, frame_height)
        left = get_landmark_features(landmarks, self.dict_features, "left", frame_width, frame_height)
        right = get_landmark_features(landmarks, self.dict_features, "right", frame_width, frame_height)
        left_shldr, left_elbow, left_wrist, left_hip, left_knee, left_ankle, left_foot = left
        right_shldr, right_elbow, right_wrist, right_hip, right_knee, right_ankle, right_foot = right

        offset_angle = find_angle(left_shldr, right_shldr, nose_coord)
        if offset_angle > self.thresholds["OFFSET_THRESH"]:
            return self._process_bad_alignment(
                frame,
                frame_width,
                frame_height,
                offset_angle,
                nose_coord,
                left_shldr,
                right_shldr,
            )

        self.state_tracker["INACTIVE_TIME_FRONT"] = 0.0
        self.state_tracker["start_inactive_time_front"] = self._now

        dist_l_sh_foot = abs(left_foot[1] - left_shldr[1])
        dist_r_sh_foot = abs(right_foot[1] - right_shldr[1])
        if dist_l_sh_foot > dist_r_sh_foot:
            shldr, elbow, wrist, hip, knee, ankle, foot = left
            multiplier = -1
        else:
            shldr, elbow, wrist, hip, knee, ankle, foot = right
            multiplier = 1

        hip_vertical_angle = find_angle(shldr, np.array([hip[0], 0]), hip)
        knee_vertical_angle = find_angle(hip, np.array([knee[0], 0]), knee)
        ankle_vertical_angle = find_angle(knee, np.array([ankle[0], 0]), ankle)

        cv2.ellipse(frame, hip, (30, 30), 0, -90, -90 + multiplier * hip_vertical_angle, self.colors["white"], 3, self.linetype)
        cv2.ellipse(frame, knee, (20, 20), 0, -90, -90 - multiplier * knee_vertical_angle, self.colors["white"], 3, self.linetype)
        cv2.ellipse(frame, ankle, (30, 30), 0, -90, -90 + multiplier * ankle_vertical_angle, self.colors["white"], 3, self.linetype)
        draw_dotted_line(frame, hip, hip[1] - 80, hip[1] + 20, self.colors["blue"])
        draw_dotted_line(frame, knee, knee[1] - 50, knee[1] + 20, self.colors["blue"])
        draw_dotted_line(frame, ankle, ankle[1] - 50, ankle[1] + 20, self.colors["blue"])

        for start, end in ((shldr, elbow), (wrist, elbow), (shldr, hip), (knee, hip), (ankle, knee), (ankle, foot)):
            cv2.line(frame, start, end, self.colors["light_blue"], 4, self.linetype)
        for point in (shldr, elbow, wrist, hip, knee, ankle, foot):
            cv2.circle(frame, point, 7, self.colors["yellow"], -1, self.linetype)

        current_state = self._get_state(int(knee_vertical_angle))
        self.state_tracker["curr_state"] = current_state
        self._update_state_sequence(current_state)
        issues = self._update_feedback(hip_vertical_angle, knee_vertical_angle, ankle_vertical_angle)
        self._track_rep_frame(
            frame=frame,
            frame_index=frame_index,
            timestamp=timestamp,
            state=current_state,
            hip_angle=float(hip_vertical_angle),
            knee_angle=float(knee_vertical_angle),
            ankle_angle=float(ankle_vertical_angle),
            issues=issues,
        )
        completed_correct = self._update_counters(current_state)
        if completed_correct is not None:
            self._finish_current_rep(
                correct=completed_correct,
                frame=frame,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        display_inactivity = self._update_inactivity()

        hip_text_x = hip[0] + 10
        knee_text_x = knee[0] + 15
        ankle_text_x = ankle[0] + 10
        if self.flip_frame:
            frame = cv2.flip(frame, 1)
            hip_text_x = frame_width - hip[0] + 10
            knee_text_x = frame_width - knee[0] + 15
            ankle_text_x = frame_width - ankle[0] + 10

        if "s3" in self.state_tracker["state_seq"] or current_state == "s1":
            self.state_tracker["LOWER_HIPS"] = False

        self.state_tracker["COUNT_FRAMES"][self.state_tracker["DISPLAY_TEXT"]] += 1
        self._show_feedback(frame)

        if display_inactivity:
            self.state_tracker["start_inactive_time"] = self._now
            self.state_tracker["INACTIVE_TIME"] = 0.0

        cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_x, hip[1]), self.font, 0.6, self.colors["light_green"], 2, self.linetype)
        cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_x, knee[1] + 10), self.font, 0.6, self.colors["light_green"], 2, self.linetype)
        cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_x, ankle[1]), self.font, 0.6, self.colors["light_green"], 2, self.linetype)

        self._draw_counters(frame, frame_width)
        expired = self.state_tracker["COUNT_FRAMES"] > self.thresholds["CNT_FRAME_THRESH"]
        self.state_tracker["DISPLAY_TEXT"][expired] = False
        self.state_tracker["COUNT_FRAMES"][expired] = 0
        self.state_tracker["prev_state"] = current_state
        return frame

    def _update_counters(self, current_state: str | None) -> bool | None:
        if current_state != "s1":
            return None

        completed_correct = None
        if len(self.state_tracker["state_seq"]) == 3 and not self.state_tracker["INCORRECT_POSTURE"]:
            self.state_tracker["SQUAT_COUNT"] += 1
            completed_correct = True
        elif "s2" in self.state_tracker["state_seq"] and len(self.state_tracker["state_seq"]) == 1:
            self.state_tracker["IMPROPER_SQUAT"] += 1
            completed_correct = False
        elif self.state_tracker["INCORRECT_POSTURE"]:
            self.state_tracker["IMPROPER_SQUAT"] += 1
            completed_correct = False

        self.state_tracker["state_seq"] = []
        self.state_tracker["INCORRECT_POSTURE"] = False
        return completed_correct

    def _update_feedback(self, hip_angle: int, knee_angle: int, ankle_angle: int) -> list[str]:
        issues = []
        if hip_angle > self.thresholds["HIP_THRESH"][1]:
            self.state_tracker["DISPLAY_TEXT"][0] = True
            # Excessive torso lean is a genuine form fault, so it must fail the rep
            # (same as knee_over_toe / squat_too_deep below). Without this the lean
            # was detected and captioned but silently ignored by the verdict.
            self.state_tracker["INCORRECT_POSTURE"] = True
            issues.append(self.feedback_issue_map[0])
        elif hip_angle < self.thresholds["HIP_THRESH"][0] and self.state_tracker["state_seq"].count("s2") == 1:
            self.state_tracker["DISPLAY_TEXT"][1] = True
            issues.append(self.feedback_issue_map[1])

        if self.thresholds["KNEE_THRESH"][0] < knee_angle < self.thresholds["KNEE_THRESH"][1] and self.state_tracker["state_seq"].count("s2") == 1:
            self.state_tracker["LOWER_HIPS"] = True
            issues.append("lower_hips")
        elif knee_angle > self.thresholds["KNEE_THRESH"][2]:
            self.state_tracker["DISPLAY_TEXT"][3] = True
            self.state_tracker["INCORRECT_POSTURE"] = True
            issues.append(self.feedback_issue_map[3])

        if ankle_angle > self.thresholds["ANKLE_THRESH"]:
            self.state_tracker["DISPLAY_TEXT"][2] = True
            self.state_tracker["INCORRECT_POSTURE"] = True
            issues.append(self.feedback_issue_map[2])

        return issues

    def _track_rep_frame(
        self,
        *,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        state: str | None,
        hip_angle: float,
        knee_angle: float,
        ankle_angle: float,
        issues: list[str],
    ) -> None:
        if self._current_rep is None and state in {"s2", "s3"}:
            self._current_rep = {
                "rep_id": self._next_rep_id,
                "start_frame": frame_index,
                "start_time": timestamp,
                "bottom_frame": frame_index,
                "bottom_knee_angle": knee_angle,
                "issues": set(),
                "angles": {
                    "hip": [],
                    "knee": [],
                    "ankle": [],
                },
                "frames": {
                    "start": frame.copy(),
                    "bottom": frame.copy(),
                },
            }

        if self._current_rep is None:
            return

        self._current_rep["angles"]["hip"].append(hip_angle)
        self._current_rep["angles"]["knee"].append(knee_angle)
        self._current_rep["angles"]["ankle"].append(ankle_angle)
        self._current_rep["issues"].update(issues)

        if knee_angle >= self._current_rep["bottom_knee_angle"]:
            self._current_rep["bottom_knee_angle"] = knee_angle
            self._current_rep["bottom_frame"] = frame_index
            self._current_rep["frames"]["bottom"] = frame.copy()

        if state == "s1":
            self._current_rep["frames"]["finish"] = frame.copy()

    def _finish_current_rep(
        self,
        *,
        correct: bool,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> None:
        if self._current_rep is None:
            return

        self._current_rep["frames"].setdefault("finish", frame.copy())
        issue_set = set(self._current_rep["issues"])
        if correct:
            issue_set.discard("lower_hips")
        issues = sorted(
            issue_set,
            key=lambda issue: self.issue_priority.index(issue) if issue in self.issue_priority else len(self.issue_priority),
        )
        worst_issue = issues[0] if issues else None
        if worst_issue:
            self._current_rep["frames"]["worst_issue"] = self._current_rep["frames"].get("bottom", frame.copy())

        angle_summary = {}
        for name, values in self._current_rep["angles"].items():
            if values:
                angle_summary[name] = {
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
            else:
                angle_summary[name] = {"min": 0.0, "max": 0.0}

        rep_id = int(self._current_rep["rep_id"])
        self.completed_reps.append(
            RepAnalysis(
                rep_id=rep_id,
                start_frame=int(self._current_rep["start_frame"]),
                end_frame=frame_index,
                bottom_frame=int(self._current_rep["bottom_frame"]),
                start_time=float(self._current_rep["start_time"]),
                end_time=float(timestamp),
                correct=correct,
                issue_tags=issues,
                worst_issue=worst_issue,
                angles=angle_summary,
            )
        )
        self.key_frame_images[rep_id] = dict(self._current_rep["frames"])
        self._next_rep_id += 1
        self._current_rep = None

    def _update_inactivity(self) -> bool:
        if self.state_tracker["curr_state"] == self.state_tracker["prev_state"]:
            end_time = self._now
            self.state_tracker["INACTIVE_TIME"] += end_time - self.state_tracker["start_inactive_time"]
            self.state_tracker["start_inactive_time"] = end_time
            if self.state_tracker["INACTIVE_TIME"] >= self.thresholds["INACTIVE_THRESH"]:
                self.state_tracker["SQUAT_COUNT"] = 0
                self.state_tracker["IMPROPER_SQUAT"] = 0
                return True
        else:
            self.state_tracker["start_inactive_time"] = self._now
            self.state_tracker["INACTIVE_TIME"] = 0.0
        return False

    def _process_bad_alignment(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        offset_angle: int,
        nose_coord: np.ndarray,
        left_shldr: np.ndarray,
        right_shldr: np.ndarray,
    ) -> np.ndarray:
        end_time = self._now
        self.state_tracker["INACTIVE_TIME_FRONT"] += end_time - self.state_tracker["start_inactive_time_front"]
        self.state_tracker["start_inactive_time_front"] = end_time

        if self.state_tracker["INACTIVE_TIME_FRONT"] >= self.thresholds["INACTIVE_THRESH"]:
            self.state_tracker["SQUAT_COUNT"] = 0
            self.state_tracker["IMPROPER_SQUAT"] = 0
            self.state_tracker["INACTIVE_TIME_FRONT"] = 0.0
            self.state_tracker["start_inactive_time_front"] = self._now

        cv2.circle(frame, nose_coord, 7, self.colors["white"], -1)
        cv2.circle(frame, left_shldr, 7, self.colors["yellow"], -1)
        cv2.circle(frame, right_shldr, 7, self.colors["magenta"], -1)

        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        self._draw_counters(frame, frame_width)
        draw_text(
            frame,
            "CAMERA NOT ALIGNED PROPERLY!!!",
            pos=(30, frame_height - 60),
            text_color=(255, 255, 230),
            font_scale=0.65,
            text_color_bg=(255, 153, 0),
        )
        draw_text(
            frame,
            f"OFFSET ANGLE: {offset_angle}",
            pos=(30, frame_height - 30),
            text_color=(255, 255, 230),
            font_scale=0.65,
            text_color_bg=(255, 153, 0),
        )
        self.state_tracker["start_inactive_time"] = self._now
        self.state_tracker["INACTIVE_TIME"] = 0.0
        self.state_tracker["prev_state"] = None
        self.state_tracker["curr_state"] = None
        return frame

    def _process_missing_pose(self, frame: np.ndarray, frame_width: int) -> np.ndarray:
        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        end_time = self._now
        self.state_tracker["INACTIVE_TIME"] += end_time - self.state_tracker["start_inactive_time"]
        if self.state_tracker["INACTIVE_TIME"] >= self.thresholds["INACTIVE_THRESH"]:
            self.state_tracker["SQUAT_COUNT"] = 0
            self.state_tracker["IMPROPER_SQUAT"] = 0
            self.state_tracker["INACTIVE_TIME"] = 0.0
        self.state_tracker["start_inactive_time"] = end_time

        self._draw_counters(frame, frame_width)
        self.state_tracker["prev_state"] = None
        self.state_tracker["curr_state"] = None
        self.state_tracker["INACTIVE_TIME_FRONT"] = 0.0
        self.state_tracker["INCORRECT_POSTURE"] = False
        self.state_tracker["DISPLAY_TEXT"] = np.full((4,), False)
        self.state_tracker["COUNT_FRAMES"] = np.zeros((4,), dtype=np.int64)
        self.state_tracker["start_inactive_time_front"] = self._now
        return frame
