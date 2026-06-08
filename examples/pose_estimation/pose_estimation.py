import time

import cv2
import mediapipe as mp


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default webcam.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    previous_time = time.perf_counter()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )
                    status = "Pose detected"
                else:
                    status = "No pose detected"

                current_time = time.perf_counter()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time

                cv2.putText(
                    frame,
                    f"{status} | FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press q to quit",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("MediaPipe Pose Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
